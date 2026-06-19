"""
FPL 新5大指標 × 既存指標 完全比較評価コード
=============================================
WCアプリと同様の評価方法をFPLデータで実施:
  - 各指標とFPL既存指標の Pearson/Spearman 相関
  - FPL高得点予測 AUC（Stratified 5-fold）
  - 複合モデルAUC
  - 指標間相関ヒートマップ
  - ポジション別分布

起動方法: python fpl_eval.py
必要ライブラリ: pip install requests pandas numpy scipy scikit-learn matplotlib seaborn
"""

import io, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# 設定
# =========================================================
SEASON   = "2024-25"
MIN_MIN  = 900     # 最低出場分数（評価の安定性のため高め）
FULL_MIN = 3420.0  # 38試合 × 90分

# =========================================================
# データ取得
# =========================================================
VAASTAV  = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{SEASON}"
HEADERS  = {"User-Agent": "Mozilla/5.0 Chrome/124.0.0.0"}

def fetch(url):
    for _ in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2)
    return None

def load_data():
    print(f"📥 vaastav {SEASON} データ取得中...")
    r_p = fetch(f"{VAASTAV}/players_raw.csv")
    r_t = fetch(f"{VAASTAV}/teams.csv")

    if r_p is None:
        # ローカルファイルを試す
        import os
        local = f"players_raw_{SEASON.replace('-','_')}.csv"
        if os.path.exists(local):
            df = pd.read_csv(local)
            print(f"  ローカルファイル使用: {local}")
        else:
            raise RuntimeError(
                f"データ取得失敗。以下のURLからCSVをダウンロードして "
                f"'{local}' として保存してください:\n"
                f"  {VAASTAV}/players_raw.csv"
            )
    else:
        df = pd.read_csv(io.StringIO(r_p.text))
        print(f"  → 全選手数: {len(df)}")

    team_map = {}
    if r_t:
        df_t = pd.read_csv(io.StringIO(r_t.text))
        if "id" in df_t.columns and "name" in df_t.columns:
            team_map = dict(zip(df_t["id"], df_t["name"]))

    return df, team_map

# =========================================================
# 整形
# =========================================================
POS_MAP = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

def prepare(df_raw, team_map):
    df = df_raw.copy()
    if "web_name" in df.columns:
        df["player_name"] = df["web_name"]
    df["position"]  = df.get("element_type", pd.Series()).map(POS_MAP).fillna("UNK")
    df["team_name"] = df.get("team", pd.Series()).map(team_map).fillna("Unknown")

    num_cols = [
        "minutes","goals_scored","assists","clean_sheets","goals_conceded",
        "saves","yellow_cards","red_cards","bonus","bps","total_points","now_cost",
        "expected_goals","expected_assists","expected_goal_involvements",
        "expected_goals_conceded","influence","creativity","threat","ict_index",
        "tackles","recoveries","clearances_blocks_interceptions",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    df["price_m"] = df["now_cost"] / 10.0
    return df

# =========================================================
# 新5大指標 算出
# =========================================================
def compute(df_raw, min_minutes=MIN_MIN):
    df = df_raw[df_raw["minutes"] >= min_minutes].copy().reset_index(drop=True)
    if df.empty:
        return df

    p90 = (df["minutes"] / 90).clip(lower=1)

    # ── ① 攻撃プロセス ──────────────────────────────────
    pos_atk = {"GK":0.3,"DEF":0.6,"MID":1.2,"FWD":1.4}
    xA_p90    = df["expected_assists"] / p90
    xGI_p90   = df["expected_goal_involvements"] / p90
    cre_max   = df["creativity"].max()
    cre_n     = df["creativity"] / (cre_max if cre_max > 0 else 1)
    df["①攻撃プロセス_raw"] = (
        xA_p90 * 2.0 + cre_n * 1.5 + xGI_p90 * 0.5
    ) * df["position"].map(pos_atk).fillna(1.0)

    # ── ② 守備プロセス v3 ────────────────────────────────
    pos_def = {"GK":1.2,"DEF":1.1,"MID":0.8,"FWD":0.4}
    saves_p90   = df["saves"] / p90
    gc_p90      = df["goals_conceded"] / p90
    cs_w        = df["clean_sheets"] * (df["minutes"] / FULL_MIN)
    def_act_p90 = (
        df["tackles"] + df["recoveries"] + df["clearances_blocks_interceptions"]
    ) / p90
    is_gk  = df["position"] == "GK"
    is_def = df["position"] == "DEF"
    is_mid = df["position"] == "MID"

    df["②守備プロセス_raw"] = np.where(
        is_gk,
        saves_p90 * 2.5 + cs_w * 8.0 - gc_p90 * 0.5,
        np.where(
            is_def,
            def_act_p90 * 0.25 + cs_w * 3.0 - gc_p90 * 0.6,
            np.where(is_mid, def_act_p90 * 0.20, def_act_p90 * 0.08)
        )
    ) * df["position"].map(pos_def).fillna(0.7)

    # ── ③ 得点近接 ──────────────────────────────────────
    xG_p90   = df["expected_goals"] / p90
    thr_max  = df["threat"].max()
    thr_n    = df["threat"] / (thr_max if thr_max > 0 else 1)
    g_p90    = df["goals_scored"] / p90
    df["③得点近接_raw"] = xG_p90 * 3.0 + thr_n * 2.0 + g_p90 * 1.0

    # ── ④ 失点近接 ──────────────────────────────────────
    df["④失点近接_raw"] = np.where(
        is_gk,
        saves_p90 * 2.0 + df["clean_sheets"] * 0.5 - df["red_cards"] * 2.0,
        np.where(
            is_def,
            df["clean_sheets"] * 0.8 - gc_p90 * 0.5 - df["red_cards"] * 2.0,
            df["clean_sheets"] * 0.3 - df["red_cards"] * 1.0,
        )
    )

    # ── ⑤ Luck ──────────────────────────────────────────
    df["⑤得点Luck"] = df["goals_scored"] - df["expected_goals"]
    df["⑤守備Luck"] = df["expected_goals_conceded"] - df["goals_conceded"]
    df["⑤Luck合計"] = df["⑤得点Luck"] + df["⑤守備Luck"]

    # ── Z標準化（①② のみ）──────────────────────────────
    for raw, norm in [
        ("①攻撃プロセス_raw","①攻撃プロセス"),
        ("②守備プロセス_raw","②守備プロセス"),
    ]:
        mu, sd = df[raw].mean(), df[raw].std()
        df[norm] = (df[raw] - mu) / (sd if sd > 0 else 1)

    df["③得点近接"]             = df["③得点近接_raw"]
    df["④失点近接"]             = df["④失点近接_raw"]
    df["総合プロセス(①+②)"]    = df["①攻撃プロセス"] + df["②守備プロセス"]
    df["総合クリティカル(③+④)"] = df["③得点近接"]  + df["④失点近接"]

    return df

# =========================================================
# 評価 A: 各指標 × 既存指標 相関
# =========================================================
def evaluate_correlations(df):
    NEW_METRICS = [
        ("①攻撃プロセス",          "① Attack Process"),
        ("②守備プロセス",          "② Defense Process"),
        ("総合プロセス(①+②)",     "① + ② Process Total"),
        ("③得点近接",              "③ Goal Threat"),
        ("④失点近接",              "④ Save Contribution"),
        ("総合クリティカル(③+④)", "③ + ④ Critical Total"),
        ("⑤得点Luck",             "⑤ Goal Luck"),
        ("⑤守備Luck",             "⑤ Defense Luck"),
    ]

    EXISTING = [
        ("expected_goals",              "xG"),
        ("expected_assists",            "xA"),
        ("expected_goal_involvements",  "xGI"),
        ("expected_goals_conceded",     "xGC（被xG）"),
        ("influence",                   "Influence"),
        ("creativity",                  "Creativity"),
        ("threat",                      "Threat"),
        ("ict_index",                   "ICT Index"),
        ("goals_scored",                "Goals"),
        ("assists",                     "Assists"),
        ("clean_sheets",                "Clean Sheets"),
        ("goals_conceded",              "Goals Conceded"),
        ("saves",                       "Saves"),
        ("bonus",                       "Bonus Points"),
        ("total_points",                "FPL Total Points"),
    ]

    print("\n" + "="*72)
    print("A. 新指標 × 既存指標 Pearson / Spearman 相関")
    print("="*72)

    results = {}
    for new_col, new_label in NEW_METRICS:
        if new_col not in df.columns: continue
        row = {}
        for ex_col, ex_label in EXISTING:
            if ex_col not in df.columns: continue
            sub = df[[new_col, ex_col]].dropna()
            if len(sub) < 10: continue
            r_p, p_p = pearsonr(sub[new_col], sub[ex_col])
            r_s, _   = spearmanr(sub[new_col], sub[ex_col])
            sig = "***" if p_p<.001 else ("**" if p_p<.01 else ("*" if p_p<.05 else ""))
            row[ex_label] = (round(r_p,3), sig, round(r_s,3))
        results[new_label] = row

    # 表示: 新指標ごとに上位相関を表示
    for new_label, row in results.items():
        print(f"\n  [{new_label}]")
        print(f"  {'既存指標':<25} {'Pearson r':>10} {'':>5} {'Spearman ρ':>12}")
        print(f"  {'-'*55}")
        for ex_label, (r_p, sig, r_s) in sorted(row.items(), key=lambda x:-abs(x[1][0])):
            print(f"  {ex_label:<25} {r_p:>+10.3f}{sig:<5} {r_s:>+12.3f}")

    return results

# =========================================================
# 評価 B: FPL高得点予測 AUC
# =========================================================
def evaluate_auc(df):
    print("\n" + "="*72)
    print("B. FPL高得点予測 AUC（Stratified 5-fold LogReg）")
    print("   ターゲット: FPL総得点 上位50% = 1, 下位50% = 0")
    print("="*72)

    y = (df["total_points"] >= df["total_points"].median()).astype(int)
    n_sp = min(5, max(2, len(df)//10))
    cv   = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=42)

    ALL_METRICS = [
        # 新指標
        ("総合プロセス(①+②)",    "🆕 ①+② プロセス合計",  True),
        ("①攻撃プロセス",         "🆕 ① 攻撃プロセス",     True),
        ("②守備プロセス",         "🆕 ② 守備プロセス",     True),
        ("③得点近接",             "🆕 ③ 得点近接",          True),
        ("④失点近接",             "🆕 ④ 失点近接",          True),
        ("総合クリティカル(③+④)","🆕 ③+④ クリティカル",   True),
        ("⑤得点Luck",             "🆕 ⑤ 得点Luck",          True),
        ("⑤守備Luck",             "🆕 ⑤ 守備Luck",          True),
        # 既存
        ("expected_goals",             "📌 xG",               False),
        ("expected_assists",           "📌 xA",               False),
        ("expected_goal_involvements", "📌 xGI",              False),
        ("ict_index",                  "📌 ICT Index",         False),
        ("threat",                     "📌 Threat",            False),
        ("creativity",                 "📌 Creativity",        False),
        ("influence",                  "📌 Influence",         False),
        ("clean_sheets",               "📌 Clean Sheets",      False),
        ("saves",                      "📌 Saves",             False),
        ("goals_scored",               "📌 Goals",             False),
        ("assists",                    "📌 Assists",           False),
    ]

    auc_results = []
    for col, label, is_new in ALL_METRICS:
        if col not in df.columns: continue
        X = StandardScaler().fit_transform(df[[col]].fillna(0))
        try:
            auc = cross_val_score(
                LogisticRegression(max_iter=1000), X, y,
                cv=cv, scoring="roc_auc"
            ).mean()
            auc_results.append((label, auc, is_new, col))
        except Exception:
            pass

    auc_results.sort(key=lambda x: -x[1])
    print(f"\n  {'指標':<30} {'AUC':>7}  {'':>40}")
    print(f"  {'-'*70}")
    for label, auc, is_new, col in auc_results:
        bar = "█" * int(auc * 40)
        marker = "◀" if is_new else ""
        print(f"  {label:<30} {auc:.3f}  {bar} {marker}")

    # 複合モデル
    print(f"\n  [複合モデル AUC]")
    combos = {
        "新①+② のみ":                       ["総合プロセス(①+②)"],
        "新③+④ のみ":                       ["総合クリティカル(③+④)"],
        "新⑤ のみ":                          ["⑤得点Luck","⑤守備Luck"],
        "新①〜④ 全部":                      ["総合プロセス(①+②)","総合クリティカル(③+④)"],
        "新①〜⑤ 全部":                      ["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck"],
        "xGI + ICT（既存ベースライン）":     ["expected_goal_involvements","ict_index"],
        "xG + xA + CS（既存）":              ["expected_goals","expected_assists","clean_sheets"],
        "新全指標 + xGI + ICT（ハイブリッド）": ["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck","expected_goal_involvements","ict_index"],
    }
    for name, cols in combos.items():
        valid = [c for c in cols if c in df.columns]
        if not valid: continue
        X = StandardScaler().fit_transform(df[valid].fillna(0))
        try:
            auc = cross_val_score(
                LogisticRegression(max_iter=1000), X, y,
                cv=cv, scoring="roc_auc"
            ).mean()
            bar = "█" * int(auc * 40)
            print(f"  {name:<42} {auc:.3f}  {bar}")
        except Exception:
            pass

    return auc_results

# =========================================================
# 評価 C: ポジション別分布（②の妥当性確認）
# =========================================================
def evaluate_position_distribution(df):
    print("\n" + "="*72)
    print("C. ② 守備プロセス ポジション別 TOP10 確認")
    print("   （GK/DEFが上位に、FW/MFの攻撃的選手が下位にいれば妥当）")
    print("="*72)

    for pos in ["GK","DEF","MID","FWD"]:
        sub = df[df["position"]==pos].nlargest(5,"②守備プロセス")
        print(f"\n  [{pos} TOP5]")
        if "player_name" in sub.columns:
            for _, r in sub.iterrows():
                print(f"    {r['player_name']:<20} ②={r['②守備プロセス']:>+6.2f}  "
                      f"CS={r.get('clean_sheets',0):.0f}  saves={r.get('saves',0):.0f}  "
                      f"min={r['minutes']:.0f}")

    print(f"\n  [② 最下位10（FW中心になるはず）]")
    bot = df.nsmallest(10,"②守備プロセス")
    if "player_name" in bot.columns:
        for _, r in bot.iterrows():
            print(f"    {r['player_name']:<20} {r['position']:>3}  ②={r['②守備プロセス']:>+6.2f}")

# =========================================================
# 可視化（WCアプリと同様の6枚構成）
# =========================================================
def visualize(df, auc_results):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"FPL 新5大指標 × 既存指標 完全比較評価\n"
        f"{SEASON}  |  {MIN_MIN}分以上出場選手  (n={len(df)})",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ① vs xGI
    ax1 = fig.add_subplot(gs[0,0])
    ax1.scatter(df["expected_goal_involvements"], df["①攻撃プロセス"],
                alpha=0.4, s=15, c="steelblue")
    ax1.set_xlabel("xGI (Existing)")
    ax1.set_ylabel("① Attack Process (New)")
    ax1.set_title("① Attack Process vs xGI")
    if "expected_goal_involvements" in df.columns:
        r, _ = pearsonr(df["①攻撃プロセス"], df["expected_goal_involvements"])
        ax1.text(0.05,0.95,f"r={r:.3f}",transform=ax1.transAxes,va="top",color="steelblue",fontsize=11)
    ax1.axhline(0,color="gray",ls="--",lw=0.7)

    # ② ポジション別箱ひげ
    ax2 = fig.add_subplot(gs[0,1])
    pos_order = ["GK","DEF","MID","FWD"]
    pos_data  = [df[df["position"]==p]["②守備プロセス"].dropna() for p in pos_order]
    pos_labels = [f"{p}\n(n={len(d)})" for p,d in zip(pos_order,pos_data)]
    bp = ax2.boxplot([d for d in pos_data if len(d)>0],
                     labels=[l for l,d in zip(pos_labels,pos_data) if len(d)>0],
                     patch_artist=True)
    colors = ["#F59E0B","#3B82F6","#8B5CF6","#EF4444"]
    for patch,color in zip(bp["boxes"],colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax2.axhline(0,color="gray",ls="--",lw=0.7)
    ax2.set_title("② Defense Process by Position\n(GK/DEF should be higher)")
    ax2.set_ylabel("z-score")

    # ③ vs Goals
    ax3 = fig.add_subplot(gs[0,2])
    ax3.scatter(df["goals_scored"], df["③得点近接"],
                alpha=0.4, s=15, c="coral")
    ax3.set_xlabel("Goals Scored (Existing)")
    ax3.set_ylabel("③ Goal Threat (New)")
    ax3.set_title("③ Goal Threat vs Goals Scored")
    r, _ = pearsonr(df["③得点近接"], df["goals_scored"])
    ax3.text(0.05,0.95,f"r={r:.3f}",transform=ax3.transAxes,va="top",color="coral",fontsize=11)

    # ④ vs Clean Sheets
    ax4 = fig.add_subplot(gs[1,0])
    ax4.scatter(df["clean_sheets"], df["④失点近接"],
                alpha=0.4, s=15, c="green")
    ax4.set_xlabel("Clean Sheets (Existing)")
    ax4.set_ylabel("④ Save Contribution (New)")
    ax4.set_title("④ Save Contribution vs Clean Sheets")
    r, _ = pearsonr(df["④失点近接"], df["clean_sheets"])
    ax4.text(0.05,0.95,f"r={r:.3f}",transform=ax4.transAxes,va="top",color="green",fontsize=11)

    # AUCバー
    ax5 = fig.add_subplot(gs[1,1])
    top_n = auc_results[:14]
    labels_a = [x[0] for x in top_n][::-1]
    values_a = [x[1] for x in top_n][::-1]
    colors_a = ["#00A651" if x[2] else "#94A3B8" for x in top_n][::-1]
    bars = ax5.barh(labels_a, values_a, color=colors_a, edgecolor="white", linewidth=0.5)
    ax5.axvline(0.5, color="#FF4B4B", ls="--", lw=1.5, label="Random (0.5)")
    ax5.set_xlabel("AUC")
    ax5.set_title("FPL High Score Prediction AUC\n(green=new, gray=existing)", fontsize=9)
    ax5.set_xlim(0.3, 1.0)
    ax5.grid(axis="x", color="#CBD5E1", lw=0.5)
    for bar, val in zip(bars, values_a):
        ax5.text(val+.005, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=7)
    ax5.legend(fontsize=7)

    # 相関ヒートマップ
    ax6 = fig.add_subplot(gs[1,2])
    hm_cols = [
        "①攻撃プロセス","②守備プロセス","③得点近接","④失点近接","⑤得点Luck",
        "expected_goals","expected_assists","ict_index","threat","creativity",
        "clean_sheets","saves","total_points",
    ]
    hm_cols = [c for c in hm_cols if c in df.columns]
    lbl = {
        "①攻撃プロセス":"①Atk","②守備プロセス":"②Def","③得点近接":"③GThr",
        "④失点近接":"④Save","⑤得点Luck":"⑤Luck","expected_goals":"xG",
        "expected_assists":"xA","ict_index":"ICT","threat":"Threat",
        "creativity":"Creat","clean_sheets":"CS","saves":"Saves","total_points":"FPLPts",
    }
    sns.heatmap(
        df[hm_cols].rename(columns=lbl).corr(),
        annot=True, fmt=".2f", cmap="coolwarm", center=0,
        ax=ax6, annot_kws={"size":7}, linewidths=0.3, square=False
    )
    ax6.set_title("Correlation Matrix\nNew Metrics vs Existing FPL Metrics", fontsize=9)
    ax6.tick_params(axis="x", labelsize=7, rotation=45)
    ax6.tick_params(axis="y", labelsize=7, rotation=0)

    plt.savefig(f"fpl_eval_{SEASON}.png", dpi=150, bbox_inches="tight")
    print(f"\n📊 グラフ保存: fpl_eval_{SEASON}.png")
    plt.show()

# =========================================================
# メイン
# =========================================================
def main():
    df_raw, team_map = load_data()
    df_prep = prepare(df_raw, team_map)
    df      = compute(df_prep, MIN_MIN)
    print(f"✅ 有効選手数（{MIN_MIN}分以上）: {len(df)}")

    if df.empty:
        print("データが不十分です。MIN_MINを下げてください。")
        return

    # TOP20 表示
    print(f"\n🏆 総合プロセス(①+②) TOP20:")
    top_cols = ["player_name","team_name","position","minutes",
                "①攻撃プロセス","②守備プロセス","総合プロセス(①+②)",
                "③得点近接","④失点近接","⑤得点Luck","total_points"]
    top_cols = [c for c in top_cols if c in df.columns]
    print(df.nlargest(20,"総合プロセス(①+②)")[top_cols]
          .to_string(index=False, float_format="{:.3f}".format))

    evaluate_correlations(df)
    auc_results = evaluate_auc(df)
    evaluate_position_distribution(df)
    visualize(df, auc_results)

    # CSV出力
    out = f"fpl_metrics_{SEASON}.csv"
    df[top_cols].sort_values("総合プロセス(①+②)", ascending=False).to_csv(out, index=False)
    print(f"\n📄 CSV保存: {out}")

if __name__ == "__main__":
    main()
