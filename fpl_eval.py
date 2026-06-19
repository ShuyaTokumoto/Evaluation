"""
FPL 新5大指標 完全比較評価ダッシュボード
=========================================
fpl_app_v5.py と同じリポジトリに置いて使用する場合は
Streamlit の multipage 機能を使い pages/ フォルダに入れるか、
このファイル単体で streamlit run fpl_eval_v2.py として起動してください。

起動方法:
  streamlit run fpl_eval_v2.py
"""

import io
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="FPL Metrics Evaluation",
    layout="wide",
    page_icon="📊",
)

COLORS = {
    "primary": "#00A651",
    "dark":    "#0D1B2A",
    "accent1": "#E8FA00",
    "accent2": "#FF4B4B",
    "muted":   "#64748B",
    "bg":      "#F0FDF4",
}

st.markdown(f"""
<style>
  .stApp {{ background-color: {COLORS['bg']}; }}
  .eval-header {{
    background: linear-gradient(135deg, {COLORS['dark']} 0%, #1a3a2a 100%);
    padding: 1.2rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; border-left: 6px solid {COLORS['primary']};
  }}
  .eval-header h1 {{
    color: {COLORS['accent1']}; font-size: 1.6rem; font-weight: 900; margin: 0;
  }}
  .eval-header p {{ color: #94A3B8; font-size: 0.85rem; margin: 0.3rem 0 0 0; }}
  .section-title {{
    font-size: 1.05rem; font-weight: 800; color: {COLORS['dark']};
    border-bottom: 3px solid {COLORS['primary']}; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0;
  }}
  [data-testid="stSidebar"] {{ background: {COLORS['dark']}; }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown p {{ color: #CBD5E1 !important; }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# データ取得（fpl_app_v5.py と同じロジック）
# =========================================================
VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FULL_MIN     = 3420.0
POS_MAP      = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}

def _get(url):
    for _ in range(3):
        try:
            r = requests.get(url, headers={"User-Agent":"Mozilla/5.0 Chrome/124.0.0.0"}, timeout=20)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2)
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def load(season):
    r_p = _get(f"{VAASTAV_BASE}/{season}/players_raw.csv")
    r_t = _get(f"{VAASTAV_BASE}/{season}/teams.csv")
    if r_p is None:
        return None, {}
    df = pd.read_csv(io.StringIO(r_p.text))
    team_map = {}
    if r_t:
        dt = pd.read_csv(io.StringIO(r_t.text))
        if "id" in dt.columns and "name" in dt.columns:
            team_map = dict(zip(dt["id"], dt["name"]))
    return df, team_map

def prepare(df_raw, team_map):
    df = df_raw.copy()
    df["player_name"] = df.get("web_name", pd.Series(dtype=str))
    df["position"]    = df.get("element_type", pd.Series(dtype=float)).map(POS_MAP).fillna("UNK")
    df["team_name"]   = df.get("team", pd.Series(dtype=float)).map(team_map).fillna("Unknown")
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

def compute(df_raw, min_minutes):
    df = df_raw[df_raw["minutes"] >= min_minutes].copy().reset_index(drop=True)
    if df.empty:
        return df
    p90 = (df["minutes"] / 90).clip(lower=1)

    # ① 攻撃プロセス
    pos_atk = {"GK":0.3,"DEF":0.6,"MID":1.2,"FWD":1.4}
    xA_p90  = df["expected_assists"] / p90
    xGI_p90 = df["expected_goal_involvements"] / p90
    cre_max = df["creativity"].max()
    cre_n   = df["creativity"] / (cre_max if cre_max > 0 else 1)
    df["①攻撃プロセス_raw"] = (
        xA_p90 * 2.0 + cre_n * 1.5 + xGI_p90 * 0.5
    ) * df["position"].map(pos_atk).fillna(1.0)

    # ② 守備プロセス v3
    pos_def     = {"GK":1.2,"DEF":1.1,"MID":0.8,"FWD":0.4}
    saves_p90   = df["saves"] / p90
    gc_p90      = df["goals_conceded"] / p90
    cs_w        = df["clean_sheets"] * (df["minutes"] / FULL_MIN)
    def_act_p90 = (df["tackles"] + df["recoveries"] + df["clearances_blocks_interceptions"]) / p90
    is_gk  = df["position"] == "GK"
    is_def = df["position"] == "DEF"
    is_mid = df["position"] == "MID"
    df["②守備プロセス_raw"] = np.where(
        is_gk,  saves_p90*2.5 + cs_w*8.0 - gc_p90*0.5,
        np.where(is_def, def_act_p90*0.25 + cs_w*3.0 - gc_p90*0.6,
        np.where(is_mid, def_act_p90*0.20, def_act_p90*0.08))
    ) * df["position"].map(pos_def).fillna(0.7)

    # ③ 得点近接
    xG_p90  = df["expected_goals"] / p90
    thr_max = df["threat"].max()
    thr_n   = df["threat"] / (thr_max if thr_max > 0 else 1)
    df["③得点近接_raw"] = xG_p90*3.0 + thr_n*2.0 + (df["goals_scored"]/p90)*1.0

    # ④ 失点近接
    df["④失点近接_raw"] = np.where(
        is_gk,  saves_p90*2.0 + df["clean_sheets"]*0.5 - df["red_cards"]*2.0,
        np.where(is_def, df["clean_sheets"]*0.8 - gc_p90*0.5 - df["red_cards"]*2.0,
                 df["clean_sheets"]*0.3 - df["red_cards"]*1.0)
    )

    # ⑤ Luck
    df["⑤得点Luck"] = df["goals_scored"] - df["expected_goals"]
    df["⑤守備Luck"] = df["expected_goals_conceded"] - df["goals_conceded"]
    df["⑤Luck合計"] = df["⑤得点Luck"] + df["⑤守備Luck"]

    # Z標準化（①②）
    for raw, norm in [("①攻撃プロセス_raw","①攻撃プロセス"),("②守備プロセス_raw","②守備プロセス")]:
        mu, sd = df[raw].mean(), df[raw].std()
        df[norm] = (df[raw] - mu) / (sd if sd > 0 else 1)

    df["③得点近接"]             = df["③得点近接_raw"]
    df["④失点近接"]             = df["④失点近接_raw"]
    df["総合プロセス(①+②)"]    = df["①攻撃プロセス"] + df["②守備プロセス"]
    df["総合クリティカル(③+④)"] = df["③得点近接"]    + df["④失点近接"]
    return df

def _styler_map(styler, func, subset=None):
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset)
    return styler.applymap(func, subset=subset)

# =========================================================
# メイン
# =========================================================
st.markdown("""
<div class="eval-header">
  <h1>📊 FPL 新5大指標 完全比較評価ダッシュボード</h1>
  <p>WCアプリと同じ手法（Pearson/Spearman相関・AUC・複合モデル・ヒートマップ）でFPL既存指標と比較</p>
</div>
""", unsafe_allow_html=True)

# サイドバー
st.sidebar.markdown(f"<div style='color:{COLORS['accent1']};font-size:1rem;font-weight:900'>📊 Evaluation Settings</div>", unsafe_allow_html=True)
season  = st.sidebar.selectbox("シーズン", ["2024-25","2023-24","2022-23"])
min_min = st.sidebar.slider("最低出場分数", 450, 2000, 900, 90)

with st.spinner(f"{season} データ取得中..."):
    df_raw, team_map = load(season)

if df_raw is None:
    st.error(f"""
    **データ取得失敗**

    以下のURLからCSVをダウンロードし、`players_raw_{season.replace('-','_')}.csv` として保存:
    ```
    {VAASTAV_BASE}/{season}/players_raw.csv
    ```
    """)
    st.stop()

df_prep = prepare(df_raw, team_map)
df = compute(df_prep, min_min)

if df.empty:
    st.warning(f"出場{min_min}分以上の選手がいません。最低出場分数を下げてください。")
    st.stop()

st.success(f"✅ {season}  |  対象選手: {len(df)}名（{min_min}分以上）")

# =========================================================
# A. 相関表（全指標ペア）
# =========================================================
st.markdown("<div class='section-title'>A. 新5大指標 × 既存FPL指標 — Pearson / Spearman 相関</div>", unsafe_allow_html=True)
st.caption("WCアプリと同じ手法。***p<0.001 / **p<0.01 / *p<0.05")

NEW_METRICS = [
    ("①攻撃プロセス",          "① Attack Process"),
    ("②守備プロセス",          "② Defense Process v3"),
    ("総合プロセス(①+②)",     "① + ② Process Total"),
    ("③得点近接",              "③ Goal Threat"),
    ("④失点近接",              "④ Save Contribution"),
    ("総合クリティカル(③+④)", "③ + ④ Critical Total"),
    ("⑤得点Luck",             "⑤ Goal Luck"),
    ("⑤守備Luck",             "⑤ Defense Luck"),
]

EXISTING = [
    ("expected_goals",             "xG"),
    ("expected_assists",           "xA"),
    ("expected_goal_involvements", "xGI"),
    ("expected_goals_conceded",    "xGC"),
    ("influence",                  "Influence"),
    ("creativity",                 "Creativity"),
    ("threat",                     "Threat"),
    ("ict_index",                  "ICT Index"),
    ("goals_scored",               "Goals"),
    ("assists",                    "Assists"),
    ("clean_sheets",               "Clean Sheets"),
    ("goals_conceded",             "Goals Conceded"),
    ("saves",                      "Saves"),
    ("bonus",                      "Bonus"),
    ("total_points",               "FPL Total Pts"),
]

# タブで新指標ごとに表示
tabs_corr = st.tabs([label for _, label in NEW_METRICS])
for (new_col, new_label), tab in zip(NEW_METRICS, tabs_corr):
    with tab:
        if new_col not in df.columns:
            st.info("このシーズンのデータにこの指標が含まれていません")
            continue
        rows = []
        for ex_col, ex_label in EXISTING:
            if ex_col not in df.columns:
                continue
            sub = df[[new_col, ex_col]].dropna()
            if len(sub) < 10:
                continue
            r_p, p_p = pearsonr(sub[new_col], sub[ex_col])
            r_s, _   = spearmanr(sub[new_col], sub[ex_col])
            sig = "***" if p_p<.001 else ("**" if p_p<.01 else ("*" if p_p<.05 else ""))
            rows.append({
                "既存指標": ex_label,
                "Pearson r": round(r_p, 3),
                "有意性": sig,
                "Spearman ρ": round(r_s, 3),
            })
        if rows:
            df_corr = pd.DataFrame(rows).sort_values("Pearson r", key=abs, ascending=False)
            st.dataframe(
                df_corr.style
                .background_gradient(subset=["Pearson r","Spearman ρ"], cmap="RdYlGn", vmin=-1, vmax=1)
                .format({"Pearson r":"{:+.3f}", "Spearman ρ":"{:+.3f}"}),
                use_container_width=True, height=480
            )

# =========================================================
# B. AUC（全指標）
# =========================================================
st.markdown("<div class='section-title'>B. FPL高得点予測 AUC（Stratified 5-fold LogReg）</div>", unsafe_allow_html=True)
st.caption("ターゲット: FPL総得点 上位50% = 1 / 下位50% = 0  |  0.5=ランダム、1.0=完全予測")

y    = (df["total_points"] >= df["total_points"].median()).astype(int)
n_sp = min(5, max(2, len(df)//10))
cv   = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=42)

ALL_METRICS = [
    ("総合プロセス(①+②)",    "🆕 ①+② Process Total",  True),
    ("①攻撃プロセス",         "🆕 ① Attack Process",    True),
    ("②守備プロセス",         "🆕 ② Defense Process",   True),
    ("③得点近接",             "🆕 ③ Goal Threat",        True),
    ("④失点近接",             "🆕 ④ Save Contribution",  True),
    ("総合クリティカル(③+④)","🆕 ③+④ Critical Total",  True),
    ("⑤得点Luck",             "🆕 ⑤ Goal Luck",          True),
    ("⑤守備Luck",             "🆕 ⑤ Defense Luck",       True),
    ("expected_goals",              "📌 xG",               False),
    ("expected_assists",            "📌 xA",               False),
    ("expected_goal_involvements",  "📌 xGI",              False),
    ("ict_index",                   "📌 ICT Index",         False),
    ("threat",                      "📌 Threat",            False),
    ("creativity",                  "📌 Creativity",        False),
    ("influence",                   "📌 Influence",         False),
    ("clean_sheets",                "📌 Clean Sheets",      False),
    ("saves",                       "📌 Saves",             False),
    ("goals_scored",                "📌 Goals",             False),
    ("assists",                     "📌 Assists",           False),
    ("bonus",                       "📌 Bonus Points",      False),
]

auc_rows = []
prog = st.progress(0, text="AUCを計算中...")
for i, (m_col, label, is_new) in enumerate(ALL_METRICS):
    prog.progress((i+1)/len(ALL_METRICS), text=f"計算中: {label}")
    if m_col not in df.columns:
        continue
    X = StandardScaler().fit_transform(df[[m_col]].fillna(0))
    try:
        auc = cross_val_score(
            LogisticRegression(max_iter=1000), X, y,
            cv=cv, scoring="roc_auc"
        ).mean()
        auc_rows.append({"指標": label, "AUC": round(auc,3), "種類": "🆕 新指標" if is_new else "📌 既存指標"})
    except Exception:
        pass
prog.empty()

if auc_rows:
    df_auc = pd.DataFrame(auc_rows).sort_values("AUC", ascending=False)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_a, ax_a = plt.subplots(figsize=(9, 7))
        fig_a.patch.set_facecolor(COLORS["bg"])
        ax_a.set_facecolor(COLORS["bg"])
        clrs = [COLORS["primary"] if "🆕" in r else "#94A3B8" for r in df_auc["種類"]]
        bars = ax_a.barh(df_auc["指標"].tolist()[::-1], df_auc["AUC"].tolist()[::-1],
                         color=clrs[::-1], edgecolor="white", lw=0.5)
        ax_a.axvline(0.5, color=COLORS["accent2"], ls="--", lw=1.5, label="Random (0.5)")
        ax_a.set_xlabel("AUC")
        ax_a.set_title("FPL High Score Prediction AUC\nNew Metrics vs Existing FPL Metrics",
                        fontweight="bold", fontsize=10)
        ax_a.set_xlim(0.3, 1.0)
        ax_a.grid(axis="x", color="#CBD5E1", lw=0.5)
        for bar, val in zip(bars[::-1], df_auc["AUC"]):
            ax_a.text(val+.005, bar.get_y()+bar.get_height()/2,
                      f"{val:.3f}", va="center", fontsize=8, color=COLORS["dark"])
        ax_a.legend(handles=[
            mpatches.Patch(color=COLORS["primary"], label="🆕 新指標"),
            mpatches.Patch(color="#94A3B8", label="📌 既存指標"),
        ], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)

    with col_table:
        pos_col = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}
        st.dataframe(
            df_auc[["指標","AUC","種類"]].style
            .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.4, vmax=0.9)
            .format({"AUC":"{:.3f}"}),
            use_container_width=True, height=600
        )

# 複合モデル
st.markdown("<div class='section-title'>B-2. 複合モデル AUC（WCアプリの「新指標フル」相当）</div>", unsafe_allow_html=True)
combos = {
    "新①+② のみ":                    ["総合プロセス(①+②)"],
    "新③+④ のみ":                    ["総合クリティカル(③+④)"],
    "新⑤ のみ":                       ["⑤得点Luck","⑤守備Luck"],
    "新①〜④ 合計":                   ["総合プロセス(①+②)","総合クリティカル(③+④)"],
    "新①〜⑤ 全部":                   ["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck"],
    "xGI + ICT (既存ベースライン)":   ["expected_goal_involvements","ict_index"],
    "xG + xA + CS (既存)":            ["expected_goals","expected_assists","clean_sheets"],
    "新全 + xGI + ICT (ハイブリッド)":["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck",
                                        "expected_goal_involvements","ict_index"],
}
combo_rows = []
for name, cols in combos.items():
    valid = [c for c in cols if c in df.columns]
    if not valid:
        continue
    X = StandardScaler().fit_transform(df[valid].fillna(0))
    try:
        auc = cross_val_score(
            LogisticRegression(max_iter=1000), X, y, cv=cv, scoring="roc_auc"
        ).mean()
        combo_rows.append({"モデル": name, "AUC": round(auc,3)})
    except Exception:
        pass

if combo_rows:
    df_combo = pd.DataFrame(combo_rows).sort_values("AUC", ascending=False)
    st.dataframe(
        df_combo.style
        .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.4, vmax=0.9)
        .format({"AUC":"{:.3f}"}),
        use_container_width=True, height=320
    )

# =========================================================
# C. 相関ヒートマップ
# =========================================================
st.markdown("<div class='section-title'>C. 指標間 相関マトリクス（全指標）</div>", unsafe_allow_html=True)
hm_cols = [
    "①攻撃プロセス","②守備プロセス","③得点近接","④失点近接","⑤得点Luck","⑤守備Luck",
    "expected_goals","expected_assists","expected_goal_involvements",
    "influence","creativity","threat","ict_index",
    "clean_sheets","saves","goals_scored","assists","bonus","total_points",
]
hm_cols = [c for c in hm_cols if c in df.columns]
lbl_map = {
    "①攻撃プロセス":"①Atk","②守備プロセス":"②Def",
    "③得点近接":"③GThr","④失点近接":"④Save",
    "⑤得点Luck":"⑤GLuck","⑤守備Luck":"⑤DLuck",
    "expected_goals":"xG","expected_assists":"xA",
    "expected_goal_involvements":"xGI",
    "influence":"Influ","creativity":"Creat","threat":"Threat","ict_index":"ICT",
    "clean_sheets":"CS","saves":"Saves","goals_scored":"Goals",
    "assists":"Assists","bonus":"Bonus","total_points":"FPLPts",
}
fig_h, ax_h = plt.subplots(figsize=(14, 11))
fig_h.patch.set_facecolor(COLORS["bg"])
sns.heatmap(
    df[hm_cols].rename(columns=lbl_map).corr(),
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    ax=ax_h, annot_kws={"size":7}, linewidths=0.3, square=True
)
ax_h.set_title("Full Correlation Matrix — New 5 Metrics vs All FPL Existing Metrics",
               fontsize=11, fontweight="bold", pad=12)
ax_h.tick_params(axis="x", labelsize=8, rotation=45)
ax_h.tick_params(axis="y", labelsize=8, rotation=0)
plt.tight_layout()
st.pyplot(fig_h, use_container_width=True)

# =========================================================
# D. ② 守備プロセス ポジション別分布
# =========================================================
st.markdown("<div class='section-title'>D. ② 守備プロセス v3 — ポジション別分布確認</div>", unsafe_allow_html=True)
st.caption("GK/DEFが高く、FW（守備免除選手）が低ければ設計通り")

col_box, col_top = st.columns([1, 2])
with col_box:
    fig_b, ax_b = plt.subplots(figsize=(5, 4))
    fig_b.patch.set_facecolor(COLORS["bg"])
    ax_b.set_facecolor(COLORS["bg"])
    pos_order = ["GK","DEF","MID","FWD"]
    pos_data  = [df[df["position"]==p]["②守備プロセス"].dropna() for p in pos_order]
    pos_lbls  = [f"{p}\n(n={len(d)})" for p,d in zip(pos_order,pos_data) if len(d)>0]
    bp = ax_b.boxplot([d for d in pos_data if len(d)>0], labels=pos_lbls, patch_artist=True)
    for patch, color in zip(bp["boxes"],["#F59E0B","#3B82F6","#8B5CF6","#EF4444"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax_b.axhline(0, color="gray", ls="--", lw=0.7)
    ax_b.set_title("② Defense Process\nby Position", fontweight="bold", fontsize=9)
    ax_b.set_ylabel("z-score")
    plt.tight_layout()
    st.pyplot(fig_b, use_container_width=True)

with col_top:
    for pos in ["GK","DEF","MID","FWD"]:
        sub = df[df["position"]==pos].nlargest(5,"②守備プロセス")
        if sub.empty:
            continue
        st.markdown(f"**{pos} TOP5**")
        show = ["player_name","team_name","minutes","clean_sheets","saves","②守備プロセス"]
        show = [c for c in show if c in sub.columns]
        st.dataframe(
            sub[show].rename(columns={
                "player_name":"選手","team_name":"チーム",
                "minutes":"出場分","clean_sheets":"CS","saves":"Saves",
                "②守備プロセス":"②Def"
            }).style.format({"②Def":"{:+.2f}"}),
            use_container_width=True, height=210
        )

# フッター
st.markdown(f"""
<div style="background:{COLORS['dark']};color:#94A3B8;font-size:.72rem;
     padding:.8rem 1rem;border-radius:8px;margin-top:2rem;text-align:center">
  Data: <b>vaastav/Fantasy-Premier-League</b> (github.com/vaastav/Fantasy-Premier-League) ·
  FPL data © Premier League · Non-commercial personal use only
</div>
""", unsafe_allow_html=True)
