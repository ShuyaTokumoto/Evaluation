"""
FPL 新5大指標 完全比較評価ダッシュボード v3
==============================================
【sklearn完全不使用】numpy + scipy のみでAUCを計算。
Python 3.14対応。Streamlit Cloudの依存関係問題を回避。

起動: streamlit run fpl_eval_v3.py
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
from scipy.stats import pearsonr, spearmanr, rankdata

warnings.filterwarnings("ignore")

# =========================================================
# ページ設定
# =========================================================
st.set_page_config(page_title="FPL Metrics Evaluation", layout="wide", page_icon="📊")

COLORS = {
    "primary": "#00A651", "dark": "#0D1B2A",
    "accent1": "#E8FA00", "accent2": "#FF4B4B",
    "muted": "#64748B",   "bg": "#F0FDF4",
}

st.markdown(f"""
<style>
  .stApp {{ background-color:{COLORS['bg']}; }}
  .eval-header {{
    background:linear-gradient(135deg,{COLORS['dark']} 0%,#1a3a2a 100%);
    padding:1.2rem 2rem; border-radius:12px;
    margin-bottom:1.5rem; border-left:6px solid {COLORS['primary']};
  }}
  .eval-header h1 {{ color:{COLORS['accent1']}; font-size:1.6rem;
    font-weight:900; margin:0; }}
  .eval-header p  {{ color:#94A3B8; font-size:.85rem; margin:.3rem 0 0 0; }}
  .section-title  {{
    font-size:1.05rem; font-weight:800; color:{COLORS['dark']};
    border-bottom:3px solid {COLORS['primary']};
    padding-bottom:.4rem; margin:1.2rem 0 .8rem 0;
  }}
  [data-testid="stSidebar"] {{ background:{COLORS['dark']}; }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown p {{ color:#CBD5E1 !important; }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# AUC（sklearn不使用・numpy/scipy のみ）
# =========================================================
def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U統計量でAUCを計算（sklearn不要）"""
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks   = rankdata(np.concatenate([pos, neg]))
    U       = ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2
    return float(U / (len(pos) * len(neg)))


def cv_auc(y_true: np.ndarray, x: np.ndarray, n_splits: int = 5, seed: int = 42) -> float:
    """Stratified K-fold AUC（sklearn不要・直接スコアでAUC）"""
    y_true = np.asarray(y_true, dtype=float)
    x      = np.asarray(x,      dtype=float)
    # 欠損を中央値で補完
    x = np.where(np.isnan(x), np.nanmedian(x), x)

    rng      = np.random.default_rng(seed)
    pos_idx  = np.where(y_true == 1)[0];  rng.shuffle(pos_idx)
    neg_idx  = np.where(y_true == 0)[0];  rng.shuffle(neg_idx)
    pos_f    = np.array_split(pos_idx, n_splits)
    neg_f    = np.array_split(neg_idx, n_splits)

    aucs = []
    for i in range(n_splits):
        val_idx = np.concatenate([pos_f[i], neg_f[i]])
        if len(val_idx) < 4:
            continue
        auc = roc_auc(y_true[val_idx], x[val_idx])
        aucs.append(max(auc, 1 - auc))   # 方向を自動修正

    return float(np.mean(aucs)) if aucs else 0.5


def cv_auc_multi(y_true: np.ndarray, X: np.ndarray,
                 n_splits: int = 5, seed: int = 42) -> float:
    """複数指標の合成スコアでAUC（正規化後の単純平均）"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return cv_auc(y_true, X, n_splits, seed)
    # 列ごとに0-1正規化して平均
    X_norm = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        col = np.where(np.isnan(col), np.nanmedian(col), col)
        mn, mx = col.min(), col.max()
        X_norm[:, j] = (col - mn) / (mx - mn + 1e-9)
    composite = X_norm.mean(axis=1)
    return cv_auc(y_true, composite, n_splits, seed)


# =========================================================
# データ取得
# =========================================================
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FULL_MIN = 3420.0
POS_MAP  = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}


def _get(url):
    for _ in range(3):
        try:
            r = requests.get(
                url, timeout=20,
                headers={"User-Agent": "Mozilla/5.0 Chrome/124.0.0.0"}
            )
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2)
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def load(season):
    r_p = _get(f"{VAASTAV}/{season}/players_raw.csv")
    r_t = _get(f"{VAASTAV}/{season}/teams.csv")
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
    for c in [
        "minutes","goals_scored","assists","clean_sheets","goals_conceded",
        "saves","yellow_cards","red_cards","bonus","bps","total_points","now_cost",
        "expected_goals","expected_assists","expected_goal_involvements",
        "expected_goals_conceded","influence","creativity","threat","ict_index",
        "tackles","recoveries","clearances_blocks_interceptions",
    ]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    df["price_m"] = df["now_cost"] / 10.0
    return df


def compute(df_raw, min_minutes):
    df = df_raw[df_raw["minutes"] >= min_minutes].copy().reset_index(drop=True)
    if df.empty:
        return df
    p90 = (df["minutes"] / 90).clip(lower=1)

    # ① 攻撃プロセス
    pa = {"GK":.3,"DEF":.6,"MID":1.2,"FWD":1.4}
    xA_p90  = df["expected_assists"] / p90
    xGI_p90 = df["expected_goal_involvements"] / p90
    cm = df["creativity"].max()
    cn = df["creativity"] / (cm if cm > 0 else 1)
    df["①攻撃プロセス_raw"] = (xA_p90*2.0 + cn*1.5 + xGI_p90*.5) * df["position"].map(pa).fillna(1)

    # ② 守備プロセス v3
    pd_w = {"GK":1.2,"DEF":1.1,"MID":.8,"FWD":.4}
    sp90 = df["saves"] / p90
    gc90 = df["goals_conceded"] / p90
    csw  = df["clean_sheets"] * (df["minutes"] / FULL_MIN)
    da90 = (df["tackles"]+df["recoveries"]+df["clearances_blocks_interceptions"]) / p90
    gk   = df["position"]=="GK"; dfw=df["position"]=="DEF"; mid=df["position"]=="MID"
    df["②守備プロセス_raw"] = np.where(
        gk,  sp90*2.5 + csw*8.0 - gc90*.5,
        np.where(dfw, da90*.25 + csw*3.0 - gc90*.6,
        np.where(mid, da90*.20, da90*.08))
    ) * df["position"].map(pd_w).fillna(.7)

    # ③ 得点近接
    xg90 = df["expected_goals"] / p90
    tm   = df["threat"].max()
    tn   = df["threat"] / (tm if tm > 0 else 1)
    df["③得点近接_raw"] = xg90*3.0 + tn*2.0 + (df["goals_scored"]/p90)*1.0

    # ④ 失点近接
    df["④失点近接_raw"] = np.where(
        gk,  sp90*2.0 + df["clean_sheets"]*.5 - df["red_cards"]*2.0,
        np.where(dfw, df["clean_sheets"]*.8 - gc90*.5 - df["red_cards"]*2.0,
                 df["clean_sheets"]*.3 - df["red_cards"]*1.0)
    )

    # ⑤ Luck
    df["⑤得点Luck"] = df["goals_scored"] - df["expected_goals"]
    df["⑤守備Luck"] = df["expected_goals_conceded"] - df["goals_conceded"]
    df["⑤Luck合計"] = df["⑤得点Luck"] + df["⑤守備Luck"]

    # Z標準化（①②）
    for raw, norm in [("①攻撃プロセス_raw","①攻撃プロセス"),
                      ("②守備プロセス_raw","②守備プロセス")]:
        mu, sd = df[raw].mean(), df[raw].std()
        df[norm] = (df[raw]-mu) / (sd if sd > 0 else 1)

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
# UI
# =========================================================
st.markdown("""
<div class="eval-header">
  <h1>📊 FPL 新5大指標 完全比較評価ダッシュボード</h1>
  <p>Pearson/Spearman相関・AUC・複合モデル・ヒートマップ ― sklearn不使用・Python 3.14対応版</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    f"<div style='color:{COLORS['accent1']};font-size:1rem;font-weight:900'>📊 Settings</div>",
    unsafe_allow_html=True
)
season  = st.sidebar.selectbox("シーズン", ["2024-25","2023-24","2022-23"])
min_min = st.sidebar.slider("最低出場分数", 450, 2000, 900, 90)

with st.spinner(f"{season} データ取得中..."):
    df_raw, team_map = load(season)

if df_raw is None:
    st.error(f"""
    **データ取得失敗**  
    以下のURLからCSVをダウンロードし、スクリプトと同フォルダに保存してください:  
    `{VAASTAV}/{season}/players_raw.csv`
    """)
    st.stop()

df_prep = prepare(df_raw, team_map)
df      = compute(df_prep, min_min)

if df.empty:
    st.warning("該当選手なし。最低出場分数を下げてください。")
    st.stop()

st.success(f"✅ {season}  |  対象選手: {len(df)} 名（{min_min}分以上）")

# =========================================================
# A. 相関
# =========================================================
st.markdown("<div class='section-title'>A. 新5大指標 × 既存FPL指標 — 相関（Pearson / Spearman）</div>",
            unsafe_allow_html=True)
st.caption("***p<0.001 / **p<0.01 / *p<0.05  |  絶対値が大きいほど強い相関")

NEW_M = [
    ("①攻撃プロセス",         "① Attack Process"),
    ("②守備プロセス",         "② Defense Process v3"),
    ("総合プロセス(①+②)",    "① + ② Process Total"),
    ("③得点近接",             "③ Goal Threat"),
    ("④失点近接",             "④ Save Contribution"),
    ("総合クリティカル(③+④)","③ + ④ Critical Total"),
    ("⑤得点Luck",            "⑤ Goal Luck"),
    ("⑤守備Luck",            "⑤ Defense Luck"),
]
EXIST = [
    ("expected_goals",             "xG"),
    ("expected_assists",           "xA"),
    ("expected_goal_involvements", "xGI"),
    ("expected_goals_conceded",    "xGC（被xG）"),
    ("influence",                  "Influence"),
    ("creativity",                 "Creativity"),
    ("threat",                     "Threat"),
    ("ict_index",                  "ICT Index"),
    ("goals_scored",               "Goals"),
    ("assists",                    "Assists"),
    ("clean_sheets",               "Clean Sheets"),
    ("goals_conceded",             "Goals Conceded"),
    ("saves",                      "Saves"),
    ("bonus",                      "Bonus Points"),
    ("total_points",               "FPL Total Points"),
]

tabs_a = st.tabs([lbl for _, lbl in NEW_M])
for (nc, nl), tab in zip(NEW_M, tabs_a):
    with tab:
        if nc not in df.columns:
            st.info("このシーズンにデータなし")
            continue
        rows = []
        for ec, el in EXIST:
            if ec not in df.columns: continue
            sub = df[[nc, ec]].dropna()
            if len(sub) < 10: continue
            rp, pp = pearsonr(sub[nc], sub[ec])
            rs, _  = spearmanr(sub[nc], sub[ec])
            sig = "***" if pp<.001 else ("**" if pp<.01 else ("*" if pp<.05 else ""))
            rows.append({"既存指標":el, "Pearson r":round(rp,3), "sig":sig, "Spearman ρ":round(rs,3)})
        if rows:
            dfc = pd.DataFrame(rows).sort_values("Pearson r", key=abs, ascending=False)
            st.dataframe(
                dfc.style
                .background_gradient(subset=["Pearson r","Spearman ρ"], cmap="RdYlGn", vmin=-1, vmax=1)
                .format({"Pearson r":"{:+.3f}", "Spearman ρ":"{:+.3f}"}),
                use_container_width=True, height=500
            )

# =========================================================
# B. AUC（sklearn不使用）
# =========================================================
st.markdown("<div class='section-title'>B. FPL高得点予測 AUC（Stratified 5-fold, sklearn不使用）</div>",
            unsafe_allow_html=True)
st.caption("ターゲット: FPL総得点 上位50%=1 / 下位50%=0  |  0.5=ランダム, 1.0=完全予測")

y = (df["total_points"] >= df["total_points"].median()).astype(int).values

ALL_M = [
    ("総合プロセス(①+②)",    "🆕 ①+② Process Total",  True),
    ("①攻撃プロセス",         "🆕 ① Attack Process",    True),
    ("②守備プロセス",         "🆕 ② Defense Process",   True),
    ("③得点近接",             "🆕 ③ Goal Threat",        True),
    ("④失点近接",             "🆕 ④ Save Contribution",  True),
    ("総合クリティカル(③+④)","🆕 ③+④ Critical Total",  True),
    ("⑤得点Luck",             "🆕 ⑤ Goal Luck",          True),
    ("⑤守備Luck",             "🆕 ⑤ Defense Luck",       True),
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
    ("bonus",                      "📌 Bonus Points",      False),
]

auc_rows = []
prog = st.progress(0, text="AUC計算中...")
for i, (mc, lbl, is_new) in enumerate(ALL_M):
    prog.progress((i+1)/len(ALL_M), text=f"計算中: {lbl}")
    if mc not in df.columns: continue
    auc = cv_auc(y, df[mc].values)
    auc_rows.append({"指標":lbl, "AUC":round(auc,3), "is_new":is_new})
prog.empty()

if auc_rows:
    df_auc = pd.DataFrame(auc_rows).sort_values("AUC", ascending=False)

    col_c, col_t = st.columns([3, 2])
    with col_c:
        fig_a, ax_a = plt.subplots(figsize=(9,7))
        fig_a.patch.set_facecolor(COLORS["bg"])
        ax_a.set_facecolor(COLORS["bg"])
        clrs  = [COLORS["primary"] if r else "#94A3B8" for r in df_auc["is_new"]]
        bars  = ax_a.barh(df_auc["指標"].tolist()[::-1], df_auc["AUC"].tolist()[::-1],
                          color=clrs[::-1], edgecolor="white", lw=.5)
        ax_a.axvline(.5, color=COLORS["accent2"], ls="--", lw=1.5)
        ax_a.set_xlabel("AUC")
        ax_a.set_title("FPL High Score Prediction AUC\n(green=new / gray=existing)",
                        fontweight="bold", fontsize=10)
        ax_a.set_xlim(.3, 1.0)
        ax_a.grid(axis="x", color="#CBD5E1", lw=.5)
        for bar, val in zip(bars[::-1], df_auc["AUC"]):
            ax_a.text(val+.005, bar.get_y()+bar.get_height()/2,
                      f"{val:.3f}", va="center", fontsize=8)
        ax_a.legend(handles=[
            mpatches.Patch(color=COLORS["primary"], label="🆕 新指標"),
            mpatches.Patch(color="#94A3B8",         label="📌 既存指標"),
        ], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)

    with col_t:
        st.dataframe(
            df_auc[["指標","AUC"]].style
            .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=.4, vmax=.9)
            .format({"AUC":"{:.3f}"}),
            use_container_width=True, height=620
        )

# 複合モデル
st.markdown("<div class='section-title'>B-2. 複合モデル AUC</div>", unsafe_allow_html=True)
combos = {
    "新①+②":                        ["総合プロセス(①+②)"],
    "新③+④":                        ["総合クリティカル(③+④)"],
    "新⑤":                           ["⑤得点Luck","⑤守備Luck"],
    "新①〜④":                       ["総合プロセス(①+②)","総合クリティカル(③+④)"],
    "新①〜⑤ 全指標":                ["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck"],
    "xGI + ICT（既存ベースライン）": ["expected_goal_involvements","ict_index"],
    "xG + xA + CS（既存）":          ["expected_goals","expected_assists","clean_sheets"],
    "新全 + xGI + ICT（ハイブリッド）":["総合プロセス(①+②)","総合クリティカル(③+④)","⑤得点Luck",
                                         "expected_goal_involvements","ict_index"],
}
crows = []
for name, cols in combos.items():
    valid = [c for c in cols if c in df.columns]
    if not valid: continue
    X = df[valid].fillna(0).values
    auc = cv_auc_multi(y, X)
    crows.append({"モデル":name, "AUC":round(auc,3)})

if crows:
    df_c = pd.DataFrame(crows).sort_values("AUC", ascending=False)
    st.dataframe(
        df_c.style
        .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=.4, vmax=.9)
        .format({"AUC":"{:.3f}"}),
        use_container_width=True, height=320
    )

# =========================================================
# C. 相関ヒートマップ
# =========================================================
st.markdown("<div class='section-title'>C. 指標間 相関マトリクス（全指標）</div>",
            unsafe_allow_html=True)
hm = [
    "①攻撃プロセス","②守備プロセス","③得点近接","④失点近接","⑤得点Luck","⑤守備Luck",
    "expected_goals","expected_assists","expected_goal_involvements",
    "influence","creativity","threat","ict_index",
    "clean_sheets","saves","goals_scored","assists","bonus","total_points",
]
hm = [c for c in hm if c in df.columns]
lm = {
    "①攻撃プロセス":"①Atk","②守備プロセス":"②Def",
    "③得点近接":"③GThr","④失点近接":"④Save",
    "⑤得点Luck":"⑤GLuck","⑤守備Luck":"⑤DLuck",
    "expected_goals":"xG","expected_assists":"xA",
    "expected_goal_involvements":"xGI","influence":"Influ",
    "creativity":"Creat","threat":"Threat","ict_index":"ICT",
    "clean_sheets":"CS","saves":"Saves","goals_scored":"Goals",
    "assists":"Ast","bonus":"Bonus","total_points":"FPLPts",
}
fig_h, ax_h = plt.subplots(figsize=(14,11))
fig_h.patch.set_facecolor(COLORS["bg"])
sns.heatmap(df[hm].rename(columns=lm).corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax_h,
            annot_kws={"size":7}, linewidths=.3, square=True)
ax_h.set_title("Full Correlation Matrix — New 5 Metrics vs All FPL Existing Metrics",
               fontsize=11, fontweight="bold", pad=12)
ax_h.tick_params(axis="x", labelsize=8, rotation=45)
ax_h.tick_params(axis="y", labelsize=8, rotation=0)
plt.tight_layout()
st.pyplot(fig_h, use_container_width=True)

# =========================================================
# D. ② 守備ポジション別分布
# =========================================================
st.markdown("<div class='section-title'>D. ② 守備プロセス v3 — ポジション別分布</div>",
            unsafe_allow_html=True)
st.caption("GK/DEFが高く FW/MFの攻撃的選手が低ければ設計通り")

col_b, col_top = st.columns([1, 2])
with col_b:
    fig_b, ax_b = plt.subplots(figsize=(5,4))
    fig_b.patch.set_facecolor(COLORS["bg"])
    ax_b.set_facecolor(COLORS["bg"])
    pd_ord = ["GK","DEF","MID","FWD"]
    pd_dat = [df[df["position"]==p]["②守備プロセス"].dropna() for p in pd_ord]
    pd_lbl = [f"{p}\n(n={len(d)})" for p,d in zip(pd_ord,pd_dat) if len(d)>0]
    bp = ax_b.boxplot([d for d in pd_dat if len(d)>0], labels=pd_lbl, patch_artist=True)
    for patch,col in zip(bp["boxes"],["#F59E0B","#3B82F6","#8B5CF6","#EF4444"]):
        patch.set_facecolor(col); patch.set_alpha(.6)
    ax_b.axhline(0, color="gray", ls="--", lw=.7)
    ax_b.set_title("② Defense by Position", fontweight="bold", fontsize=9)
    ax_b.set_ylabel("z-score")
    plt.tight_layout()
    st.pyplot(fig_b, use_container_width=True)

with col_top:
    for pos in ["GK","DEF","MID","FWD"]:
        sub = df[df["position"]==pos].nlargest(5,"②守備プロセス")
        if sub.empty: continue
        st.markdown(f"**{pos} TOP5**")
        sc = ["player_name","team_name","minutes","clean_sheets","saves","②守備プロセス"]
        sc = [c for c in sc if c in sub.columns]
        st.dataframe(
            sub[sc].rename(columns={
                "player_name":"選手","team_name":"チーム",
                "minutes":"出場分","clean_sheets":"CS",
                "saves":"Saves","②守備プロセス":"②Def"
            }).style.format({"②Def":"{:+.2f}"}),
            use_container_width=True, height=210
        )

# フッター
st.markdown(f"""
<div style="background:{COLORS['dark']};color:#94A3B8;font-size:.72rem;
     padding:.8rem 1rem;border-radius:8px;margin-top:2rem;text-align:center">
  Data: <b>vaastav/Fantasy-Premier-League</b> ·
  FPL data © Premier League · Non-commercial personal use only
</div>
""", unsafe_allow_html=True)
