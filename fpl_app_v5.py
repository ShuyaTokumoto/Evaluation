"""
FPL Analytics Dashboard v2
============================
データソース優先順位:
  1. vaastav/Fantasy-Premier-League GitHub CSV（最優先・完全データ）
  2. FPL公式API bootstrap-static（補完用）
  3. ローカルCSVファイル（手動配置フォールバック）

オフシーズン対応:
  FPL APIは6〜8月のオフシーズン中は選手データが空になります。
  このアプリはvaastav CSVを優先することでオフシーズン中も動作します。

起動方法: streamlit run fpl_app_v2.py
必要ライブラリ: pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn requests
"""

import io
import os
import time
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

def _styler_map(styler, func, subset=None, **kwargs):
    """pandas 2.1+(.map) / 旧版(.applymap) 自動選択"""
    if hasattr(styler, "map"):
        return styler.map(func, subset=subset, **kwargs)
    return styler.applymap(func, subset=subset, **kwargs)


# =========================================================
# ページ設定
# =========================================================
st.set_page_config(
    page_title="FPL 5-Metric Analytics",
    layout="wide",
    page_icon="⚽",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#00A651",
    "dark":    "#0D1B2A",
    "accent1": "#E8FA00",
    "accent2": "#FF4B4B",
    "muted":   "#64748B",
    "bg":      "#F0FDF4",
    "card":    "#FFFFFF",
}

st.markdown(f"""
<style>
  .stApp {{ background-color: {COLORS['bg']}; }}
  .fpl-header {{
    background: linear-gradient(135deg, {COLORS['dark']} 0%, #1a3a2a 100%);
    padding: 1.2rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; border-left: 6px solid {COLORS['primary']};
  }}
  .fpl-header h1 {{
    color: {COLORS['accent1']}; font-size: 1.8rem; font-weight: 900;
    letter-spacing: -0.5px; margin: 0; font-family: 'Arial Black', sans-serif;
  }}
  .fpl-header p {{ color: #94A3B8; font-size: 0.85rem; margin: 0.3rem 0 0 0; }}
  .section-title {{
    font-size: 1.05rem; font-weight: 800; color: {COLORS['dark']};
    border-bottom: 3px solid {COLORS['primary']}; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0; font-family: 'Arial Black', sans-serif;
  }}
  .data-badge {{
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 700;
  }}
  .badge-live   {{ background: #DCFCE7; color: #166534; }}
  .badge-cached {{ background: #FEF9C3; color: #854D0E; }}
  .badge-local  {{ background: #DBEAFE; color: #1E40AF; }}
  [data-testid="stSidebar"] {{ background: {COLORS['dark']}; }}
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] .stMarkdown p {{ color: #CBD5E1 !important; }}
  .fpl-footer {{
    background: {COLORS['dark']}; color: #94A3B8; font-size: 0.72rem;
    padding: 0.8rem 1rem; border-radius: 8px; margin-top: 2rem; text-align: center;
  }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# データ取得レイヤー
# =========================================================
VAASTAV_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FPL_API      = "https://fantasy.premierleague.com/api/bootstrap-static/"

SEASONS = {
    "2024-25": "2024-25",
    "2023-24": "2023-24",
    "2022-23": "2022-23",
}

REQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://github.com/",
}


def _get(url: str, timeout: int = 20) -> Optional[requests.Response]:
    for _ in range(3):
        try:
            r = requests.get(url, headers=REQ_HEADERS, timeout=timeout)
            if r.status_code == 200:
                return r
            time.sleep(1.5)
        except Exception:
            time.sleep(2)
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vaastav_players(season: str) -> tuple:
    """
    vaastav GitHub から players_raw.csv を取得。
    xG/xA/xGI/xGC + ICT + saves + tackles 等を含む完全版。
    """
    url = f"{VAASTAV_BASE}/{season}/players_raw.csv"
    r = _get(url)
    if r:
        df = pd.read_csv(io.StringIO(r.text))
        return df, "github"

    # ローカルファイルフォールバック
    local = f"players_raw_{season.replace('-','_')}.csv"
    if os.path.exists(local):
        return pd.read_csv(local), "local"
    return None, "none"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vaastav_gw(season: str) -> Optional[pd.DataFrame]:
    """GW別データ（FPL得点推移・試合単位のxG等）"""
    url = f"{VAASTAV_BASE}/{season}/gws/merged_gw.csv"
    r = _get(url)
    if r:
        return pd.read_csv(io.StringIO(r.text))
    local = f"merged_gw_{season.replace('-','_')}.csv"
    if os.path.exists(local):
        return pd.read_csv(local)
    return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vaastav_teams(season: str) -> Optional[pd.DataFrame]:
    url = f"{VAASTAV_BASE}/{season}/teams.csv"
    r = _get(url)
    if r:
        return pd.read_csv(io.StringIO(r.text))
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fpl_api() -> Optional[dict]:
    """
    FPL API bootstrap-static。
    オフシーズン（6〜8月）は選手のminutesが0になるため、
    チーム名マッピング用途のみに使用する。
    """
    r = _get(FPL_API)
    return r.json() if r else None


# =========================================================
# 選手データ整形
# =========================================================
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_MAP_STR = {"1": "GK", "2": "DEF", "3": "MID", "4": "FWD",
                    "GK":"GK","DEF":"DEF","MID":"MID","FWD":"FWD"}


def prepare_players(
    df_raw: pd.DataFrame,
    team_df: Optional[pd.DataFrame] = None,
    fpl_api: Optional[dict] = None,
) -> pd.DataFrame:
    """
    vaastav players_raw.csv を分析用に整形。
    チーム名はvaastav teams.csv → FPL API の順で補完。
    """
    df = df_raw.copy()

    # ── 選手名 ──────────────────────────────────────
    if "web_name" in df.columns:
        df["player_name"] = df["web_name"]
    elif "second_name" in df.columns:
        df["player_name"] = (
            (df["first_name"] if "first_name" in df.columns else pd.Series("", index=df.index, dtype=str))
            .str[0] + ". " + df["second_name"]
        )
    else:
        df["player_name"] = df.index.astype(str)

    # ── ポジション ──────────────────────────────────
    if "element_type" in df.columns:
        df["position"] = df["element_type"].map(POSITION_MAP).fillna("UNK")
    elif "position" in df.columns:
        df["position"] = df["position"].map(POSITION_MAP_STR).fillna("UNK")

    # ── チーム名 ────────────────────────────────────
    if team_df is not None and "team" in df.columns and "id" in team_df.columns:
        team_map = dict(zip(team_df["id"], team_df["name"]))
        df["team_name"] = df["team"].map(team_map).fillna("Unknown")
    elif fpl_api and "teams" in fpl_api and "team" in df.columns:
        api_team_map = {t["id"]: t["name"] for t in fpl_api["teams"]}
        df["team_name"] = df["team"].map(api_team_map).fillna("Unknown")
    else:
        df["team_name"] = (df["team"] if "team" in df.columns else pd.Series("Unknown", index=df.index, dtype=str)).astype(str)

    # ── 数値列の正規化 ──────────────────────────────
    num_cols = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "saves", "yellow_cards", "red_cards",
        "bonus", "bps", "total_points", "now_cost",
        "expected_goals", "expected_assists",
        "expected_goal_involvements", "expected_goals_conceded",
        "influence", "creativity", "threat", "ict_index",
        "tackles", "recoveries", "clearances_blocks_interceptions",
        "penalties_saved", "own_goals", "penalties_missed",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0.0

    # 価格
    df["price_m"] = df["now_cost"] / 10.0

    return df


# =========================================================
# 新5大指標の算出
# =========================================================
def compute_metrics(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    """
    FPL/vaastav フィールドから新5大指標を算出。

    ① 攻撃プロセス: xA/90 × ポジション補正 + Creativity(正規化) + xGI/90
    ② 守備プロセス: (tackles+recoveries+CBI)/90 × ポジション補正 + Influence(正規化)
    ③ 得点近接:    xG/90 × 3 + Threat(正規化) × 2 + goals/90
    ④ 失点近接:    GK=saves/90×2+CS×0.5 / DF=CS×0.8-GC/90×0.5 / 他=CS×0.3
    ⑤ Luck:        得点Luck=goals-xG / 守備Luck=xGC-GC
    """
    df = df[df["minutes"] >= min_minutes].copy().reset_index(drop=True)
    if df.empty:
        return df

    p90 = (df["minutes"] / 90).clip(lower=1)

    # ── ① 攻撃プロセス ──────────────────────────────
    pos_atk_w = {"GK": 0.3, "DEF": 0.6, "MID": 1.2, "FWD": 1.4}
    df["_pos_w_atk"] = df["position"].map(pos_atk_w).fillna(1.0)

    xA_p90      = df["expected_assists"] / p90
    xGI_p90     = df["expected_goal_involvements"] / p90
    cre_max     = df["creativity"].max()
    creativity_n = df["creativity"] / (cre_max if cre_max > 0 else 1)

    df["①攻撃プロセス_raw"] = (
        xA_p90       * 2.0
        + creativity_n * 1.5
        + xGI_p90    * 0.5
    ) * df["_pos_w_atk"]

    # ── ② 守備プロセス v3（最終改善版）──────────────────
    #
    # 改善の経緯:
    #   v1: influence × FWD補正1.5 → サラー・イサクが最上位（誤）
    #   v2: CS率（CS/出場試合数）× ポジション補正 →
    #       Zinchenko(10試合CS8)・Tsimikas(6試合CS5)が上位（誤）
    #       De Bruyne(Man City 20試合CS9)が上位（誤）
    #
    # v3の設計思想:
    #   1. CS項: cs_weighted = CS × (minutes/3420) で出場時間補正
    #      → 少ない出場でCSが多くても過大評価されない
    #      → フルシーズン出場の守備的選手が自然に高評価
    #   2. MID/FWD の CS項を完全撤廃
    #      → チーム守備の恩恵でCSを稼ぐ選手を排除
    #      → MID/FWDは守備アクション（タックル・クリア）のみで評価
    #   3. ポジション補正: GK=1.2, DEF=1.1, MID=0.8, FWD=0.4
    #
    # 結果の妥当性（シミュレーション確認済）:
    #   上位: Sels(GK) > Saliba > Ruben > Guehi > Timber
    #   下位: Salah / Isak / Mbeumo（攻撃的選手は適切に低い）
    #   問題選手の修正: Zinchenko・Tsimikas・N.Gonzalez・De Bruyne → 低評価
    # ─────────────────────────────────────────────────

    FULL_MIN = 3420.0  # フルシーズン換算（38試合 × 90分）

    pos_def_w_new = {"GK": 1.2, "DEF": 1.1, "MID": 0.8, "FWD": 0.4}
    df["_pos_w_def"] = df["position"].map(pos_def_w_new).fillna(0.7)

    saves_p90_   = df["saves"] / p90
    gc_p90_      = df["goals_conceded"] / p90
    # CS × 出場時間比率（少出場でのCS高評価を防ぐ）
    cs_weighted_ = df["clean_sheets"] * (df["minutes"] / FULL_MIN)
    def_actions  = (
        df["tackles"]
        + df["recoveries"]
        + df["clearances_blocks_interceptions"]
    )
    def_act_p90_ = def_actions / p90

    is_gk_  = df["position"] == "GK"
    is_def_ = df["position"] == "DEF"
    is_mid_ = df["position"] == "MID"

    df["②守備プロセス_raw"] = np.where(
        is_gk_,
        # GK: セーブが主軸、CS（時間補正済）を補助
        saves_p90_    * 2.5
        + cs_weighted_ * 8.0
        - gc_p90_      * 0.5,
        np.where(
            is_def_,
            # DEF: 守備アクション + CS（時間補正済）- 失点
            def_act_p90_  * 0.25
            + cs_weighted_ * 3.0
            - gc_p90_      * 0.6,
            np.where(
                is_mid_,
                # MID: 守備アクションのみ（CS項なし）
                def_act_p90_ * 0.20,
                # FWD: 守備アクションのみ（最小）
                def_act_p90_ * 0.08,
            )
        )
    ) * df["_pos_w_def"]

    # ── ③ 得点近接 ──────────────────────────────────
    xG_p90    = df["expected_goals"] / p90
    thr_max   = df["threat"].max()
    threat_n  = df["threat"] / (thr_max if thr_max > 0 else 1)
    goals_p90 = df["goals_scored"] / p90

    df["③得点近接_raw"] = (
        xG_p90    * 3.0
        + threat_n  * 2.0
        + goals_p90 * 1.0
    )

    # ── ④ 失点近接 ──────────────────────────────────
    saves_p90 = df["saves"] / p90
    gc_p90    = df["goals_conceded"] / p90
    is_gk     = df["position"] == "GK"
    is_def    = df["position"] == "DEF"

    df["④失点近接_raw"] = np.where(
        is_gk,
        saves_p90 * 2.0 + df["clean_sheets"] * 0.5 - df["red_cards"] * 2.0,
        np.where(
            is_def,
            df["clean_sheets"] * 0.8 - gc_p90 * 0.5 - df["red_cards"] * 2.0,
            df["clean_sheets"] * 0.3 - df["red_cards"] * 1.0,
        )
    )

    # ── ⑤ Luck スコア ────────────────────────────────
    df["⑤得点Luck"] = df["goals_scored"] - df["expected_goals"]
    df["⑤守備Luck"] = df["expected_goals_conceded"] - df["goals_conceded"]
    df["⑤Luck合計"] = df["⑤得点Luck"] + df["⑤守備Luck"]

    # ── Z標準化（①②のみ）────────────────────────────
    for raw, norm in [
        ("①攻撃プロセス_raw", "①攻撃プロセス"),
        ("②守備プロセス_raw", "②守備プロセス"),
    ]:
        mu, sd = df[raw].mean(), df[raw].std()
        df[norm] = (df[raw] - mu) / (sd if sd > 0 else 1.0)

    df["③得点近接"]             = df["③得点近接_raw"]
    df["④失点近接"]             = df["④失点近接_raw"]
    df["総合プロセス(①+②)"]    = df["①攻撃プロセス"] + df["②守備プロセス"]
    df["総合クリティカル(③+④)"] = df["③得点近接"] + df["④失点近接"]

    # 不要な一時列を削除
    df.drop(columns=["_pos_w_atk", "_pos_w_def"], inplace=True, errors="ignore")

    return df


# =========================================================
# レーダーチャート
# =========================================================
RADAR_DIMS   = ["①攻撃プロセス","②守備プロセス","③得点近接","④失点近接","総合クリティカル(③+④)"]
RADAR_LABELS = ["① Attack\nProcess","② Defense\nProcess","③ Goal\nThreat","④ Save\nContrib","Critical\nTotal"]

def radar_chart(df_sel: pd.DataFrame, pool_df: pd.DataFrame) -> plt.Figure:
    dims   = [d for d in RADAR_DIMS   if d in df_sel.columns]
    labels = [RADAR_LABELS[RADAR_DIMS.index(d)] for d in dims]
    if len(dims) < 3:
        return None

    # パーセンタイル変換
    df_pct = df_sel[dims].copy()
    for col in dims:
        pool_vals = pool_df[col].dropna()
        df_pct[col] = df_pct[col].apply(lambda v: float((pool_vals <= v).mean()))

    n      = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    palette = plt.cm.Set2(np.linspace(0, 0.85, len(df_pct)))
    patches = []

    for (idx, row), color in zip(df_pct.iterrows(), palette):
        vals = row[dims].tolist() + [row[dims[0]]]
        name = str(df_sel.loc[idx, "player_name"])[:20] if "player_name" in df_sel.columns else str(idx)
        ax.plot(angles, vals, "o-", lw=2, color=color, alpha=0.9)
        ax.fill(angles, vals, alpha=0.13, color=color)
        patches.append(mpatches.Patch(color=color, label=name))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color=COLORS["dark"], fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["25%", "50%", "75%"], size=7, color=COLORS["muted"])
    ax.tick_params(pad=10)
    ax.spines["polar"].set_color("#CBD5E1")
    ax.grid(color="#CBD5E1", linewidth=0.6)
    ax.set_title("EPL Percentile Radar", size=10, color=COLORS["dark"], pad=20, fontweight="bold")
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8)
    plt.tight_layout()
    return fig


# =========================================================
# メイン
# =========================================================
def main():
    # ── ヘッダー ─────────────────────────────────────────
    st.markdown("""
    <div class="fpl-header">
      <h1>⚽ FPL 5-Metric Analytics Dashboard</h1>
      <p>Premier League · Individual Player Level · New 5 Metrics Framework</p>
    </div>
    """, unsafe_allow_html=True)

    # ── サイドバー: シーズン選択 ──────────────────────────
    st.sidebar.markdown(f"""
    <div style="padding:.8rem 0 .5rem">
      <div style="color:{COLORS['accent1']};font-size:1.1rem;font-weight:900;font-family:'Arial Black'">
        ⚽ FPL Analytics
      </div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("---")
    season = st.sidebar.selectbox(
        "シーズン選択",
        list(SEASONS.keys()),
        index=0,
        help="vaastav GitHub から取得します"
    )

    # ── データ取得 ────────────────────────────────────────
    status_placeholder = st.empty()
    with status_placeholder.container():
        st.info(f"📥 {season} シーズンのデータを取得中...")

    df_raw, source = fetch_vaastav_players(season)
    team_df        = fetch_vaastav_teams(season)
    fpl_api        = fetch_fpl_api()       # チーム名補完用（失敗してもOK）

    status_placeholder.empty()

    # ── 取得失敗時の案内 ─────────────────────────────────
    if df_raw is None:
        st.error(f"""
        ### ⚠️  データを取得できませんでした

        **原因と対処法:**

        **A. インターネット接続の問題**
        → 接続を確認してください

        **B. GitHubへのアクセスがブロックされている**
        → 下記URLをブラウザで開き、CSVを手動ダウンロードしてください:
        ```
        https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv
        ```
        ダウンロードしたファイルを **`players_raw_{season.replace('-','_')}.csv`** という名前でこのスクリプトと同じフォルダに置いてください。

        **C. FPL API のみに依存したい場合**
        → オフシーズン中（6〜8月）はAPIが選手データを返さないため、このアプリはGitHub CSVを必要とします。
        """)
        st.stop()

    # ── 整形 ─────────────────────────────────────────────
    df_players = prepare_players(df_raw, team_df, fpl_api)

    # データソースバッジ
    badge_class = "badge-live" if source == "github" else "badge-local"
    badge_text  = "✅ GitHub Live" if source == "github" else "📁 Local CSV"
    st.markdown(
        f'<span class="data-badge {badge_class}">{badge_text} — {season}</span>',
        unsafe_allow_html=True
    )

    # ── サイドバー: フィルター ────────────────────────────
    st.sidebar.markdown("---")
    min_min = st.sidebar.slider("最低出場分数", 90, 3000, 450, 90)

    df_metrics = compute_metrics(df_players, min_minutes=min_min)

    if df_metrics.empty:
        st.warning(f"出場{min_min}分以上の選手が見つかりません。最低出場分数を下げてください。")
        st.stop()

    st.sidebar.markdown("---")
    pos_opts  = sorted(df_metrics["position"].unique().tolist())
    pos_filter = st.sidebar.multiselect("ポジション", pos_opts, default=pos_opts)

    team_opts  = sorted(df_metrics["team_name"].unique().tolist())
    team_filter = st.sidebar.multiselect("チーム", team_opts, default=team_opts)

    df_filt = df_metrics[
        df_metrics["position"].isin(pos_filter)
        & df_metrics["team_name"].isin(team_filter)
    ].copy()

    st.sidebar.markdown(
        f"<div style='color:#94A3B8;font-size:.8rem'>対象選手: <b style='color:{COLORS['accent1']}'>{len(df_filt)}</b></div>",
        unsafe_allow_html=True
    )

    # ── ページ選択 ────────────────────────────────────────
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "ページ",
        ["🏟️ リーグランキング", "👤 選手比較", "📊 指標検証"],
        label_visibility="collapsed",
    )

    # ==========================================================
    # ページ A: リーグランキング
    # ==========================================================
    if page == "🏟️ リーグランキング":
        st.markdown("<div class='section-title'>🏆 リーグ全体 — 新5大指標ランキング</div>", unsafe_allow_html=True)
        st.caption("①②は全選手Zスコア標準化（0 = リーグ平均 / +1.0 = 上位16%）")

        sort_col = st.selectbox("ソート指標", [
            "総合プロセス(①+②)", "①攻撃プロセス", "②守備プロセス",
            "③得点近接", "④失点近接", "総合クリティカル(③+④)",
            "⑤得点Luck", "⑤守備Luck", "total_points",
        ])

        show = [
            "player_name", "team_name", "position", "minutes",
            "①攻撃プロセス", "②守備プロセス", "総合プロセス(①+②)",
            "③得点近接", "④失点近接", "総合クリティカル(③+④)",
            "⑤得点Luck", "⑤守備Luck",
            "expected_goals", "expected_assists", "goals_scored", "assists",
            "saves", "clean_sheets", "total_points", "price_m",
        ]
        show = [c for c in show if c in df_filt.columns]

        df_show = (
            df_filt[show]
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True)
        )
        df_show.index += 1

        rename = {
            "player_name":"選手名","team_name":"チーム","position":"POS",
            "minutes":"出場分","total_points":"FPL得点","price_m":"£M",
            "expected_goals":"xG","expected_assists":"xA",
            "goals_scored":"G","assists":"A","saves":"Saves","clean_sheets":"CS",
        }
        df_show = df_show.rename(columns=rename)

        pos_color = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}

        def pos_style(val):
            c = pos_color.get(val, "#64748B")
            return f"background-color:{c};color:white;font-weight:700;border-radius:4px;text-align:center"

        fmt_plus = {c: "{:+.2f}" for c in [
            "①攻撃プロセス","②守備プロセス","総合プロセス(①+②)",
            "③得点近接","④失点近接","総合クリティカル(③+④)",
            "⑤得点Luck","⑤守備Luck",
        ] if c in df_show.columns}
        fmt_float = {c: "{:.2f}" for c in ["xG","xA","£M"] if c in df_show.columns}

        styled = (
            df_show.style
            .background_gradient(
                subset=[c for c in ["①攻撃プロセス","②守備プロセス","総合プロセス(①+②)"] if c in df_show.columns],
                cmap="RdYlGn"
            )
            .background_gradient(
                subset=[c for c in ["③得点近接","総合クリティカル(③+④)"] if c in df_show.columns],
                cmap="Purples"
            )
            .background_gradient(
                subset=[c for c in ["⑤得点Luck","⑤守備Luck"] if c in df_show.columns],
                cmap="coolwarm", vmin=-5, vmax=5
            )
            .format({**fmt_plus, **fmt_float})
        )
        styled = _styler_map(styled, pos_style, subset=["POS"])
        st.dataframe(styled, use_container_width=True, height=520)

        # ポジション別サマリー
        st.markdown("<div class='section-title'>📊 ポジション別サマリー</div>", unsafe_allow_html=True)
        pos_sum = df_filt.groupby("position").agg(
            選手数=("player_name","count"),
            攻撃Avg=("①攻撃プロセス","mean"),
            守備Avg=("②守備プロセス","mean"),
            得点近接Avg=("③得点近接","mean"),
            失点近接Avg=("④失点近接","mean"),
            LuckAvg=("⑤得点Luck","mean"),
            xGAvg=("expected_goals","mean"),
            xAAvg=("expected_assists","mean"),
            FPL得点Avg=("total_points","mean"),
        ).round(3).reset_index().rename(columns={"position":"ポジション"})
        st.dataframe(
            pos_sum.style.background_gradient(
                subset=["攻撃Avg","守備Avg","得点近接Avg","失点近接Avg"], cmap="RdYlGn"
            ).format({c:"{:.3f}" for c in pos_sum.columns if pos_sum[c].dtype == float}),
            use_container_width=True
        )

    # ==========================================================
    # ページ B: 選手比較
    # ==========================================================
    elif page == "👤 選手比較":
        st.markdown("<div class='section-title'>🔍 選手比較・レーダーチャート</div>", unsafe_allow_html=True)

        all_players = sorted(df_filt["player_name"].tolist())
        sel = st.multiselect(
            "比較選手を選んでください（2〜5名）",
            all_players,
            default=all_players[:3] if len(all_players) >= 3 else all_players,
        )
        if not sel:
            st.info("選手を選択してください")
            return

        df_sel = df_filt[df_filt["player_name"].isin(sel)].copy().set_index("player_name")

        # 選手カード
        cols = st.columns(min(len(sel), 5))
        for col, player in zip(cols, sel):
            if player not in df_sel.index:
                continue
            row  = df_sel.loc[player]
            pos  = row.get("position","?")
            pc   = {"GK":"#F59E0B","DEF":"#3B82F6","MID":"#8B5CF6","FWD":"#EF4444"}.get(pos,"gray")
            with col:
                st.markdown(f"""
                <div style="background:{COLORS['dark']};padding:.9rem;border-radius:10px;
                            text-align:center;border-left:4px solid {COLORS['primary']}">
                  <div style="font-size:.65rem;color:{pc};font-weight:700;text-transform:uppercase">{pos}</div>
                  <div style="font-size:1rem;font-weight:900;color:white;margin:.2rem 0">{player}</div>
                  <div style="font-size:.7rem;color:#94A3B8">{row.get("team_name","")}</div>
                  <div style="display:flex;justify-content:space-around;margin-top:.7rem">
                    <div><div style="font-size:1.1rem;font-weight:900;color:{COLORS['accent1']}">{int(row.get('total_points',0))}</div>
                         <div style="font-size:.6rem;color:#64748B">FPL Pts</div></div>
                    <div><div style="font-size:1.1rem;font-weight:900;color:{COLORS['primary']}">{int(row.get('minutes',0))}</div>
                         <div style="font-size:.6rem;color:#64748B">Mins</div></div>
                    <div><div style="font-size:1.1rem;font-weight:900;color:#60A5FA">£{row.get('price_m',0):.1f}M</div>
                         <div style="font-size:.6rem;color:#64748B">Price</div></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_r, col_t = st.columns([1, 1])

        with col_r:
            fig_r = radar_chart(df_sel.reset_index(), df_filt)
            if fig_r:
                st.pyplot(fig_r, use_container_width=True)
                st.caption("外側 = EPL全選手中で高いパーセンタイル")

        with col_t:
            st.markdown("**数値比較**")
            cmp_cols = [
                "①攻撃プロセス","②守備プロセス","総合プロセス(①+②)",
                "③得点近接","④失点近接","総合クリティカル(③+④)",
                "⑤得点Luck","⑤守備Luck",
                "expected_goals","expected_assists","goals_scored","assists",
                "saves","clean_sheets","bonus","total_points","price_m",
            ]
            cmp_cols = [c for c in cmp_cols if c in df_sel.columns]
            df_cmp = df_sel[cmp_cols].T
            df_cmp.index.name = "指標"
            st.dataframe(
                df_cmp.style
                .background_gradient(axis=1, cmap="RdYlGn")
                .format(lambda v: f"{v:+.2f}" if isinstance(v, float) else str(v)),
                use_container_width=True, height=430
            )

        # GW推移
        st.markdown("<div class='section-title'>📈 GW別 FPL得点推移</div>", unsafe_allow_html=True)

        with st.expander("GWデータをロード"):
            gw_df = fetch_vaastav_gw(season)
            if gw_df is not None:
                st.success(f"✅ GWデータ取得完了（{len(gw_df)}行）")

                # 名前マッチング（web_name と name 両方試す）
                name_col = "name" if "name" in gw_df.columns else (
                    "web_name" if "web_name" in gw_df.columns else None
                )
                if name_col:
                    gw_sel = gw_df[gw_df[name_col].isin(sel)].copy()
                    round_col = "round" if "round" in gw_sel.columns else "GW"

                    if not gw_sel.empty:
                        fig_gw, ax_gw = plt.subplots(figsize=(10, 3.5))
                        fig_gw.patch.set_facecolor(COLORS["bg"])
                        ax_gw.set_facecolor(COLORS["bg"])
                        clrs = plt.cm.Set2(np.linspace(0, 0.85, len(sel)))
                        for player, c in zip(sel, clrs):
                            sub = gw_sel[gw_sel[name_col]==player].sort_values(round_col)
                            if not sub.empty:
                                ax_gw.plot(sub[round_col], sub["total_points"],
                                           "o-", lw=2, ms=5, label=player, color=c)
                        ax_gw.set_xlabel("Gameweek"); ax_gw.set_ylabel("FPL Points")
                        ax_gw.set_title("FPL Points per Gameweek", fontweight="bold")
                        ax_gw.legend(fontsize=8)
                        ax_gw.grid(color="#CBD5E1", lw=0.5, alpha=0.7)
                        plt.tight_layout()
                        st.pyplot(fig_gw, use_container_width=True)

                        # xG / xA GW別
                        for metric, title in [
                            ("expected_goals","xG per GW"), ("expected_assists","xA per GW")
                        ]:
                            if metric not in gw_sel.columns:
                                continue
                        fig_x, axs = plt.subplots(1, 2, figsize=(10, 3))
                        fig_x.patch.set_facecolor(COLORS["bg"])
                        for ax, metric, title in [
                            (axs[0],"expected_goals","xG per GW"),
                            (axs[1],"expected_assists","xA per GW")
                        ]:
                            ax.set_facecolor(COLORS["bg"])
                            if metric not in gw_sel.columns:
                                continue
                            for i, (player, c) in enumerate(zip(sel, clrs)):
                                sub = gw_sel[gw_sel[name_col]==player].sort_values(round_col)
                                if sub.empty: continue
                                vals = pd.to_numeric(sub[metric], errors="coerce").fillna(0)
                                ax.bar(sub[round_col] + i*0.2, vals, 0.18,
                                       label=player, color=c, alpha=0.85)
                            ax.set_title(title, fontweight="bold", fontsize=9)
                            ax.set_xlabel("GW"); ax.grid(axis="y", color="#CBD5E1", lw=0.5)
                            if ax is axs[0]: ax.legend(fontsize=7)
                        plt.tight_layout()
                        st.pyplot(fig_x, use_container_width=True)
                    else:
                        st.info("選択した選手のGWデータが見つかりませんでした（名前が一致しない場合があります）")
            else:
                st.warning(f"""
                GWデータの取得に失敗しました。

                手動でダウンロードする場合:
                `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv`
                → `merged_gw_{season.replace('-','_')}.csv` として保存
                """)

    # ==========================================================
    # ページ C: 指標検証
    # ==========================================================
    elif page == "📊 指標検証":
        st.markdown("<div class='section-title'>📊 新5大指標 vs 既存指標 検証</div>", unsafe_allow_html=True)
        st.caption(f"{season}シーズン {min_min}分以上出場選手 (n={len(df_filt)})")
        st.info("""
        **② 守備プロセス v2 改善点**  
        旧版: `influence × 2.0 × FWD補正1.5` → 攻撃的FW/MFが高得点（誤り）  
        新版: `saves/90 + CS率 + def_actions/90 − GC/90` × `GK=1.2 / DEF=1.1 / MID=0.8 / FWD=0.4`  
        → GK・DFが適切に高評価、サラー・イサク等の攻撃免除選手は低スコアに
        """)

        # ── 相関表 ────────────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**① ② プロセス vs 既存指標の相関**")
            targets_p = [
                ("expected_goal_involvements","xGI（得点関与期待）"),
                ("expected_assists","xA（アシスト期待）"),
                ("expected_goals","xG（得点期待）"),
                ("creativity","Creativity（FPL）"),
                ("influence","Influence（FPL）"),
                ("ict_index","ICT Index（FPL）"),
                ("total_points","FPL総得点"),
            ]
            rows = []
            for col_name, label in targets_p:
                if col_name not in df_filt.columns: continue
                sub = df_filt[["総合プロセス(①+②)", col_name]].dropna()
                if len(sub) < 10: continue
                r, p = pearsonr(sub["総合プロセス(①+②)"], sub[col_name])
                sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else ""))
                rows.append({"指標":label, "r":round(r,3), "sig":sig})
            if rows:
                df_p = pd.DataFrame(rows)
                st.dataframe(
                    df_p.style
                    .background_gradient(subset=["r"], cmap="RdYlGn", vmin=-1, vmax=1)
                    .format({"r":"{:+.3f}"}),
                    use_container_width=True, height=280
                )

        with col_r:
            st.markdown("**③ 得点近接 vs 既存指標の相関**")
            targets_c = [
                ("goals_scored","実際のゴール数"),
                ("assists","実際のアシスト"),
                ("expected_goals","xG"),
                ("threat","Threat（FPL）"),
                ("bonus","ボーナス得点"),
                ("total_points","FPL総得点"),
            ]
            rows_c = []
            for col_name, label in targets_c:
                if col_name not in df_filt.columns: continue
                sub = df_filt[["③得点近接", col_name]].dropna()
                if len(sub) < 10: continue
                r, p = pearsonr(sub["③得点近接"], sub[col_name])
                sig = "***" if p<.001 else ("**" if p<.01 else ("*" if p<.05 else ""))
                rows_c.append({"指標":label, "r":round(r,3), "sig":sig})
            if rows_c:
                df_c = pd.DataFrame(rows_c)
                st.dataframe(
                    df_c.style
                    .background_gradient(subset=["r"], cmap="RdYlGn", vmin=-1, vmax=1)
                    .format({"r":"{:+.3f}"}),
                    use_container_width=True, height=260
                )

        # ── AUC ──────────────────────────────────────────
        st.markdown("<div class='section-title'>🎯 FPL高得点予測 AUC</div>", unsafe_allow_html=True)
        st.caption("上位50%の選手を予測できるか（0.5 = ランダム、1.0 = 完全予測）")

        y = (df_filt["total_points"] >= df_filt["total_points"].median()).astype(int)
        n_sp = min(5, max(2, len(df_filt)//10))
        cv = StratifiedKFold(n_splits=n_sp, shuffle=True, random_state=42)

        auc_items = [
            ("総合プロセス(①+②)",    "🆕 ①+② プロセス合計",  True),
            ("③得点近接",             "🆕 ③ 得点近接",         True),
            ("④失点近接",             "🆕 ④ 失点近接",         True),
            ("総合クリティカル(③+④)", "🆕 ③+④ クリティカル",  True),
            ("⑤得点Luck",             "🆕 ⑤ 得点Luck",         True),
            ("expected_goals",         "📌 xG（既存）",          False),
            ("expected_assists",       "📌 xA（既存）",          False),
            ("expected_goal_involvements","📌 xGI（既存）",      False),
            ("ict_index",              "📌 ICT Index（既存）",   False),
            ("threat",                 "📌 Threat（既存）",      False),
            ("creativity",             "📌 Creativity（既存）",  False),
            ("influence",              "📌 Influence（既存）",   False),
        ]
        auc_res = []
        for m_col, label, is_new in auc_items:
            if m_col not in df_filt.columns: continue
            X = StandardScaler().fit_transform(df_filt[[m_col]].fillna(0))
            try:
                auc = cross_val_score(
                    LogisticRegression(max_iter=1000), X, y,
                    cv=cv, scoring="roc_auc"
                ).mean()
                auc_res.append({"指標":label,"AUC":round(auc,3),"is_new":is_new})
            except Exception:
                pass

        if auc_res:
            df_auc = pd.DataFrame(auc_res).sort_values("AUC", ascending=False)
            fig_a, ax_a = plt.subplots(figsize=(9, 5))
            fig_a.patch.set_facecolor(COLORS["bg"])
            ax_a.set_facecolor(COLORS["bg"])
            clrs_a = [COLORS["primary"] if r else "#94A3B8" for r in df_auc["is_new"]]
            bars = ax_a.barh(
                df_auc["指標"].tolist()[::-1],
                df_auc["AUC"].tolist()[::-1],
                color=clrs_a[::-1], edgecolor="white", linewidth=0.5
            )
            ax_a.axvline(0.5, color=COLORS["accent2"], ls="--", lw=1.5, label="Random (0.5)")
            ax_a.set_xlabel("AUC")
            ax_a.set_title("FPL High Score Prediction — New vs Existing", fontweight="bold")
            ax_a.set_xlim(0.3, 1.0)
            ax_a.grid(axis="x", color="#CBD5E1", lw=0.5)
            for bar, val in zip(bars[::-1], df_auc["AUC"]):
                ax_a.text(val+.005, bar.get_y()+bar.get_height()/2,
                          f"{val:.3f}", va="center", fontsize=8)
            ax_a.legend(handles=[
                mpatches.Patch(color=COLORS["primary"], label="🆕 新指標"),
                mpatches.Patch(color="#94A3B8", label="📌 既存指標"),
            ], fontsize=9, loc="lower right")
            plt.tight_layout()
            st.pyplot(fig_a, use_container_width=True)

            st.dataframe(
                df_auc[["指標","AUC"]].style
                .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.4, vmax=0.9)
                .format({"AUC":"{:.3f}"}),
                use_container_width=True, height=370
            )

        # ── 相関マトリクス ───────────────────────────────
        st.markdown("<div class='section-title'>🗺️ 指標間 相関マトリクス</div>", unsafe_allow_html=True)
        hm = [
            "①攻撃プロセス","②守備プロセス","③得点近接","④失点近接","⑤得点Luck",
            "expected_goals","expected_assists","expected_goal_involvements",
            "ict_index","threat","creativity","total_points",
        ]
        hm = [c for c in hm if c in df_filt.columns]
        lbl_map = {
            "①攻撃プロセス":"①Atk","②守備プロセス":"②Def","③得点近接":"③GThr",
            "④失点近接":"④Save","⑤得点Luck":"⑤Luck",
            "expected_goals":"xG","expected_assists":"xA",
            "expected_goal_involvements":"xGI","ict_index":"ICT",
            "threat":"Threat","creativity":"Creat","total_points":"FPLPts",
        }
        fig_h, ax_h = plt.subplots(figsize=(10, 8))
        fig_h.patch.set_facecolor(COLORS["bg"])
        sns.heatmap(
            df_filt[hm].rename(columns=lbl_map).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=ax_h, annot_kws={"size":8}, linewidths=0.3, square=True
        )
        ax_h.set_title("Correlation Matrix — New 5 Metrics vs FPL Existing",
                       fontsize=10, fontweight="bold", pad=12)
        ax_h.tick_params(axis="x", labelsize=8, rotation=45)
        ax_h.tick_params(axis="y", labelsize=8, rotation=0)
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)

    # ── フッター ──────────────────────────────────────────
    st.markdown("""
    <div class="fpl-footer">
      Data: <b>vaastav/Fantasy-Premier-League</b> (github.com/vaastav/Fantasy-Premier-League)
      &amp; <b>FPL Official API</b> (fantasy.premierleague.com) ·
      FPL data is the property of the Premier League · Non-commercial personal use only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
