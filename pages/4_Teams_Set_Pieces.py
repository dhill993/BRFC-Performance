# pages/4_Set_Pieces.py
import os
import pathlib
import re
import uuid
import shutil
import time
from utils import get_statsbomb_creds
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import streamlit as st

from Benchmarks.team_benchmark import build_team_comparison, render_team_comparison_table_ranked
from Config.config_metrics import team_metrics  # pretty labels (optional)

st.set_page_config(layout="wide", page_title="Set Pieces")

# ---------- CSS ----------
def load_css(file_path: pathlib.Path):
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(pathlib.Path("assets/styles.css"))

TABLE_CSS = """
<style>
.comparison-table{width:100%;border-collapse:collapse;margin:8px 0}
.comparison-table th,.comparison-table td{border:1px solid #e5e7eb;padding:8px;text-align:left}
.rank-green{background:#dcfce7}
.rank-amber{background:#fef9c3}
.rank-red{background:#fee2e2}
.comp-cell{font-variant-numeric:tabular-nums;text-align:center}
</style>
"""
st.markdown(TABLE_CSS, unsafe_allow_html=True)

# =========================
# In-session cross-page caches
# =========================
if "comps_cache" not in st.session_state:
    st.session_state["comps_cache"] = None                 # competitions df
if "valid_team_pairs_cache" not in st.session_state:
    st.session_state["valid_team_pairs_cache"] = None      # set[(cid,sid)]
if "team_stats_cache" not in st.session_state:
    st.session_state["team_stats_cache"] = {}              # {(cid,sid): DataFrame}

# =========================
# Session CSV cache helpers
# =========================
SESS_BASE = pathlib.Path(".tmp_sessions")
SESS_BASE.mkdir(parents=True, exist_ok=True)

def _purge_old_sessions(base: pathlib.Path, max_age_hours: int = 8):
    now = time.time()
    cutoff = now - max_age_hours * 3600
    for p in base.glob("*"):
        try:
            if p.is_dir() and p.stat().st_mtime < cutoff:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass

def _get_session_dir() -> pathlib.Path:
    if "_session_id" not in st.session_state:
        st.session_state["_session_id"] = str(uuid.uuid4())
        _purge_old_sessions(SESS_BASE, max_age_hours=8)
    d = SESS_BASE / st.session_state["_session_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d

def session_csv_path(kind: str, comp_id: int, season_id: int) -> pathlib.Path:
    d = _get_session_dir()
    return d / f"{kind}_{int(comp_id)}_{int(season_id)}.csv"

def load_session_csv(kind: str, comp_id: int, season_id: int) -> pd.DataFrame | None:
    p = session_csv_path(kind, comp_id, season_id)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def save_session_csv(df: pd.DataFrame, kind: str, comp_id: int, season_id: int) -> pathlib.Path:
    p = session_csv_path(kind, comp_id, season_id)
    try:
        df.to_csv(p, index=False)
    except Exception as e:
        st.warning(f"Could not save session CSV: {e}")
    return p

def session_files_list() -> list[str]:
    d = _get_session_dir()
    return [str(p.name) for p in sorted(d.glob("*.csv"))]

def clear_this_session_cache():
    d = _get_session_dir()
    try:
        shutil.rmtree(d, ignore_errors=True)
        st.session_state.pop("_session_id", None)
        st.session_state["team_stats_cache"].clear()
        st.session_state["comps_cache"] = None
        st.session_state["valid_team_pairs_cache"] = None
        st.success("Cleared this session’s temp cache.")
    except Exception as e:
        st.error(f"Failed to clear session cache: {e}")


SB_USERNAME, SB_PASSWORD = get_statsbomb_creds()
if not SB_USERNAME or not SB_PASSWORD:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()
USERNAME=SB_USERNAME
PASSWORD=SB_PASSWORD
# ---------- League → Season helpers ----------
def _norm_season(name: str) -> str:
    """Convert '2024/2025' -> '2024-25' display."""
    if isinstance(name, str) and "/" in name and len(name) >= 7:
        a, b = name.split("/", 1)
        return f"{a}-{b[-2:]}"
    return name

@st.cache_data(show_spinner=False)
def fetch_competitions(user, pwd) -> pd.DataFrame:
    r = requests.get(
        "https://data.statsbomb.com/api/v4/competitions",
        auth=HTTPBasicAuth(user, pwd),
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    keep = ["competition_id", "season_id", "competition_name", "season_name", "country_name"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["season_norm"]  = df["season_name"].map(_norm_season)
    df["league_key"]   = df["country_name"].astype(str) + " | " + df["competition_name"].astype(str)
    df["league_label"] = df["country_name"].astype(str) + " - " + df["competition_name"].astype(str)
    return df

def probe_team_pairs_progress(season_df: pd.DataFrame, user: str, pwd: str) -> set[tuple]:
    """
    Same as your probe function, but with a visual progress bar.
    """
    valids = []
    with st.status("Probing leagues and seasons for available team data...", expanded=True) as status:
        total = len(season_df)
        prog = st.progress(0.0)
        for i, (_, r) in enumerate(season_df.iterrows(), start=1):
            cid, sid = int(r["competition_id"]), int(r["season_id"])
            found = False
            for base in ("v2", "v4"):
                url = f"https://data.statsbomb.com/api/{base}/competitions/{cid}/seasons/{sid}/team-stats"
                try:
                    resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=25)
                    txt = (resp.text or "").strip()
                    if resp.status_code == 200 and txt:
                        try:
                            arr = resp.json()
                            if isinstance(arr, list) and len(arr) > 0:
                                valids.append((cid, sid))
                                found = True
                                break
                        except Exception:
                            pass
                except requests.RequestException:
                    pass
            status.update(label=f"Checked {i}/{total} season pairs")
            prog.progress(i / max(1, total))
        status.update(label="Finished probing seasons", state="complete")
    return set(valid_pairs for valid_pairs in set(valids))

def fetch_teams_for_pairs_progress(pairs_df: pd.DataFrame, user, pwd) -> pd.DataFrame:
    """
    Fetch team-stats for given pairs with a progress bar.
    """
    frames = []
    with st.status("Fetching team stats for selected league seasons...", expanded=True) as status:
        total = len(pairs_df)
        prog = st.progress(0.0)
        for i, (_, comp) in enumerate(pairs_df.iterrows(), start=1):
            cid = int(comp["competition_id"]); sid = int(comp["season_id"])
            urls = [
                f"https://data.statsbomb.com/api/v2/competitions/{cid}/seasons/{sid}/team-stats",
                f"https://data.statsbomb.com/api/v4/competitions/{cid}/seasons/{sid}/team-stats",
            ]
            got = None
            for url in urls:
                try:
                    r = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=30, headers={"Accept":"application/json"})
                except requests.RequestException:
                    continue
                if r.status_code != 200:
                    continue
                txt = (r.text or "").strip()
                if not txt:
                    continue
                try:
                    rows = r.json()
                except Exception:
                    continue
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                if df.empty:
                    continue
                # enrich with context
                df["competition_id"]   = cid
                df["season_id"]        = sid
                df["competition_name"] = comp.get("competition_name","")
                df["country_name"]     = comp.get("country_name","")
                df["season_norm"]      = comp.get("season_norm","")
                df["league_key"]       = comp.get("league_key","")
                df["league_label"]     = comp.get("league_label","")
                got = df
                break
            if got is not None:
                frames.append(got)
            status.update(label=f"Fetched {i}/{total} season blocks")
            prog.progress(i / max(1, total))
        status.update(label="Finished fetching team stats", state="complete")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _season_sort_key(s: str) -> tuple:
    if not isinstance(s, str) or "-" not in s:
        return (0, 0)
    y1, y2 = s.split("-", 1)
    try:
        return (int(y1), int(y2))
    except Exception:
        return (0, 0)

# ---------- Set-piece detection / classification ----------
SP_INCLUDE_TOKENS = [
    "set_piece", "setpiece", "sp_",
    "corner", "corners", "from_corner",
    "throw_in", "throwin",
    "free_kick", "freekick", "fk_", "_fk", "from_fk",
    "direct_free", "dfk",
]
SP_EXCLUDE_TOKENS = ["pen", "penalty", "penalties"]
DEFENCE_TOKENS = ["conceded", "allowed", "against", "faced", "opp_", "opponent", "against_", "_against"]

def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _metric_cols(df: pd.DataFrame, minutes_col: str = "team_season_minutes") -> list[str]:
    base = [c for c in df.columns if _is_numeric(df[c])]
    drop = {"competition_id","season_id","team_id","account_id"}
    if "team_season_minutes" in df.columns:
        idx = list(df.columns).index("team_season_minutes")
        after = df.columns[idx+1:]
        base = [c for c in after if _is_numeric(df[c])]
    return [c for c in base if c not in drop]

def _is_set_piece_key(k: str) -> bool:
    kl = k.lower()
    if any(t in kl for t in SP_EXCLUDE_TOKENS):
        return False
    return any(t in kl for t in SP_INCLUDE_TOKENS)

def _sp_type(k: str) -> str:
    kl = k.lower()
    if any(t in kl for t in ["direct_free", "dfk"]):                       return "Direct FK"
    if any(t in kl for t in ["free_kick", "freekick", "from_fk", "_fk"]):  return "Free Kicks"
    if any(t in kl for t in ["throw_in", "throwin"]):                      return "Throw-ins"
    if any(t in kl for t in ["corner", "corners", "from_corner"]):         return "Corners"
    if any(t in kl for t in ["set_piece", "setpiece", "sp_"]):             return "Other SP"
    return "Other SP"

def _phase(k: str) -> str:
    kl = k.lower()
    return "Defence" if any(t in kl for t in DEFENCE_TOKENS) else "Attack"

def _negatives_for_sp(cols: list[str]) -> set[str]:
    s = set()
    for c in cols:
        cl = c.lower()
        if any(t in cl for t in DEFENCE_TOKENS):
            s.add(c)
        if re.search(r"(against|conceded|allowed|faced)", cl):
            s.add(c)
    return s

# ---------- Cached pair loader (memory -> CSV -> API) ----------
def get_team_pair_cached(cid: int, sid: int, meta: dict, user: str, pwd: str) -> pd.DataFrame:
    key = (int(cid), int(sid))

    # 1) in-memory
    if key in st.session_state["team_stats_cache"]:
        return st.session_state["team_stats_cache"][key].copy()

    # 2) per-session CSV
    cached = load_session_csv("teams", cid, sid)
    if cached is not None and not cached.empty:
        df = cached.copy()
        for k in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]:
            df[k] = meta.get(k, "")
        df["competition_id"] = cid
        df["season_id"] = sid
        st.session_state["team_stats_cache"][key] = df.copy()
        return df

    # 3) fetch from API for just this pair
    to_fetch = pd.DataFrame([{
        "competition_id": cid, "season_id": sid,
        "competition_name": meta.get("competition_name",""),
        "country_name": meta.get("country_name",""),
        "season_norm": meta.get("season_norm",""),
        "league_key": meta.get("league_key",""),
        "league_label": meta.get("league_label",""),
    }])
    fetched = fetch_teams_for_pairs_progress(to_fetch, user, pwd)
    if fetched is None or fetched.empty:
        return pd.DataFrame()

    df_pair = fetched.copy()
    save_session_csv(
        df_pair.drop(columns=[c for c in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]
                              if c in df_pair.columns]),
        "teams", cid, sid
    )
    st.session_state["team_stats_cache"][key] = df_pair.copy()
    return df_pair

# ---------- UI ----------
st.title("🧩 Set Pieces (Teams)")

# 0) Small cache controls
with st.expander("⚙️ Session temp cache"):
    c1, c2 = st.columns(2)
    c1.write(session_files_list())
    if c2.button("🧹 Clear this session’s temp cache"):
        clear_this_session_cache()

# 1) Competitions (cached per session)
if st.session_state["comps_cache"] is None:
    with st.status("Loading competitions...", expanded=False) as status:
        st.session_state["comps_cache"] = fetch_competitions(USERNAME, PASSWORD)
        status.update(label="Competitions loaded", state="complete")

comps = st.session_state["comps_cache"]
if comps.empty:
    st.warning("No competitions available.")
    st.stop()

# 2) Probe seasons that actually have team-stats, with progress (cached per session)
if st.session_state["valid_team_pairs_cache"] is None:
    st.session_state["valid_team_pairs_cache"] = probe_team_pairs_progress(comps, USERNAME, PASSWORD)

valid_pairs = st.session_state["valid_team_pairs_cache"]
if not valid_pairs:
    st.warning("No leagues with published team data found.")
    st.stop()

comps["pair"] = list(zip(comps["competition_id"].astype(int), comps["season_id"].astype(int)))
comps_valid = comps[comps["pair"].isin(valid_pairs)].copy()

league_options = sorted(comps_valid["league_label"].unique().tolist())
sel_league = st.selectbox("League", league_options)

league_block = comps_valid[comps_valid["league_label"] == sel_league].copy()
league_block = league_block.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)

season_options = league_block["season_norm"].tolist()
sel_season = st.selectbox("Season", season_options, index=0)

row_sel = league_block[league_block["season_norm"] == sel_season].iloc[0]
comp_id = int(row_sel["competition_id"])
season_id = int(row_sel["season_id"])

# 3) Load team stats for *this* pair via memory/CSV/API
meta = {
    "competition_name": row_sel.get("competition_name", ""),
    "country_name": row_sel.get("country_name", ""),
    "season_norm": row_sel.get("season_norm", ""),
    "league_key": row_sel.get("league_key", ""),
    "league_label": row_sel.get("league_label", ""),
}
df_team_stats = get_team_pair_cached(comp_id, season_id, meta, USERNAME, PASSWORD)
if df_team_stats.empty:
    st.warning("No team data found for this league/season.")
    st.stop()
else:
    st.caption("Loaded team-season from in-memory session cache / temp CSVs ✅")

df = df_team_stats.copy()

# 4) Detect set-piece metrics and classify
def _metric_cols(df: pd.DataFrame, minutes_col: str = "team_season_minutes") -> list[str]:
    base = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    drop = {"competition_id","season_id","team_id","account_id"}
    if minutes_col in df.columns:
        idx = list(df.columns).index(minutes_col)
        after = df.columns[idx+1:]
        base = [c for c in after if pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in base if c not in drop]

all_metric_cols = _metric_cols(df, minutes_col="team_season_minutes")
sp_metric_keys = [c for c in all_metric_cols if _is_set_piece_key(c)]
if not sp_metric_keys:
    st.info("No set-piece metrics detected in this dataset.")
    st.stop()

sp_types  = {k: _sp_type(k) for k in sp_metric_keys}
sp_phases = {k: _phase(k) for k in sp_metric_keys}

# 5) Team pickers
teams = sorted(df["team_name"].dropna().unique())
c1, c2 = st.columns(2)
with c1:
    t1 = st.selectbox("Team", teams, key="sp_team1")
with c2:
    t2 = st.selectbox("Compare to", ["Average"] + [t for t in teams if t != t1], key="sp_team2")

# 6) Filters
st.markdown("#### Filters")
type_choice = st.radio(
    "Set-piece type",
    ["All", "Corners", "Throw-ins", "Free Kicks", "Direct FK", "Other SP"],
    horizontal=True,
    index=0,
)
phase_choice = st.radio(
    "Phase",
    ["All", "Attack", "Defence"],
    horizontal=True,
    index=0,
)

filtered = sp_metric_keys
if type_choice != "All":
    filtered = [k for k in filtered if sp_types.get(k) == type_choice]
if phase_choice != "All":
    filtered = [k for k in filtered if sp_phases.get(k) == phase_choice]

if not filtered:
    st.info("No metrics match your current filters.")
    st.stop()

neg_set = _negatives_for_sp(filtered)

# 7) Build comparison with status indicator
with st.status("Computing comparison table...", expanded=False) as status:
    try:
        comp_df = build_team_comparison(
            df=df,
            team1=t1,
            team2=None if t2 == "Average" else t2,
            minutes_col="team_season_minutes",
            metrics_mapping=team_metrics,
            negative_features=neg_set,
            include_metrics=filtered,
            include_groups=None,
            pretty=True,
            decimals=2,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()
    status.update(label="Comparison ready", state="complete")

phase_by_key = {k: _phase(k) for k in filtered}
if "MetricKey" in comp_df.columns:
    comp_df["Group"] = comp_df["MetricKey"].map(phase_by_key).fillna("Attack")

# 8) Render
st.caption(f"{sel_league} · {sel_season}  |  {len(filtered)} metrics  |  Type: {type_choice}  |  Phase: {phase_choice}")

view = st.radio("View", ["Compare", "Percentile ranks"], horizontal=True, index=0)

if view == "Compare":
    render_team_comparison_table_ranked(
        comp_df,
        team1=t1,
        team2=("Average" if t2 == "Average" else t2),
    )
else:
    r1_col = f"{t1}_rank"; p1_col = f"{t1}_pct"
    r2_col = f"{t2}_rank" if f"{t2}_rank" in comp_df.columns else None
    p2_col = f"{t2}_pct" if f"{t2}_pct" in comp_df.columns else None
    n_teams = df["team_name"].nunique()

    def _rank_class(r: float, n: int) -> str:
        if pd.isna(r): return ""
        r = int(r)
        top = max(1, round(0.25 * n))
        mid = max(1, round(0.75 * n))
        if 1 <= r <= top: return "rank-green"
        if r <= mid:      return "rank-amber"
        return "rank-red"

    html = [
        "<table class='comparison-table'>",
        "<thead>",
        f"<tr><th>📌 Metric</th>"
        f"<th>🔵 {t1}</th><th>Rank</th><th>%tile</th>"
        f"<th>🟡 {t2}</th><th>Rank</th><th>%tile</th></tr>",
        "</thead><tbody>",
    ]
    for _, row in comp_df.iterrows():
        r1 = row.get(r1_col, np.nan); p1 = row.get(p1_col, np.nan)
        cls1 = _rank_class(r1, n_teams)
        r1_txt = "" if pd.isna(r1) else str(int(r1))
        p1_txt = "" if pd.isna(p1) else f"{float(p1):.2%}"

        if r2_col and p2_col:
            r2 = row.get(r2_col, np.nan); p2 = row.get(p2_col, np.nan)
            cls2 = _rank_class(r2, n_teams)
            t2_val = row.get(t2, "")
            r2_txt = "" if pd.isna(r2) else str(int(r2))
            p2_txt = "" if pd.isna(p2) else f"{float(p2):.2%}"
        else:
            t2_val, r2_txt, p2_txt, cls2 = "", "", "", ""

        html.append("<tr>")
        html.append(f"<td>{row['Metric']}</td>")
        html.append(f"<td class='{cls1}'>{row[t1]}</td>")
        html.append(f"<td class='comp-cell {cls1}'>{r1_txt}</td>")
        html.append(f"<td class='comp-cell'>{p1_txt}</td>")
        html.append(f"<td class='{cls2}'>{t2_val}</td>")
        html.append(f"<td class='comp-cell {cls2}'>{r2_txt}</td>")
        html.append(f"<td class='comp-cell'>{p2_txt}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)

# 9) Debug expander
with st.expander("Show included set-piece metric keys"):
    dbg = pd.DataFrame({
        "MetricKey": filtered,
        "Type": [sp_types[k] for k in filtered],
        "Phase": [sp_phases[k] for k in filtered],
        "Pretty": [team_metrics.get(k, k) for k in filtered],
    }).sort_values(["Phase", "Type", "Pretty"])
    st.dataframe(dbg, hide_index=True)
