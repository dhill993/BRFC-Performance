# pages/2_Teams.py
import os
import pathlib
import uuid
import shutil
import time
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
import streamlit as st
from utils import get_statsbomb_creds
from Benchmarks.team_benchmark import (
    build_team_comparison,
    render_team_comparison_table_ranked,
)
from Config.config_metrics import team_metrics  # friendly labels for team metrics

st.set_page_config(layout="wide", page_title="Teams")

# ---------- CSS ----------
def load_css(file_path: pathlib.Path):
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(pathlib.Path("assets/styles.css"))

# =========================
# In-session cross-page caches
# =========================
if "comps_cache" not in st.session_state:
    st.session_state["comps_cache"] = None           # competitions df
if "valid_team_pairs_cache" not in st.session_state:
    st.session_state["valid_team_pairs_cache"] = None  # set of (comp_id, season_id)
if "teams_cache" not in st.session_state:
    st.session_state["teams_cache"] = {}             # {(comp_id, season_id): DataFrame}

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
        # clear in-memory caches too
        st.session_state["teams_cache"].clear()
        st.session_state["comps_cache"] = None
        st.session_state["valid_team_pairs_cache"] = None
        st.success("Cleared this session’s temp cache.")
    except Exception as e:
        st.error(f"Failed to clear session cache: {e}")


SB_USERNAME, SB_PASSWORD = get_statsbomb_creds()
if not SB_USERNAME or not SB_PASSWORD:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()

username,password=SB_USERNAME,SB_PASSWORD
# ---------- Helpers: League -> Season (team-stats) ----------
def _norm_season(name: str) -> str:
    """Convert '2024/2025' -> '2024-25'."""
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
    df["season_norm"] = df["season_name"].map(_norm_season)
    df["league_key"]   = df["country_name"].astype(str) + " | " + df["competition_name"].astype(str)
    df["league_label"] = df["country_name"].astype(str) + " - " + df["competition_name"].astype(str)
    return df

def probe_team_pairs_progress(season_df: pd.DataFrame, user: str, pwd: str) -> set[tuple]:
    """
    Return set of (competition_id, season_id) pairs that actually have team-stats.
    Probes v2 then v4 with a progress bar.
    """
    valids = []
    with st.status("Probing leagues for published team data...", expanded=True) as status:
        total = len(season_df)
        prog = st.progress(0.0)
        for i, (_, r) in enumerate(season_df.iterrows(), start=1):
            cid, sid = int(r["competition_id"]), int(r["season_id"])
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
                                break
                        except Exception:
                            pass
                except requests.RequestException:
                    pass
            status.update(label=f"Checked {i}/{total} season pairs")
            prog.progress(i / max(1, total))
        status.update(label="Finished probing seasons", state="complete")
    return set(valids)

def fetch_teams_for_pairs_progress(pairs_df: pd.DataFrame, user, pwd) -> pd.DataFrame:
    """
    Fetch team-stats for given (competition_id, season_id) rows with a progress bar.
    Tries v2 then v4 for each pair.
    """
    frames = []
    with st.status("Fetching team stats...", expanded=True) as status:
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
                    r = requests.get(
                        url,
                        auth=HTTPBasicAuth(user, pwd),
                        timeout=30,
                        headers={"Accept": "application/json"},
                    )
                except requests.RequestException:
                    r = None
                if r is None or r.status_code != 200:
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

def _rank_class(r: float, n_teams: int) -> str:
    if pd.isna(r):
        return ""
    r = int(r)
    top = max(1, round(0.25 * n_teams))
    mid = max(1, round(0.75 * n_teams))
    if 1 <= r <= top:
        return "rank-green"
    if r <= mid:
        return "rank-amber"
    return "rank-red"

def build_negative_override(df_in: pd.DataFrame, minutes_col: str = "team_season_minutes") -> list:
    metric_cols = [c for c in df_in.columns if c not in ("team_name", minutes_col)]
    neg = []
    for c in metric_cols:
        lc = c.lower()
        if any(k in lc for k in ("conceded", "allowed", "against", "faced", "opp_", "opponent", "against_", "shots_against", "pressures_against")):
            neg.append(c)
        if ("corner" in lc or "corners" in lc) and any(t in lc for t in ("conceded", "against", "allowed", "faced")):
            neg.append(c)
        if any(k in lc for k in ("yellow_cards", "red_cards", "cards", "fouls")):
            neg.append(c)
        if "ppda" in lc:
            neg.append(c)
    return sorted(set(neg), key=neg.index)

# ---------- unified pair getter (memory -> CSV -> API) ----------
def get_teams_pair_cached(cid: int, sid: int, meta: dict, user: str, pwd: str) -> pd.DataFrame:
    key = (int(cid), int(sid))

    # 1) in-memory cache (survives page switches)
    if key in st.session_state["teams_cache"]:
        return st.session_state["teams_cache"][key].copy()

    # 2) per-session CSV cache
    cached_csv = load_session_csv("teams", cid, sid)
    if cached_csv is not None and not cached_csv.empty:
        df = cached_csv.copy()
        # re-attach meta
        for k in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]:
            df[k] = meta.get(k, "")
        df["competition_id"] = cid
        df["season_id"] = sid
        st.session_state["teams_cache"][key] = df.copy()
        return df

    # 3) fetch from API (single pair), then persist to CSV + memory
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
    # save a slim CSV per pair without meta
    save_session_csv(
        df_pair.drop(columns=[c for c in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]
                              if c in df_pair.columns]),
        "teams", cid, sid
    )
    st.session_state["teams_cache"][key] = df_pair.copy()
    return df_pair

# ---------- UI ----------
st.title("🏴 Teams")

# Small cache controls
with st.expander("⚙️ Session temp cache"):
    c1, c2 = st.columns(2)
    c1.write(session_files_list())
    if c2.button("🧹 Clear this session’s temp cache"):
        clear_this_session_cache()

# 1) Competitions (load once per session)
if st.session_state["comps_cache"] is None:
    with st.status("Loading competitions...", expanded=False) as status:
        st.session_state["comps_cache"] = fetch_competitions(username, password)
        status.update(label="Competitions loaded", state="complete")

comps = st.session_state["comps_cache"]
if comps is None or comps.empty:
    st.warning("No competitions available.")
    st.stop()

# 1b) Probe valid team pairs (once per session)
if st.session_state["valid_team_pairs_cache"] is None:
    st.session_state["valid_team_pairs_cache"] = probe_team_pairs_progress(comps, username, password)

valid_pairs = st.session_state["valid_team_pairs_cache"]
if not valid_pairs:
    st.warning("No leagues with published team data found.")
    st.stop()

comps["pair"] = list(zip(comps["competition_id"].astype(int), comps["season_id"].astype(int)))
comps_valid = comps[comps["pair"].isin(valid_pairs)].copy()

league_options = sorted(comps_valid["league_label"].unique().tolist())
sel_league = st.selectbox("League", league_options)

league_block = comps_valid[comps_valid["league_label"] == sel_league].copy()
league_block = league_block.sort_values("season_norm",
                                        key=lambda s: s.map(_season_sort_key),
                                        ascending=False)

season_options = league_block["season_norm"].tolist()
sel_season = st.selectbox("Season", season_options, index=0)

row = league_block[league_block["season_norm"] == sel_season].iloc[0]

# 2) Load team-stats for seasons in this league (from cache first)
need = league_block[["competition_id","season_id","competition_name",
                     "country_name","season_norm","league_key","league_label"]].drop_duplicates()

frames = []
for _, r in need.iterrows():
    cid, sid = int(r["competition_id"]), int(r["season_id"])
    meta = {
        "competition_name": r.get("competition_name",""),
        "country_name": r.get("country_name",""),
        "season_norm": r.get("season_norm",""),
        "league_key": r.get("league_key",""),
        "league_label": r.get("league_label",""),
    }
    df_pair = get_teams_pair_cached(cid, sid, meta, username, password)
    if not df_pair.empty:
        frames.append(df_pair)

df_team_stats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if df_team_stats.empty:
    st.warning("No team data found for the chosen league or seasons.")
    st.stop()
else:
    st.caption("Loaded from in-memory session cache / temp CSVs ✅")

df = df_team_stats[(df_team_stats["competition_id"] == int(row["competition_id"])) &
                   (df_team_stats["season_id"]      == int(row["season_id"]))].copy()
if df.empty:
    st.warning("No team rows for this league and season.")
    st.stop()

# 3) Team selections (same league and season only)
teams = sorted(df["team_name"].dropna().unique())
c1, c2 = st.columns(2)
with c1:
    t1 = st.selectbox("Select Team 1", teams, key="team1")
with c2:
    t2 = st.selectbox("Compare To", ["Average"] + [t for t in teams if t != t1], key="team2")

# 4) View and grouping
view = st.radio("View", ["Compare", "Percentile ranks"], horizontal=True, index=1)
group_choice = st.radio("Metric group", ["All", "Attack", "Defence", "Discipline"], horizontal=True, index=0)
include_groups = None if group_choice == "All" else [group_choice]

# 5) Build and render
try:
    neg_override = build_negative_override(df, minutes_col="team_season_minutes")

    comparison_df = build_team_comparison(
        df=df,
        team1=t1,
        team2=None if t2 == "Average" else t2,
        pretty=True,
        decimals=2,
        metrics_mapping=team_metrics,
        minutes_col="team_season_minutes",
        include_groups=include_groups,
        negative_features=neg_override,
    )

    if comparison_df.empty:
        st.info(f"No metrics available for the selected group: {group_choice}.")
        st.stop()

    if view == "Compare":
        st.markdown(f"### 📈 Metric by Metric - {sel_league} ({sel_season}) · {group_choice}")
        render_team_comparison_table_ranked(
            comparison_df,
            team1=t1,
            team2=("Average" if t2 == "Average" else t2),
        )
    else:
        st.markdown(f"### 🏅 League ranks and percentiles - {sel_league} ({sel_season}) · {group_choice}")

        r1_col = f"{t1}_rank"
        p1_col = f"{t1}_pct"
        r2_col = f"{t2}_rank" if f"{t2}_rank" in comparison_df.columns else None
        p2_col = f"{t2}_pct" if f"{t2}_pct" in comparison_df.columns else None

        n_teams = df["team_name"].nunique()

        html = [
            "<table class='comparison-table'>",
            "<thead>",
            f"<tr><th>📌 Metric</th>"
            f"<th>🔵 {t1}</th><th>Rank</th><th>%tile</th>"
            f"<th>🟡 {t2}</th><th>Rank</th><th>%tile</th>"
            f"</tr>",
            "</thead><tbody>",
        ]

        for _, rowx in comparison_df.iterrows():
            r1 = rowx.get(r1_col, np.nan); p1 = rowx.get(p1_col, np.nan)
            cls1 = _rank_class(r1, n_teams)
            r1_txt = "" if pd.isna(r1) else str(int(r1))
            p1_txt = "" if pd.isna(p1) else f"{float(p1):.2%}"

            if r2_col and p2_col:
                r2 = rowx.get(r2_col, np.nan); p2 = rowx.get(p2_col, np.nan)
                cls2 = _rank_class(r2, n_teams)
                t2_val = rowx.get(t2, "")
                r2_txt = "" if pd.isna(r2) else str(int(r2))
                p2_txt = "" if pd.isna(p2) else f"{float(p2):.2%}"
            else:
                t2_val, r2_txt, p2_txt, cls2 = "", "", "", ""

            html.append("<tr>")
            html.append(f"<td>{rowx['Metric']}</td>")
            html.append(f"<td class='{cls1}'>{rowx[t1]}</td>")
            html.append(f"<td class='comp-cell {cls1}'>{r1_txt}</td>")
            html.append(f"<td class='comp-cell'>{p1_txt}</td>")
            html.append(f"<td class='{cls2}'>{t2_val}</td>")
            html.append(f"<td class='comp-cell {cls2}'>{r2_txt}</td>")
            html.append(f"<td class='comp-cell'>{p2_txt}</td>")
            html.append("</tr>")

        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)

except ValueError as e:
    st.error(str(e))
