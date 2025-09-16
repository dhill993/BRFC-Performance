# pages/3_Similar_Players.py
import os
import pathlib
import uuid
import shutil
import time
from datetime import date
from typing import List, Tuple
from utils import get_statsbomb_creds
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from Config.config_metrics import (
    position_mapping,
    metrics_mapping,
    get_metrics_for_profile,
    add_derived_player_metrics,
)
from Benchmarks.players_benchmark import NEGATIVE_FEATURES

st.set_page_config(layout="wide", page_title="Find Similar Players")

# ---------- CSS ----------
def load_css(file_path: pathlib.Path):
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<style>
.comparison-table{width:100%;border-collapse:collapse;margin:8px 0}
.comparison-table th,.comparison-table td{border:1px solid #e5e7eb;padding:8px;text-align:left}
.simbar-wrap{height:10px;background:#eee;border-radius:6px;overflow:hidden}
.simbar-fill{height:100%;width:0%}
.simbar-fill.good{background:#22c55e}
.simbar-fill.mid{background:#f59e0b}
.simbar-fill.bad{background:#ef4444}
.simval{font-weight:600;font-variant-numeric:tabular-nums}
</style>
""",
    unsafe_allow_html=True,
)
load_css(pathlib.Path("assets/styles.css"))

# =========================
# In-session cross-page caches
# =========================
if "comps_cache" not in st.session_state:
    st.session_state["comps_cache"] = None               # competitions df
if "valid_player_pairs_cache" not in st.session_state:
    st.session_state["valid_player_pairs_cache"] = None  # set[(cid,sid)]
if "players_cache" not in st.session_state:
    st.session_state["players_cache"] = {}               # {(cid,sid): DataFrame}

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
        st.session_state["players_cache"].clear()
        st.session_state["comps_cache"] = None
        st.session_state["valid_player_pairs_cache"] = None
        st.success("Cleared this session’s temp cache.")
    except Exception as e:
        st.error(f"Failed to clear session cache: {e}")

# ---------- ENV ----------
SB_USERNAME, SB_PASSWORD = get_statsbomb_creds()
if not SB_USERNAME or not SB_PASSWORD:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()
username = SB_USERNAME
password = SB_PASSWORD

# ---------- Helpers ----------
def _norm_season(name: str) -> str:
    """Convert '2024/2025' -> '2024-25' style labels."""
    if isinstance(name, str) and "/" in name and len(name) >= 7:
        a, b = name.split("/", 1)
        return f"{a}-{b[-2:]}"
    return str(name)

def _season_sort_key(s: str) -> tuple:
    """Sort '2025-26' chronologically (desc later)."""
    if not isinstance(s, str) or "-" not in s:
        return (0, 0)
    y1, y2 = s.split("-", 1)
    try:
        return (int(y1), int(y2))
    except Exception:
        return (0, 0)

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
    df["league_key"] = df["country_name"].astype(str) + " | " + df["competition_name"].astype(str)
    df["league_label"] = df["country_name"].astype(str) + " - " + df["competition_name"].astype(str)
    return df

def probe_player_pairs_progress(comps_df: pd.DataFrame, user: str, pwd: str) -> set[tuple]:
    """
    Probe which (competition_id, season_id) pairs have player-stats with status and progress.
    """
    valids = []
    with st.status("Probing seasons for published player data...", expanded=True) as status:
        total = len(comps_df)
        prog = st.progress(0.0)
        for i, (_, r) in enumerate(comps_df.iterrows(), start=1):
            cid, sid = int(r["competition_id"]), int(r["season_id"])
            url = f"https://data.statsbomb.com/api/v4/competitions/{cid}/seasons/{sid}/player-stats"
            try:
                resp = requests.get(url, auth=HTTPBasicAuth(user, pwd), timeout=25)
                txt = (resp.text or "").strip()
                if resp.status_code == 200 and txt:
                    try:
                        arr = resp.json()
                        if isinstance(arr, list) and len(arr) > 0:
                            valids.append((cid, sid))
                    except Exception:
                        pass
            except requests.RequestException:
                pass
            status.update(label=f"Checked {i}/{total} season pairs")
            prog.progress(i / max(1, total))
        status.update(label="Finished probing seasons", state="complete")
    return set(valids)

def fetch_players_for_pairs_progress(pairs_df: pd.DataFrame, user, pwd) -> pd.DataFrame:
    """
    Fetch player-stats for the exact pairs with a progress bar.
    """
    frames = []
    with st.status("Fetching player stats for selected seasons...", expanded=True) as status:
        total = len(pairs_df)
        prog = st.progress(0.0)
        for i, (_, comp) in enumerate(pairs_df.iterrows(), start=1):
            cid = int(comp["competition_id"]); sid = int(comp["season_id"])
            url = f"https://data.statsbomb.com/api/v4/competitions/{cid}/seasons/{sid}/player-stats"
            try:
                r = requests.get(
                    url,
                    auth=HTTPBasicAuth(user, pwd),
                    timeout=30,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                txt = (r.text or "").strip()
                if txt:
                    try:
                        rows = r.json()
                        if rows:
                            df = pd.DataFrame(rows)
                            if not df.empty:
                                df["competition_id"] = cid
                                df["season_id"] = sid
                                df["competition_name"] = comp.get("competition_name", "")
                                df["country_name"] = comp.get("country_name", "")
                                df["season_norm"] = comp.get("season_norm", "")
                                df["league_key"] = comp.get("league_key", "")
                                df["league_label"] = comp.get("league_label", "")
                                frames.append(df)
                    except Exception:
                        pass
            status.update(label=f"Fetched {i}/{total} season blocks")
            prog.progress(i / max(1, total))
        status.update(label="Finished fetching player stats", state="complete")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ---------- unified pair getter (memory -> CSV -> API) ----------
def get_players_pair_cached(cid: int, sid: int, meta: dict, user: str, pwd: str) -> pd.DataFrame:
    key = (int(cid), int(sid))

    # 1) in-memory cache (survives page switches)
    if key in st.session_state["players_cache"]:
        return st.session_state["players_cache"][key].copy()

    # 2) per-session CSV cache
    cached_csv = load_session_csv("players", cid, sid)
    if cached_csv is not None and not cached_csv.empty:
        df = cached_csv.copy()
        # re-attach minimal meta for UI/filters
        for k in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]:
            df[k] = meta.get(k, "")
        df["competition_id"] = cid
        df["season_id"] = sid
        st.session_state["players_cache"][key] = df.copy()
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
    fetched = fetch_players_for_pairs_progress(to_fetch, user, pwd)
    if fetched is None or fetched.empty:
        return pd.DataFrame()

    df_pair = fetched.copy()
    # save a slim CSV per pair without meta
    save_session_csv(
        df_pair.drop(columns=[c for c in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]
                              if c in df_pair.columns]),
        "players", cid, sid
    )
    st.session_state["players_cache"][key] = df_pair.copy()
    return df_pair

# ---------- de-dup helpers ----------
def _dedupe_players(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate player-season rows. Keep the row with most minutes."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df["_mins__"] = pd.to_numeric(df.get("player_season_minutes"), errors="coerce").fillna(0)
    if "player_id" in df.columns:
        subset = [c for c in ["player_id", "competition_id", "season_id"] if c in df.columns]
    else:
        subset = [c for c in ["player_name", "competition_id", "season_id"] if c in df.columns]
    if subset:
        df = (df.sort_values("_mins__", ascending=False)
                .drop_duplicates(subset=subset, keep="first"))
    else:
        df = df.drop_duplicates()
    return df.drop(columns=["_mins__"], errors="ignore")

def _prep_p1_and_cands(p1_row: pd.Series, cand_df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Standardize and sign-fix NEGATIVE_FEATURES; return (p1_vec, candidates_matrix)."""
    p1_df = pd.DataFrame([p1_row[cols].to_dict()])
    both = pd.concat([p1_df, cand_df[cols].copy()], ignore_index=True)
    for c in cols:
        both[c] = pd.to_numeric(both[c], errors="coerce")
        if c in NEGATIVE_FEATURES:
            both[c] = -1.0 * both[c]
    both = both.astype(float)
    both = both.fillna(both.mean(numeric_only=True))
    std = both.std(ddof=0).replace(0, 1.0)
    both = (both - both.mean()) / std
    both = both.fillna(0.0)
    p1_vec = both.iloc[0].to_numpy(dtype=float)
    X_cand = both.iloc[1:].to_numpy(dtype=float)
    return p1_vec, X_cand

def _cosine_to_all(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return B_norm @ a_norm

# ---------- UI ----------
st.title("🧭 Find Similar Players")

# Small cache controls
with st.expander("⚙️ Session temp cache"):
    c1, c2 = st.columns(2)
    c1.write(session_files_list())
    if c2.button("🧹 Clear this session’s temp cache"):
        clear_this_session_cache()

# 0) competitions and valid season pairs (cached once per session)
if st.session_state["comps_cache"] is None:
    with st.status("Loading competitions...", expanded=False) as status:
        st.session_state["comps_cache"] = fetch_competitions(username, password)
        status.update(label="Competitions loaded", state="complete")

comps = st.session_state["comps_cache"]
if comps is None or comps.empty:
    st.warning("No competitions available.")
    st.stop()

if st.session_state["valid_player_pairs_cache"] is None:
    st.session_state["valid_player_pairs_cache"] = probe_player_pairs_progress(comps, username, password)

valid_pairs = st.session_state["valid_player_pairs_cache"]
if not valid_pairs:
    st.warning("No published player-stats yet.")
    st.stop()

comps["pair"] = list(zip(comps["competition_id"].astype(int), comps["season_id"].astype(int)))
comps_valid = comps[comps["pair"].isin(valid_pairs)].copy()

# --- Player 1 pool: League then season
st.header("Player 1 pool")
p1_league_options = sorted(comps_valid["league_label"].unique().tolist())
p1_league = st.selectbox("League", p1_league_options)

p1_block = comps_valid[comps_valid["league_label"] == p1_league].copy()
p1_block = p1_block.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)

p1_season_options = p1_block["season_norm"].tolist()
if not p1_season_options:
    st.warning("This league has no published seasons yet.")
    st.stop()

p1_season = st.selectbox("Season", p1_season_options, index=0)
row_p1 = p1_block[p1_block["season_norm"] == p1_season].iloc[0]
p1_comp_id = int(row_p1["competition_id"])
p1_season_id = int(row_p1["season_id"])

# --- Candidate pool: independent league and season
st.header("Candidate pool")
cand_league = st.selectbox("Candidate league", sorted(comps_valid["league_label"].unique().tolist()))
cand_block = comps_valid[comps_valid["league_label"] == cand_league].copy()
cand_block = cand_block.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)

cand_season_options = cand_block["season_norm"].tolist()
cand_season = st.selectbox("Candidate season", cand_season_options, index=0)

row_cand = cand_block[cand_block["season_norm"] == cand_season].iloc[0]
cand_comp_id = int(row_cand["competition_id"])
cand_season_id = int(row_cand["season_id"])

# --- Load the two pairs via memory/CSV/API (fast on revisit)
pairs_needed = [
    (p1_comp_id, p1_season_id, row_p1),
    (cand_comp_id, cand_season_id, row_cand),
]

frames = []
for cid, sid, meta_row in pairs_needed:
    meta = {
        "competition_name": meta_row.get("competition_name",""),
        "country_name": meta_row.get("country_name",""),
        "season_norm": meta_row.get("season_norm",""),
        "league_key": meta_row.get("league_key",""),
        "league_label": meta_row.get("league_label",""),
    }
    df_pair = get_players_pair_cached(cid, sid, meta, username, password)
    if not df_pair.empty:
        frames.append(df_pair)

players_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
players_all = _dedupe_players(players_all)  # <<< IMPORTANT: dedupe early
if players_all.empty:
    st.warning("No player data returned for the chosen league and season selections.")
    st.stop()
else:
    st.caption("Loaded from in-memory session cache / temp CSVs ✅")

# --- Clean and derive
num_cols = players_all.select_dtypes(include="number").columns
players_all[num_cols] = players_all[num_cols].fillna(0)
players_all = add_derived_player_metrics(players_all)
players_all = players_all[players_all["primary_position"] != "Goalkeeper"].copy()
players_all["Profile"] = players_all["primary_position"].map(position_mapping).fillna("Unknown")

# Age calculation from birth_date
players_all["birth_date"] = pd.to_datetime(players_all.get("birth_date"), errors="coerce")
today = pd.Timestamp("today").normalize()
players_all["Age"] = players_all["birth_date"].apply(
    lambda d: int((today - d).days // 365.25) if pd.notna(d) else np.nan
)

# --- Player 1 universe and minutes filtering
p1_pool = players_all[(players_all["competition_id"] == p1_comp_id) & (players_all["season_id"] == p1_season_id)].copy()
p1_pool = _dedupe_players(p1_pool)  # <<< dedupe P1 pool too
if p1_pool.empty:
    st.warning("No players found in the Player 1 league and season.")
    st.stop()

mins_series = pd.to_numeric(p1_pool.get("player_season_minutes"), errors="coerce").fillna(0)
lower_default = 0 if _season_sort_key(p1_season) >= _season_sort_key("2025-26") else 600
upper = int(max(mins_series.max(), lower_default))
min_low, min_high = st.slider(
    "Minutes Played range",
    min_value=int(lower_default),
    max_value=int(upper) if upper > int(lower_default) else int(lower_default) + 1,
    value=(int(lower_default), int(upper) if upper > int(lower_default) else int(lower_default) + 1),
    step=10,
)

# apply minutes filter to both universes
p1_pool = p1_pool[
    (pd.to_numeric(p1_pool["player_season_minutes"], errors="coerce").fillna(0) >= min_low) &
    (pd.to_numeric(p1_pool["player_season_minutes"], errors="coerce").fillna(0) <= min_high)
].copy()
players_all = players_all[
    (pd.to_numeric(players_all["player_season_minutes"], errors="coerce").fillna(0) >= min_low) &
    (pd.to_numeric(players_all["player_season_minutes"], errors="coerce").fillna(0) <= min_high)
].copy()

# for the most current type seasons, require minutes > 0
if _season_sort_key(p1_season) >= _season_sort_key("2025-26"):
    p1_pool = p1_pool[pd.to_numeric(p1_pool["player_season_minutes"], errors="coerce").fillna(0) > 0]
    players_all = players_all[pd.to_numeric(players_all["player_season_minutes"], errors="coerce").fillna(0) > 0]

if p1_pool.empty:
    st.warning("No players match the selected minutes range for Player 1 league and season.")
    st.stop()

# Optional profile pre-filter for P1 list
profile_choices = ["All"] + sorted(p1_pool["Profile"].dropna().unique().tolist())
p1_profile_filter = st.selectbox("Filter profile for Player 1", profile_choices, index=0)
p1_candidates = p1_pool if p1_profile_filter == "All" else p1_pool[p1_pool["Profile"] == p1_profile_filter]

p1_name = st.selectbox("Select Player 1", sorted(p1_candidates["player_name"].unique()), key="player1_sel")
p1_prof_series = players_all.loc[players_all["player_name"] == p1_name, "Profile"]
p1_profile = p1_prof_series.iloc[0] if not p1_prof_series.empty else "Unknown"

# --- Candidate pool for same profile and selected league season
candidates = players_all[
    (players_all["Profile"] == p1_profile) &
    (players_all["competition_id"] == cand_comp_id) &
    (players_all["season_id"] == cand_season_id)
].copy()
candidates = _dedupe_players(candidates)  # <<< dedupe candidate pool

# Age slider for candidates if available
age_series = pd.to_numeric(candidates.get("Age"), errors="coerce").dropna()
if not age_series.empty:
    a_min, a_max = int(age_series.min()), int(age_series.max())
    age_low, age_high = st.slider("Candidate age range", min_value=a_min, max_value=a_max, value=(a_min, a_max), step=1)
    candidates = candidates[
        pd.to_numeric(candidates["Age"], errors="coerce").between(age_low, age_high, inclusive="both")
    ].copy()

if candidates.empty:
    st.warning("No candidate players after filters.")
    st.stop()

# --- Metric selection
label_to_key = {v: k for k, v in metrics_mapping.items()}
key_to_label = metrics_mapping.copy()
default_keys = get_metrics_for_profile(p1_profile) or []

st.markdown("#### Choose metrics for similarity")
use_profile_defaults = st.checkbox("Use profile default metrics", value=True)

if use_profile_defaults:
    chosen_keys = default_keys
else:
    chosen_labels = st.multiselect(
        "Pick one or more metrics",
        options=sorted(label_to_key.keys()),
        default=[],
    )
    chosen_keys = [label_to_key.get(lbl, lbl) for lbl in chosen_labels if label_to_key.get(lbl, None)]

chosen_keys = [k for k in (chosen_keys or default_keys) if k in candidates.columns]
if not chosen_keys:
    st.error("None of the selected metrics exist in the candidate season.")
    st.stop()

# --- Build similarity with status indicator
with st.status("Computing similarity...", expanded=False) as status:
    p1_full = players_all[
        (players_all["player_name"] == p1_name) &
        (players_all["competition_id"] == p1_comp_id) &
        (players_all["season_id"] == p1_season_id)
    ].copy()
    p1_full = _dedupe_players(p1_full)

    if p1_full.empty:
        st.error("Could not find Player 1 row after filters.")
        st.stop()

    if pd.to_numeric(p1_full["player_season_minutes"], errors="coerce").fillna(0).max() <= 0:
        st.error("Player 1 has 0 minutes. Widen the minutes range.")
        st.stop()

    # exclude Player 1 if he is in candidates
    if "player_id" in candidates.columns and "player_id" in p1_full.columns:
        p1_ids = set(pd.to_numeric(p1_full["player_id"], errors="coerce").dropna().astype(int).tolist())
        candidates = candidates[~pd.to_numeric(candidates["player_id"], errors="coerce").isin(p1_ids)]
    else:
        candidates = candidates[candidates["player_name"] != p1_name]

    if candidates.empty:
        st.warning("No candidate players once Player 1 is excluded.")
        st.stop()

    p1_row_full = p1_full.iloc[0]
    p1_vec, X_cand = _prep_p1_and_cands(p1_row_full, candidates, chosen_keys)
    sims = _cosine_to_all(p1_vec, X_cand)
    status.update(label="Similarity computed", state="complete")

# --- Results table
# keep player_id just for dedupe; drop before display
cand_display_cols = ["player_name", "team_name", "competition_name", "season_norm", "player_season_minutes", "Profile", "Age"]
if "player_id" in candidates.columns:
    cand_display_cols = ["player_id"] + cand_display_cols

res = candidates[cand_display_cols + chosen_keys].copy()
res["similarity"] = sims

# sort then dedupe result rows (extra guard)
res = res.sort_values("similarity", ascending=False)
if "player_id" in res.columns:
    res = res.drop_duplicates(subset=["player_id"], keep="first")
else:
    res = res.drop_duplicates(subset=["player_name", "competition_name", "season_norm"], keep="first")

top_n = st.slider("How many similar players?", min_value=5, max_value=50, value=15, step=1)
res_top = res.head(top_n).reset_index(drop=True)

# drop player_id from display, if present
if "player_id" in res_top.columns:
    res_top = res_top.drop(columns=["player_id"], errors="ignore")

rename_cols = {
    "player_name": "Player",
    "team_name": "Team",
    "competition_name": "League",
    "season_norm": "Season",
    "player_season_minutes": "Minutes",
    "Profile": "Profile",
    "Age": "Age",
    "similarity": "Similarity",
}
for k in chosen_keys:
    rename_cols[k] = metrics_mapping.get(k, k)

res_top = res_top.rename(columns=rename_cols)

def sim_bar(p: float) -> str:
    pct = float(max(0.0, min(1.0, p))) * 100.0
    cls = "good" if pct >= 80 else ("mid" if pct >= 60 else "bad")
    return f"""
    <div class="simbar-wrap"><div class="simbar-fill {cls}" style="width:{pct:.0f}%"></div></div>
    <div class="simval">{pct:.1f}%</div>
    """

res_top["Minutes"] = pd.to_numeric(res_top["Minutes"], errors="coerce").round(0).astype("Int64")
res_top["Similarity"] = res_top["Similarity"].map(sim_bar)

st.markdown("### 🔎 Similar Players")
st.caption(
    f"P1: {p1_name} ({p1_profile}) · {p1_league} {p1_season}  ->  "
    f"Candidates: {cand_league} {cand_season} · Metrics ({len(chosen_keys)}): "
    + ", ".join([metrics_mapping.get(k, k) for k in chosen_keys])
)

def render_results_table(df: pd.DataFrame):
    base_cols = ["Player", "Team", "League", "Season", "Minutes", "Age", "Similarity"]
    metric_cols = [c for c in df.columns if c not in base_cols]
    ordered = base_cols + metric_cols
    html = [
        "<table class='comparison-table'>",
        "<thead><tr>" + "".join([f"<th>{c}</th>" for c in ordered]) + "</tr></thead><tbody>"
    ]
    for _, r in df.iterrows():
        html.append("<tr>")
        for c in ordered:
            val = r[c]
            if c == "Similarity":
                html.append(f"<td style='min-width:140px'>{val}</td>")
            else:
                if isinstance(val, (int, float, np.floating)) and not pd.isna(val):
                    if c in ("Age", "Minutes"):
                        html.append(f"<td>{int(val)}</td>")
                    else:
                        html.append(f"<td>{val:,.2f}</td>")
                else:
                    html.append(f"<td>{'' if pd.isna(val) else val}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)

render_results_table(res_top)

with st.expander("Show raw chosen features for Player 1"):
    p1_vals = players_all[
        (players_all["player_name"] == p1_name) &
        (players_all["competition_id"] == p1_comp_id) &
        (players_all["season_id"] == p1_season_id)
    ][chosen_keys].iloc[0].rename(index=lambda k: metrics_mapping.get(k, k))
    st.dataframe(p1_vals.to_frame("Value"))
