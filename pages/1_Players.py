# pages/1_Players.py
import os
import pathlib
import uuid
import shutil
import time
from datetime import date
from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import pandas as pd
import streamlit as st
from utils import get_statsbomb_creds
from Config.league_weights import league_weight
from Config.config_metrics import (
    position_mapping,
    metrics_mapping,
    get_metrics_for_profile,
    add_derived_player_metrics,
    DETAILED_PROFILE_METRICS,
)
from Benchmarks.players_benchmark import compare_players_profiled, NEGATIVE_FEATURES,_pct_rank

st.set_page_config(layout="wide", page_title="Players")

# ---- CSS ----
def load_css(file_path: pathlib.Path):
    if file_path.exists():
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<style>
.comparison-table{width:100%;border-collapse:collapse;margin:8px 0}
.comparison-table th,.comparison-table td{border:1px solid #e5e7eb;padding:8px;text-align:left}
.cell-better{background:#c6f6d5;font-weight:600}
.cell-worse{background:#fecaca;font-weight:600}
.cell-equal{background:#f7fafc;font-weight:600}
.pbar-wrap{height:10px;background:#eee;border-radius:6px;overflow:hidden}
.pbar-fill{height:100%;width:0%}
.pbar-fill.good{background:#22c55e}
.pbar-fill.mid{background:#f59e0b}
.pbar-fill.bad{background:#ef4444}
.pbar-val{font-weight:600;font-variant-numeric:tabular-nums}
</style>
""", unsafe_allow_html=True)
load_css(pathlib.Path("assets/styles.css"))

# =========================
# In-session, cross-page memory caches
# =========================
if "comps_cache" not in st.session_state:
    st.session_state["comps_cache"] = None           # competitions df
if "valid_pairs_cache" not in st.session_state:
    st.session_state["valid_pairs_cache"] = None     # set of (comp_id, season_id)
if "players_cache" not in st.session_state:
    st.session_state["players_cache"] = {}           # {(comp_id, season_id): DataFrame}

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
        # also clear in-memory caches
        st.session_state["players_cache"].clear()
        st.session_state["comps_cache"] = None
        st.session_state["valid_pairs_cache"] = None
        st.success("Cleared this session’s temp cache.")
    except Exception as e:
        st.error(f"Failed to clear session cache: {e}")

username , password = get_statsbomb_creds()
if not username or not password:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()

# ---- Helpers ----
def _norm_season(name: str) -> str:
    """Convert '2024/2025' -> '2024-25' style labels."""
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
    df["league_key"] = df["country_name"].astype(str) + " | " + df["competition_name"].astype(str)
    df["league_label"] = df["country_name"].astype(str) + " - " + df["competition_name"].astype(str)
    return df

def probe_player_pairs_progress(season_df: pd.DataFrame, user: str, pwd: str) -> set[tuple]:
    """Return set of (competition_id, season_id) pairs that actually have v4 player-stats."""
    valids = []
    with st.status("Probing leagues for published player data...", expanded=True) as stat:
        total = len(season_df)
        prog = st.progress(0.0)
        for i, (_, r) in enumerate(season_df.iterrows(), start=1):
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
            stat.update(label=f"Checked {i}/{total} season pairs")
            prog.progress(i / max(1, total))
        stat.update(label="Finished probing seasons", state="complete")
    return set(valids)

def fetch_players_for_pairs_progress(pairs_df: pd.DataFrame, user, pwd) -> pd.DataFrame:
    """Fetch player-stats for the exact (competition_id, season_id) rows supplied."""
    frames = []
    with st.status("Fetching player stats...", expanded=True) as stat:
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
                    except Exception:
                        rows = None
                    if rows:
                        df = pd.DataFrame(rows)
                        if not df.empty:
                            df["competition_id"] = cid
                            df["season_id"] = sid
                            df["competition_name"] = comp.get("competition_name","")
                            df["country_name"] = comp.get("country_name","")
                            df["season_norm"] = comp.get("season_norm","")
                            df["league_key"] = comp.get("league_key","")
                            df["league_label"] = comp.get("league_label","")
                            frames.append(df)
            stat.update(label=f"Fetched {i}/{total} season blocks")
            prog.progress(i / max(1, total))
        stat.update(label="Finished fetching player stats", state="complete")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _season_sort_key(s: str) -> tuple:
    """Sort season_norm like '2025-26' chronologically."""
    if not isinstance(s, str) or "-" not in s:
        return (0, 0)
    y1, y2 = s.split("-", 1)
    try:
        return (int(y1), int(y2))
    except Exception:
        return (0, 0)

# ---------- unified pair getter (in-memory -> CSV -> API) ----------
def get_players_pair_cached(cid: int, sid: int, meta: dict, user: str, pwd: str) -> pd.DataFrame:
    key = (int(cid), int(sid))

    # 1) in-memory (survives page changes)
    if key in st.session_state["players_cache"]:
        return st.session_state["players_cache"][key].copy()

    # 2) session CSV
    cached_csv = load_session_csv("players", cid, sid)
    if cached_csv is not None and not cached_csv.empty:
        df = cached_csv.copy()
        for k in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]:
            df[k] = meta.get(k, "")
        df["competition_id"] = cid
        df["season_id"] = sid
        st.session_state["players_cache"][key] = df.copy()
        return df

    # 3) API fetch (single row frame), then persist CSV and memory
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
    # save per-pair CSV without meta columns
    save_session_csv(
        df_pair.drop(columns=[c for c in ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"] if c in df_pair.columns]),
        "players", cid, sid
    )
    st.session_state["players_cache"][key] = df_pair.copy()
    return df_pair

# ========== UI ==========
st.title("🧑‍🤝‍🧑 Players")

# 0) Small cache controls
with st.expander("⚙️ Session temp cache"):
    c1, c2 = st.columns(2)
    c1.write(session_files_list())
    if c2.button("🧹 Clear this session’s temp cache"):
        clear_this_session_cache()

# 1) Competitions (load once per session)
if st.session_state["comps_cache"] is None:
    with st.status("Loading competitions...", expanded=False) as s0:
        st.session_state["comps_cache"] = fetch_competitions(username, password)
        s0.update(label="Competitions loaded", state="complete")

comps = st.session_state["comps_cache"]
if comps is None or comps.empty:
    st.warning("No competitions available.")
    st.stop()

# 1b) Probe valid pairs (once per session)
if st.session_state["valid_pairs_cache"] is None:
    st.session_state["valid_pairs_cache"] = probe_player_pairs_progress(comps, username, password)

valid_pairs = st.session_state["valid_pairs_cache"]
if not valid_pairs:
    st.warning("No leagues with published player data found.")
    st.stop()

comps["pair"] = list(zip(comps["competition_id"].astype(int), comps["season_id"].astype(int)))
comps_valid = comps[comps["pair"].isin(valid_pairs)].copy()

# 2) Pick the league first
league_options = sorted(comps_valid["league_label"].unique().tolist())
sel_league = st.selectbox("League", league_options)

# All seasons (with data) for this league
league_block = comps_valid[comps_valid["league_label"] == sel_league].copy()
league_block = league_block.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)

season_options = league_block["season_norm"].tolist()
default_season = season_options[0] if season_options else None
sel_season = st.selectbox("Season", season_options, index=0 if default_season else None)

if sel_season not in set(league_block["season_norm"]):
    st.info("Selected season has no player data; using nearest season with data.")
    sel_season = default_season

# P1 universe
rowA = league_block[league_block["season_norm"] == sel_season].iloc[0]
league_comp_id = int(rowA["competition_id"])
league_season_id = int(rowA["season_id"])

# 3) Optional different league-season for Player 2 pool
st.markdown("##### Optional: choose a different league and season for Player 2")
c1, c2 = st.columns(2)
with c1:
    sel_league_p2 = st.selectbox("Player 2 League", ["Same as above"] + league_options, index=0)
with c2:
    if sel_league_p2 == "Same as above":
        p2_seasons = season_options
    else:
        lb = comps_valid[comps_valid["league_label"] == sel_league_p2].copy()
        lb = lb.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)
        p2_seasons = lb["season_norm"].tolist()
    sel_season_p2 = st.selectbox("Player 2 Season", p2_seasons, index=0)

if sel_league_p2 == "Same as above":
    rowB = rowA
else:
    league_block_p2 = comps_valid[comps_valid["league_label"] == sel_league_p2].copy()
    league_block_p2 = league_block_p2.sort_values("season_norm", key=lambda s: s.map(_season_sort_key), ascending=False)
    if sel_season_p2 not in set(league_block_p2["season_norm"]):
        st.info("Player 2 season has no player data; using nearest season with data.")
        sel_season_p2 = league_block_p2["season_norm"].iloc[0]
    rowB = league_block_p2[league_block_p2["season_norm"] == sel_season_p2].iloc[0]

p2_comp_id = int(rowB["competition_id"])
p2_season_id = int(rowB["season_id"])

# 4) Pull players only for needed pairs or all with toggle
mode = st.radio("Load scope", ["Only selected leagues", "All leagues with data (season-wide)"], horizontal=True, index=0)
if mode == "Only selected leagues":
    need = pd.concat([rowA.to_frame().T, rowB.to_frame().T], ignore_index=True)[
        ["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]
    ].drop_duplicates()
else:
    need = comps_valid[["competition_id","season_id","competition_name","country_name","season_norm","league_key","league_label"]].drop_duplicates()

# === Truly sticky players data: memory -> CSV -> API ===
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
    df_pair = get_players_pair_cached(cid, sid, meta, username, password)
    if not df_pair.empty:
        frames.append(df_pair)

players_all_raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if players_all_raw.empty:
    st.warning("No player data returned.")
    st.stop()
else:
    st.caption("Loaded from in-memory session cache / temp CSVs ✅")

# 5) Clean + derived + profiles
players_all = players_all_raw.copy()
num_cols = players_all.select_dtypes(include="number").columns
players_all[num_cols] = players_all[num_cols].fillna(0)
players_all = add_derived_player_metrics(players_all)
players_all = players_all[players_all["primary_position"] != "Goalkeeper"].copy()
players_all["Profile"] = players_all["primary_position"].map(position_mapping).fillna("Unknown")

# 6) Player 1 universe
players_leagueA = players_all[
    (players_all["competition_id"] == league_comp_id) & (players_all["season_id"] == league_season_id)
].copy()

# --- NEW: Team filter (Player 1 universe) ---
team_choices = ["All teams"] + sorted(players_leagueA["team_name"].dropna().unique().tolist())
sel_team = st.selectbox(
    "Team (Player 1 universe)",
    team_choices,
    index=0,
    help="Filter the Player 1 list by team to jump straight to a player."
)

if sel_team != "All teams":
    players_leagueA = players_leagueA[players_leagueA["team_name"] == sel_team].copy()

if players_leagueA.empty:
    st.warning("No players left after applying the team filter. Try another team or widen filters.")
    st.stop()



# minutes slider
mins_series = pd.to_numeric(players_leagueA.get("player_season_minutes", pd.Series(dtype=float)), errors="coerce").fillna(0)
def _tuple_from_label(lbl: str) -> tuple:
    if not isinstance(lbl, str) or "-" not in lbl: return (0, 0)
    a, b = lbl.split("-", 1)
    try: return (int(a), int(b))
    except: return (0, 0)
lower_default = 0 if _tuple_from_label(sel_season) >= _tuple_from_label("2025-26") else 600
upper = int(max(mins_series.max(), lower_default))
min_cut = st.slider("Minimum Minutes (Player 1 + Player 2 universes)", min_value=int(lower_default), max_value=int(upper),
                    value=int(lower_default), step=10)

players_leagueA = players_leagueA[pd.to_numeric(players_leagueA["player_season_minutes"], errors="coerce").fillna(0) >= min_cut]
players_all = players_all[pd.to_numeric(players_all["player_season_minutes"], errors="coerce").fillna(0) >= min_cut]

if players_leagueA.empty:
    st.warning("No players meet the minutes threshold for Player 1 league and season.")
    st.stop()

# 7) P1 selection with optional profile pre-filter
profile_choices = ["All"] + sorted(players_leagueA["Profile"].dropna().unique().tolist())
p1_profile_filter = st.selectbox("Filter profile for Player 1", profile_choices, index=0)
p1_pool = players_leagueA if p1_profile_filter == "All" else players_leagueA[players_leagueA["Profile"] == p1_profile_filter]

p1 = st.selectbox("Select Player 1", sorted(p1_pool["player_name"].unique()), key="player1")
p1_profile = players_all.loc[players_all["player_name"] == p1, "Profile"].iloc[0]

# 8) Player 2 pool
p2_pool = players_all[
    (players_all["Profile"] == p1_profile) &
    (players_all["competition_id"] == p2_comp_id) &
    (players_all["season_id"] == p2_season_id)
].copy()
# --- NEW: Team filter (Player 2 universe; optional) ---
p2_team_choices = ["All teams"] + sorted(p2_pool["team_name"].dropna().unique().tolist())
sel_team_p2 = st.selectbox(
    "Team (Player 2 universe)",
    p2_team_choices,
    index=0,
    help="Filter the Player 2 list by team (optional)."
)
if sel_team_p2 != "All teams":
    p2_pool = p2_pool[p2_pool["team_name"] == sel_team_p2].copy()

if p2_pool.empty:
    st.warning("No Player 2 candidates left after applying the team filter.")
    st.stop()

all_names_p2 = [""] + sorted([n for n in p2_pool["player_name"].unique() if n != p1])
p2 = st.selectbox("Select Player 2 (or blank for average)", all_names_p2, key="player2")

# 9) Metric lens selection
lens_options = ["Use profile defaults"] + list(DETAILED_PROFILE_METRICS.keys())
metric_lens = st.selectbox("Metric lens", lens_options, index=0)

# 10) Extra metrics on top of the chosen lens (optional)
label_to_key = {v: k for k, v in metrics_mapping.items()}
key_to_label = metrics_mapping.copy()

if metric_lens == "Use profile defaults":
    base_keys = get_metrics_for_profile(p1_profile) or []
else:
    base_keys = DETAILED_PROFILE_METRICS.get(metric_lens, [])

existing_metric_keys = [k for k in key_to_label if k in players_leagueA.columns]
extra_candidate_labels = sorted([key_to_label[k] for k in existing_metric_keys if k not in set(base_keys)])

st.markdown("#### Add extra metrics (applies to both views)")
extra_labels = st.multiselect("Type to add metrics", options=extra_candidate_labels, default=[])
extra_keys = [label_to_key[lbl] for lbl in extra_labels if lbl in label_to_key]

chosen_metric_keys_base = []
_seen = set()
for k in list(base_keys) + list(extra_keys):
    if k in players_leagueA.columns and k not in _seen:
        chosen_metric_keys_base.append(k); _seen.add(k)

# 11) Build compare table (FAIR COHORT = Player 1's profile)
df_for_compare = players_leagueA if not p2 else players_all
comp_df, _, bio1, bio2 = compare_players_profiled(
    df_for_compare,
    p1,
    p2 or None,
    avg_name="Average Player",
    decimals=2,
    metrics_mapping=metrics_mapping,
    get_metrics_for_profile=get_metrics_for_profile,
    percentile_scope="profile_league",
    cohort_profile=p1_profile,
    league_weight_func=league_weight,
    include_metrics=chosen_metric_keys_base,
    metrics_group_name=(None if metric_lens == "Use profile defaults" else metric_lens),
    detailed_profile_metrics=DETAILED_PROFILE_METRICS
)

# ---- Bios tidy ----
def _tidy_bio(bio: pd.DataFrame) -> pd.DataFrame:
    out = bio.copy()
    if "birth_date" in out.columns:
        today = date.today()
        def calc_age(val):
            bd = pd.to_datetime(val, errors="coerce")
            if pd.isna(bd): return pd.NA
            bd = bd.date()
            return (today.year - bd.year) - int((today.month, today.day) < (bd.month, bd.day))
        out["Age"] = out["birth_date"].apply(calc_age).astype("Int64")
        out = out.drop(columns=["birth_date"])
    if "player_season_minutes" in out.columns:
        out["Minutes"] = pd.to_numeric(out["player_season_minutes"], errors="coerce").round(0).astype("Int64")
        out = out.drop(columns=["player_season_minutes"])
    desired = [c for c in ["player_name", "team_name", "Profile", "Age", "Minutes"] if c in out.columns]
    rest = [c for c in out.columns if c not in desired]
    return out[desired + rest]

left, right = st.columns(2)
with left:
    st.markdown(f"### 🎯 {p1} — {p1_profile or 'Unknown'}")
    st.dataframe(_tidy_bio(bio1).set_index("player_name"))

if p2:
    with right:
        p2_prof_series = players_all.loc[players_all["player_name"] == p2, "Profile"]
        p2_profile = p2_prof_series.iloc[0] if not p2_prof_series.empty else "Unknown"
        st.markdown(f"### 👤 {p2} — {p2_profile}")
        st.dataframe(_tidy_bio(bio2).set_index("player_name"))
else:
    st.caption(f"Comparing to league average ({sel_league} — {sel_season})")

# ---- Percentile bars ----
def render_percentile_bars(df: pd.DataFrame, player: str, cohort_profile: str, metric_keys: list[str]):
    id_col = "player_name"
    sub = df[df["Profile"] == cohort_profile].copy()
    if sub.empty or not metric_keys:
        st.info("No percentile data available for this profile.")
        return

    for m in metric_keys:
        if m not in sub.columns:
            continue

        series = pd.to_numeric(sub[m], errors="coerce")
        hib = (m not in NEGATIVE_FEATURES)  # higher-is-better?
        pct = _pct_rank(series, higher_is_better=hib).fillna(0.0)

        # pick the first row for the player if duplicated
        idx = sub.index[sub["player_name"] == player]
        if len(idx) == 0:
            continue
        val = float(pct.loc[idx[0]])  # 0..1, 1 = best

        color = "good" if val >= 0.66 else ("mid" if val >= 0.33 else "bad")
        label = metrics_mapping.get(m, m)

        # Optional: nicer ordinal suffix
        pctl = int(round(val * 100))
        suffix = "th" if 10 <= (pctl % 100) <= 20 else {1:"st",2:"nd",3:"rd"}.get(pctl % 10, "th")

        st.markdown(
            f"""
            <div style="margin-bottom:6px;">
              <div style="font-size:13px;font-weight:600;">
                {label}: <span class='pbar-val'>{pctl}{suffix} %</span>
              </div>
              <div class='pbar-wrap'><div class='pbar-fill {color}' style='width:{val*100:.1f}%;'></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- Comparison renderer ----
label_to_key = {v: k for k, v in metrics_mapping.items()}

def _parse_float(x):
    if isinstance(x, (int, float)): return float(x)
    if not isinstance(x, str) or not x: return np.nan
    try: return float(x.replace(",", ""))
    except Exception: return np.nan

def _is_better(metric_key: str, a: float, b: float) -> int:
    if np.isnan(a) or np.isnan(b): return 0
    if metric_key in NEGATIVE_FEATURES:
        return 1 if a < b else (-1 if a > b else 0)
    return 1 if a > b else (-1 if a < b else 0)

def render_comparison_table(col_df: pd.DataFrame, col1: str, col2: str, metric_keys: list[str]):
    chosen_labels = set(metrics_mapping.get(k, k) for k in metric_keys)
    df_show = col_df[col_df["Metric"].isin(chosen_labels)].copy()
    if df_show.empty:
        st.info("No metrics selected or no data available.")
        return
    html = [
        "<table class='comparison-table'>",
        "<thead>",
        f"<tr><th>Metric</th><th>{col1}</th><th>{col2}</th></tr>",
        "</thead><tbody>",
    ]
    for _, row in df_show.iterrows():
        label = row["Metric"]
        key = label_to_key.get(label, None)
        v1 = _parse_float(row[col1]); v2 = _parse_float(row[col2])
        cmp = _is_better(key or "", v1, v2)
        cls1, cls2 = ("cell-better", "cell-worse") if cmp > 0 else \
                     ("cell-worse", "cell-better") if cmp < 0 else \
                     ("cell-equal", "cell-equal")
        html.append("<tr>")
        html.append(f"<td>{label}</td>")
        html.append(f"<td class='{cls1}'>{row[col1]}</td>")
        html.append(f"<td class='{cls2}'>{row[col2]}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)

# ---- View switch ----
view_mode = st.radio("View mode", ["Percentiles", "Comparison"], index=0, horizontal=True)
if view_mode == "Percentiles":
    st.markdown(f"### 📊 {p1} — Percentile ranks in {sel_league} ({sel_season})")
    render_percentile_bars(players_leagueA, p1, p1_profile, chosen_metric_keys_base)
else:
    st.markdown(f"### 📈 Profiled Comparison (metrics: {len(chosen_metric_keys_base)}) — Lens: {metric_lens}")
    render_comparison_table(comp_df, p1, p2 if p2 else "Average Player", chosen_metric_keys_base)
