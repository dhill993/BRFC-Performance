import os
from pathlib import Path
import re
import glob
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from utils import get_statsbomb_creds
from utils import (
    SERVICE,
    find_local_drive_root,                    # cached Drive client if available
    get_drive_service,          # builds a Drive client from secrets
    list_files_from_drive,      # Drive search helper
    download_file,              # Drive download helper
    drive_resolve_path          # resolve nested folder ids
)


st.set_page_config(layout="wide", page_title="Set Pieces - Players (Latest Season)")

# Drive client (service account on cloud, user OAuth locally)
svc = SERVICE or get_drive_service()

# --- Resolve working root ---
# Local:  G:/My Drive/Bristol Rovers FC/Set Piece Stats
# Cloud:  data/drive  (download cache)
my_drive = find_local_drive_root()
DRIVE_ROOT = Path("data/drive")   # default cache
local_candidate = None
if my_drive:
    maybe = my_drive / "Bristol Rovers FC" / "Set Piece Stats"
    if maybe.exists():
        local_candidate = maybe
        DRIVE_ROOT = maybe

# IMPORTANT: decide mode only from whether we found the real local path
IS_LOCAL = local_candidate is not None
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

# --- Sidebar (single block, unique keys) ---
with st.sidebar:
    st.markdown("### Data source")
    if IS_LOCAL:
        drive_input = st.text_input(
            "Drive root",
            value=str(DRIVE_ROOT),
            key="psp_drive_root",
        )
        DRIVE_ROOT = Path(drive_input)

        # Default TRUE avoids surprise filtering
        st.checkbox(
            "Show all leagues (ignore local folders)",
            value=True,
            key="psp_show_all",
        )
    else:
        st.write(f"Cloud cache: {DRIVE_ROOT}")


# Candidate mapping files
LATEST_CSV_CANDIDATES = [
    DRIVE_ROOT / "df_competitions_last_season.csv",
    DRIVE_ROOT / "df_competitions_last_season.tsv",
    DRIVE_ROOT / "competitions_latest.csv",
    DRIVE_ROOT / "competitions_latest.tsv",
]
# keep parent variants for your existing local layout if needed
if DRIVE_ROOT.parent != DRIVE_ROOT:
    LATEST_CSV_CANDIDATES += [
        DRIVE_ROOT.parent / "df_competitions_last_season.csv",
        DRIVE_ROOT.parent / "df_competitions_last_season.tsv",
        DRIVE_ROOT.parent / "competitions_latest.csv",
        DRIVE_ROOT.parent / "competitions_latest.tsv",
    ]

HOPS_GLOB = [
    "assets/HOPS_results_*.csv", "./assets/HOPS_results_*.csv",
    str(Path.home() / "Desktop/Bristol Rovers Benchmark App/assets/HOPS_results_*.csv")
]

SP_METRICS = [
    "xG_from_SP", "xGCreated_from_SP", "SP_shots", "SP_goals", "SP_key_passes", "SP_assists_from_SP",
    "SP_first_contacts_won_off", "SP_first_contacts_won_def",
    "SP_aerials_won_off", "SP_aerials_won_def", "SP_clearances", "SP_blocks",
]

SP_SUFFIX_BY_LABEL = {"Corner": "Corner", "Free Kick": "FreeKick", "Throw-in": "ThrowIn"}

BIO_COLS = [
    "player.id", "player.name", "team.name", "primary_position",
    "player_age", "player_season_minutes",
    "competition_id", "season_id", "league_label", "season_name", "season_norm"
]

FRIENDLY = {
    "player.id": "Player ID", "player_id": "Player ID", "player.name": "Player", "team.name": "Team",
    "primary_position": "Primary Position", "player_age": "Age", "player_season_minutes": "Minutes", "HOPS": "HOPS",
    "xG_from_SP": "xG from SP", "xGCreated_from_SP": "xG Created", "SP_shots": "SP Shots", "SP_goals": "SP Goals",
    "SP_key_passes": "SP Key Passes", "SP_assists_from_SP": "SP Assists", "SP_first_contacts_won_off": "First Contacts Won (Off)",
    "SP_first_contacts_won_def": "First Contacts Won (Def)", "SP_aerials_won_off": "Aerial Duels Won (Off)",
    "SP_aerials_won_def": "Aerial Duels Won (Def)", "SP_clearances": "Clearances", "SP_blocks": "Blocks",
}

def _read_latest_mapping() -> tuple[pd.DataFrame, Path | None]:
    # local first
    for p in LATEST_CSV_CANDIDATES:
        if p.exists():
            df = _read_delim(p)
            if not df.empty:
                return df, p

    # cloud fallback
    if not svc:
        return pd.DataFrame(), None

    top_id = st.secrets.get("drive", {}).get("folder_id")
    search_parents = [top_id] if top_id else [None]

    needles = ["df_competitions_last_season", "competitions_latest"]
    found = []
    for parent in search_parents:
        for n in needles:
            found.extend(list_files_from_drive(svc, name_contains=n, parent_id=parent))

    if not found:
        return pd.DataFrame(), None

    found.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    f = found[0]
    out_path = DRIVE_ROOT / f["name"]
    if not out_path.exists():
        download_file(svc, f["id"], out_path)

    df = _read_delim(out_path)
    return (df, out_path) if not df.empty else (pd.DataFrame(), None)

def _latest_file(patterns: list[str]) -> str | None:
    """Return path to most recent file across several glob patterns (or None)."""
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _normalize_sp_type(s: str) -> str:
    s = str(s or "").strip()
    mapping = {
        "Corner": "Corner", "corner": "Corner",
        "Free Kick": "Free Kick", "FreeKick": "Free Kick",
        "Throw-in": "Throw-in", "ThrowIn": "Throw-in", "Throw In": "Throw-in",
    }
    return mapping.get(s, s if s else "Unknown")


def _build_totals_from_breakdown(bdf: pd.DataFrame) -> pd.DataFrame:
    if bdf.empty:
        return pd.DataFrame()

    schema = _detect_schema(bdf)

    if schema == "LONG":
        # only use metrics that actually exist
        metrics = [m for m in SP_METRICS if m in bdf.columns]
        if not metrics:
            return pd.DataFrame()
        # make sure they are numeric
        bnum = bdf.copy()
        for m in metrics:
            bnum[m] = pd.to_numeric(bnum[m], errors="coerce")
        g = bnum.groupby(
            [c for c in ["player.id", "player.name", "team.name"] if c in bnum.columns],
            as_index=False
        )[metrics].sum(min_count=1)

        # carry over single-valued metadata if present
        for c in ["competition_id", "season_id", "league_label", "season_name", "season_norm"]:
            if c in bdf.columns and c not in g.columns and bdf[c].notna().any():
                g[c] = bdf[c].dropna().iloc[0]
        return g

    if schema == "WIDE":
        id_cols = [c for c in ["player.id", "player.name", "team.name"] if c in bdf.columns]
        if not id_cols:
            # if ids are missing there is nothing to align on
            return pd.DataFrame()
        base = bdf[id_cols].copy()

        for m in SP_METRICS:
            total_col = f"{m}_Total"
            if total_col in bdf.columns:
                base[m] = pd.to_numeric(bdf[total_col], errors="coerce")
            else:
                parts = [f"{m}_{suf}" for suf in SP_SUFFIX_BY_LABEL.values() if f"{m}_{suf}" in bdf.columns]
                if parts:
                    # sum parts after numeric conversion
                    tmp = bdf[parts].apply(pd.to_numeric, errors="coerce")
                    base[m] = tmp.sum(axis=1, min_count=1)
                else:
                    base[m] = 0

        # carry over single-valued metadata if present
        for c in ["competition_id", "season_id", "league_label", "season_name", "season_norm"]:
            if c in bdf.columns and c not in base.columns and bdf[c].notna().any():
                base[c] = bdf[c].dropna().iloc[0]
        return base

    return pd.DataFrame()


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def _slice_wide_breakdown(bdf: pd.DataFrame, sp_label: str) -> pd.DataFrame:
    suf = SP_SUFFIX_BY_LABEL.get(sp_label, None)
    if suf is None:
        return pd.DataFrame()
    keep = [c for c in BIO_COLS if c in bdf.columns]
    metric_cols = [f"{m}_{suf}" for m in SP_METRICS if f"{m}_{suf}" in bdf.columns]
    if not metric_cols:
        return pd.DataFrame()
    id_cols = [c for c in ["player.id", "player.name", "team.name"] if c in bdf.columns]
    cols = (keep + metric_cols) if keep else (id_cols + metric_cols)
    # guard if even id_cols are missing
    cols = [c for c in cols if c in bdf.columns]
    return bdf[cols].copy()


def _pretty_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [FRIENDLY.get(c, c) for c in df.columns]
    return out

def _with_hops(df: pd.DataFrame, hops: pd.DataFrame) -> pd.DataFrame:
    if df.empty or hops.empty:
        return df

    # Build a safe key without mutating inputs
    def pid_col(frame: pd.DataFrame) -> pd.Series | None:
        if "player.id" in frame.columns:
            return pd.to_numeric(frame["player.id"], errors="coerce")
        if "player_id" in frame.columns:
            return pd.to_numeric(frame["player_id"], errors="coerce")
        return None

    left_pid = pid_col(df)
    right_pid = pid_col(hops)

    if left_pid is None or right_pid is None:
        st.error("'player.id' or 'player_id' is missing from one of the dataframes.")
        return df

    left = df.copy()
    right = hops.copy()

    left["__pid__"] = left_pid
    right["__pid__"] = right_pid

    # If HOPS not present, nothing to merge
    if "HOPS" not in right.columns:
        left.drop(columns=["__pid__"], inplace=True)
        return left

    # Keep only the columns we need from hops
    right = right[["__pid__", "HOPS"]].dropna(subset=["__pid__"])

    # Deduplicate per player: prefer max numeric HOPS, else first non-null
    hops_num = pd.to_numeric(right["HOPS"], errors="coerce")
    if hops_num.notna().any():
        right = (
            pd.DataFrame({"__pid__": right["__pid__"], "HOPS": hops_num})
            .groupby("__pid__", as_index=False)["HOPS"]
            .max()
        )
    else:
        right = right.dropna(subset=["HOPS"]).drop_duplicates("__pid__", keep="first")

    merged = left.merge(right, on="__pid__", how="left")
    merged.drop(columns=["__pid__"], inplace=True)
    return merged


def _available_team_position(df: pd.DataFrame):
    teams = ["All teams"] + sorted(df.get("team.name", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    positions = ["All positions"] + sorted(df.get("primary_position", pd.Series(dtype=object)).dropna().astype(str).unique().tolist())
    return teams, positions


def _detect_schema(bdf: pd.DataFrame) -> str:
    if "set_piece_type" in bdf.columns:
        return "LONG"
    for m in SP_METRICS:
        for suf in SP_SUFFIX_BY_LABEL.values():
            if f"{m}_{suf}" in bdf.columns:
                return "WIDE"
    return "UNKNOWN"

@st.cache_data(show_spinner=False)
def _read_delim(pathlike) -> pd.DataFrame:
    if not pathlike:
        return pd.DataFrame()

    p = Path(pathlike)
    if not p.exists():
        return pd.DataFrame()

    # Try smart sniffing first, then fallbacks for encodings and separators
    attempts = [
        dict(sep=None, engine="python", low_memory=False, on_bad_lines="skip", encoding="utf-8"),
        dict(sep=None, engine="python", low_memory=False, on_bad_lines="skip", encoding="utf-8-sig"),
    ]

    # If extension hints TSV, prefer tab next
    if p.suffix.lower() == ".tsv":
        attempts += [
            dict(sep="\t", low_memory=False, on_bad_lines="skip", encoding="utf-8"),
            dict(sep="\t", low_memory=False, on_bad_lines="skip", encoding="utf-8-sig"),
            dict(sep="\t", low_memory=False, on_bad_lines="skip", encoding="latin1"),
        ]
    else:
        attempts += [
            dict(sep=",", low_memory=False, on_bad_lines="skip", encoding="utf-8"),
            dict(sep=",", low_memory=False, on_bad_lines="skip", encoding="utf-8-sig"),
            dict(sep=",", low_memory=False, on_bad_lines="skip", encoding="latin1"),
        ]

    for kw in attempts:
        try:
            # compression="infer" is default, so .gz, .zip, .bz2, .xz are handled
            return pd.read_csv(p, **kw)
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_csv(pathlike) -> pd.DataFrame:
    try:
        p = Path(pathlike)
        if not p.exists():
            return pd.DataFrame()
        if p.suffix.lower() == ".tsv":
            return pd.read_csv(p, sep="\t", low_memory=False, on_bad_lines="skip", encoding="utf-8")
        # CSV or unknown, try utf-8 then latin1
        try:
            return pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="utf-8")
        except Exception:
            return pd.read_csv(p, low_memory=False, on_bad_lines="skip", encoding="latin1")
    except Exception:
        return pd.DataFrame()


def calculate_age(birth_date):
    try:
        bd = pd.to_datetime(birth_date, errors="coerce")
        if pd.isna(bd):
            return np.nan
        today = pd.Timestamp.today().normalize()
        age = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
        return int(age)
    except Exception:
        return np.nan
# ---- StatsBomb: player bio for a single (comp, season)
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_bio_map(comp_id: int, season_id: int, user: str | None, pwd: str | None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      player.id, primary_position, player_season_minutes, player_age
    Robust to minor schema differences and nested player objects.
    """
    out_cols = ["player.id", "primary_position", "player_season_minutes", "player_age"]
    empty = pd.DataFrame(columns=out_cols)

    if not user or not pwd:
        return empty

    url = f"https://data.statsbomb.com/api/v4/competitions/{int(comp_id)}/seasons/{int(season_id)}/player-stats"
    try:
        r = requests.get(
            url,
            auth=HTTPBasicAuth(user, pwd),
            timeout=45,
            headers={
                "Accept": "application/json",
                "User-Agent": "brfc-sp-app/1.0",
            },
        )
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return empty

    if r.status_code == 429:
        st.warning("Rate limited by StatsBomb. Please try again in a moment.")
        return empty
    if r.status_code == 401:
        st.error("StatsBomb credentials rejected (401).")
        return empty
    if r.status_code == 403:
        st.error("StatsBomb access forbidden (403). Check your plan or IP allowlist.")
        return empty
    if r.status_code != 200:
        st.error(f"Failed to fetch data. Status code: {r.status_code}")
        return empty

    txt = (r.text or "").strip()
    if not txt:
        st.warning("No data returned from API.")
        return empty

    # Parse and unwrap common container keys
    try:
        rows = r.json()
        if isinstance(rows, dict):
            for key in ("data", "players", "items", "results"):
                if key in rows and isinstance(rows[key], list):
                    rows = rows[key]
                    break
    except Exception as e:
        st.error(f"Error parsing JSON data: {e}")
        return empty

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No player data found.")
        return empty

    # If there is a nested player object, normalize it to columns
    # Common shapes: a column named "player" with dicts, or "person"
    for nested_col in ("player", "person"):
        if nested_col in df.columns and df[nested_col].apply(lambda x: isinstance(x, dict)).any():
            norm = pd.json_normalize(df[nested_col])
            # prefix to avoid collisions, but also expose id and position
            for c in norm.columns:
                pref = f"{nested_col}.{c}"
                if pref not in df.columns:
                    df[pref] = norm[c]

    # helper: case-insensitive column pick
    def pick(*choices):
        lower = {c.lower(): c for c in df.columns}
        for ch in choices:
            if ch.lower() in lower:
                return lower[ch.lower()]
        return None

    # Try flat and normalized names
    pid_col  = pick("player.id", "player_id", "playerId", "id", "player.id.x", "player.person_id", "player.personId", "player.id_y")
    pos_col  = pick("primary_position", "primaryPosition", "position", "player.position", "player.primary_position")
    mins_col = pick("player_season_minutes", "player_minutes", "minutes", "player.minutes")
    bday_col = pick("player_birth_date", "birth_date", "dob", "date_of_birth", "player.birth_date", "player.dob")

    if pid_col is None:
        st.warning("Player ID column not found in StatsBomb response.")
        return empty

    # Coerce and build outputs
    pid = pd.to_numeric(df[pid_col], errors="coerce")

    if pos_col:
        primary_position = df[pos_col].astype(str).str.strip()
    else:
        primary_position = pd.Series(pd.NA, index=df.index, dtype="object")

    if mins_col:
        minutes = pd.to_numeric(df[mins_col], errors="coerce").fillna(0.0)
    else:
        minutes = pd.Series(0.0, index=df.index, dtype="float")

    if bday_col:
        player_age = df[bday_col].apply(calculate_age)
    else:
        player_age = pd.Series(np.nan, index=df.index, dtype="float")

    out = pd.DataFrame(
        {
            "player.id": pid,
            "primary_position": primary_position,
            "player_season_minutes": minutes,
            "player_age": player_age,
        }
    ).dropna(subset=["player.id"])

    # One row per player
    out = out.sort_values("player.id").drop_duplicates("player.id", keep="first")
    # Ensure correct column order
    return out[out_cols]
def _merge_bio_prefer_api(base: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """
    Merge on player id and prefer API values for:
      - primary_position
      - player_season_minutes
      - player_age
    """
    if base.empty or bio.empty:
        return base

    # Determine base key
    base_key = "player.id" if "player.id" in base.columns else ("player_id" if "player_id" in base.columns else None)
    if base_key is None:
        return base

    # Standardize key to "player.id" on both sides
    left = base.copy()
    if base_key == "player_id":
        left.rename(columns={"player_id": "player.id"}, inplace=True)

    right = bio.copy()
    if "player.id" not in right.columns and "player_id" in right.columns:
        right.rename(columns={"player_id": "player.id"}, inplace=True)
    if "player.id" not in right.columns:
        return base

    # Coerce to numeric keys and drop missing
    left["player.id"] = pd.to_numeric(left["player.id"], errors="coerce")
    right["player.id"] = pd.to_numeric(right["player.id"], errors="coerce")
    left = left.dropna(subset=["player.id"])
    right = (
        right.dropna(subset=["player.id"])
             .sort_values("player.id")
             .drop_duplicates("player.id", keep="first")
    )

    # Keep only the API columns we might use to avoid a flood of __api columns
    keep_api = ["player.id", "primary_position", "player_season_minutes", "player_age"]
    right = right[[c for c in keep_api if c in right.columns]]

    merged = left.merge(right, on="player.id", how="left", suffixes=("", "__api"))

    # Prefer API values if present
    def coalesce(col: str):
        api_col = f"{col}__api"
        if api_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[api_col].combine_first(merged[col])
                merged.drop(columns=[api_col], inplace=True)
            else:
                merged.rename(columns={api_col: col}, inplace=True)

    for col in ["primary_position", "player_season_minutes", "player_age"]:
        coalesce(col)

    # Ensure numeric types where appropriate
    if "player_season_minutes" in merged.columns:
        merged["player_season_minutes"] = pd.to_numeric(merged["player_season_minutes"], errors="coerce")
    if "player_age" in merged.columns:
        merged["player_age"] = pd.to_numeric(merged["player_age"], errors="coerce")

    # Restore original key name if needed
    if base_key == "player_id":
        merged.rename(columns={"player.id": "player_id"}, inplace=True)

    # Preserve original column order, then append any new cols at the end
    orig_cols = list(base.columns)
    new_cols = [c for c in merged.columns if c not in orig_cols]
    return merged[orig_cols + new_cols]
# ========= UI =========
st.title("🧩 Set Pieces - Players (Latest Season)")



# Helper function to display error messages and stop further execution
def display_error_message(message: str):
    st.error(message)
    st.stop()

# latest-season CSV/TSV
latest_df, latest_path = _read_latest_mapping()

if latest_path is None or latest_df.empty:
    searched = "\n".join(str(p) for p in LATEST_CSV_CANDIDATES)
    display_error_message(
        "Could not find or read latest-season mapping file.\nSearched:\n" + searched
    )

required_cols = {"league_label", "competition_id", "season_id"}
if not required_cols.issubset(set(latest_df.columns)):
    display_error_message(
        f"`{latest_path.name}` missing required columns {required_cols}. "
        f"Found: {list(latest_df.columns)}"
    )

# Build safe names for folders
latest_df["safe_league"] = latest_df["league_label"].map(_safe_name)

# Apply local filter only if (a) local AND (b) user didn’t opt to show all
if IS_LOCAL and not st.session_state.get("psp_show_all", True):
    try:
        present_folders = {p.name for p in DRIVE_ROOT.iterdir() if p.is_dir()}
    except Exception:
        present_folders = set()

    filtered = latest_df[latest_df["safe_league"].isin(present_folders)]

    # If filtering nukes variety (e.g., leaves 0 or 1 leagues), skip it
    if filtered["safe_league"].nunique() >= 3:
        latest_df = filtered
    else:
        st.info("Showing all leagues (local folder filter skipped — few matching folders found).")

# League select
league_options = sorted(latest_df["league_label"].dropna().astype(str).unique().tolist())
if not league_options:
    display_error_message("No leagues available in the latest-season mapping.")

league_pick = st.selectbox("League (latest season only)", league_options)

row = latest_df[latest_df["league_label"] == league_pick].iloc[0]
safe_folder = row["safe_league"]
comp_id     = int(row["competition_id"])
season_id   = int(row["season_id"])

TOP_FOLDER_ID = st.secrets.get("drive", {}).get("folder_id")
SP_ROOT_NAME  = st.secrets.get("drive", {}).get("set_piece_root_name", "Set Piece Stats")

LEAGUE_FOLDER_ID = None
if svc and TOP_FOLDER_ID:
    sp_root_id = drive_resolve_path(svc, TOP_FOLDER_ID, SP_ROOT_NAME)
    if sp_root_id:
        LEAGUE_FOLDER_ID = drive_resolve_path(svc, sp_root_id, safe_folder)

# Resolve league directory
if IS_LOCAL:
    league_dir = DRIVE_ROOT / safe_folder
    if not league_dir.exists():
        display_error_message(f"Folder not found: {league_dir}")
else:
    # On cloud, create a cache folder now; files will be fetched later if missing
    league_dir = DRIVE_ROOT / safe_folder
    league_dir.mkdir(parents=True, exist_ok=True)

# Function to pick the most recent file based on modification time
def _pick_latest(files):
    if not files:
        return None
    try:
        files.sort(key=os.path.getmtime, reverse=True)
    except Exception:
        files = sorted(files)
    return files[0]

totals_path    = _pick_latest(list(league_dir.glob("SP_totals_*.csv")))
breakdown_path = _pick_latest(list(league_dir.glob("SP_breakdown_*.csv")))

def _download_latest(needle: str, dest_dir: Path) -> Path | None:
    if not svc:
        return None
    # search inside the league folder first
    files = []
    if LEAGUE_FOLDER_ID:
        files = list_files_from_drive(svc, name_contains=needle, parent_id=LEAGUE_FOLDER_ID)
    # fallback: search under the Set Piece Stats root
    if not files and TOP_FOLDER_ID:
        sp_root_id = drive_resolve_path(svc, TOP_FOLDER_ID, SP_ROOT_NAME)
        if sp_root_id:
            files = list_files_from_drive(svc, name_contains=needle, parent_id=sp_root_id)
    # final fallback: broad search
    if not files:
        files = list_files_from_drive(svc, name_contains=needle)

    if not files:
        return None

    files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    f = files[0]
    out = dest_dir / f["name"]
    if not out.exists():
        download_file(svc, f["id"], out)
    return out

if not totals_path:
    totals_path = _download_latest("SP_totals_", league_dir)
if not breakdown_path:
    breakdown_path = _download_latest("SP_breakdown_", league_dir)

# Cloud helpers to search inside the league subfolder
def _drive_find_child_folder_id(svc, child_name: str) -> str | None:
    cfg = st.secrets.get("drive", {})
    parent_id = cfg.get("folder_id")
    drive_id  = cfg.get("drive_id")

    q = f"name = '{child_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    if parent_id:
        q += f" and '{parent_id}' in parents"

    kwargs = dict(
        q=q,
        fields="files(id,name,parents,driveId)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    if drive_id:
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drive_id
    else:
        kwargs["corpora"] = "allDrives"

    try:
        res = svc.files().list(**kwargs).execute()
        items = res.get("files", [])
        return items[0]["id"] if items else None
    except Exception:
        return None

def _drive_list_latest_in_folder(svc, folder_id: str, needle: str):
    drive_id = st.secrets.get("drive", {}).get("drive_id")
    q = f"name contains '{needle}' and '{folder_id}' in parents and trashed = false"
    kwargs = dict(
        q=q,
        fields="files(id,name,modifiedTime,parents)",
        pageSize=50,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        corpora="drive" if drive_id else "allDrives",
    )
    if drive_id:
        kwargs["driveId"] = drive_id
    try:
        res = svc.files().list(**kwargs).execute()
        files = res.get("files", [])
        files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
        return files
    except Exception:
        return []

# If not found locally, fetch from Google Drive using the service account
if (not totals_path or not breakdown_path) and svc:
    def _download_latest(needle: str, dest_dir: Path) -> Path | None:
        files = []
        # 1) Try inside the league folder
        if LEAGUE_FOLDER_ID:
            files = _drive_list_latest_in_folder(svc, LEAGUE_FOLDER_ID, needle)
        # 2) Try inside the Set Piece Stats root
        if not files and TOP_FOLDER_ID:
            sp_root_id = drive_resolve_path(svc, TOP_FOLDER_ID, SP_ROOT_NAME)
            if sp_root_id:
                files = _drive_list_latest_in_folder(svc, sp_root_id, needle)
        # 3) Broad search
        if not files:
            files = list_files_from_drive(svc, name_contains=needle)

        if not files:
            return None

        f = files[0]
        out = dest_dir / f["name"]
        if not out.exists():
            download_file(svc, f["id"], out)
        return out

    if not totals_path:
        totals_path = _download_latest("SP_totals_", league_dir)
    if not breakdown_path:
        breakdown_path = _download_latest("SP_breakdown_", league_dir)

# Read the files if they exist, otherwise return an empty DataFrame
totals_df    = _read_csv(totals_path) if totals_path else pd.DataFrame()
breakdown_df = _read_csv(breakdown_path) if breakdown_path else pd.DataFrame()

# If both totals and breakdown are empty, provide a user friendly message
if totals_df.empty and breakdown_df.empty:
    display_error_message("Both totals and breakdown files are missing or empty.")
# Filter to this pair if competition_id and season_id are present
def _filter_pair(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "competition_id" in df.columns and "season_id" in df.columns:
        ci = pd.to_numeric(df["competition_id"], errors="coerce")
        si = pd.to_numeric(df["season_id"], errors="coerce")
        return df[(ci == comp_id) & (si == season_id)]
    return df

# Apply the filter to both totals and breakdown dataframes
totals_df = _filter_pair(totals_df)
breakdown_df = _filter_pair(breakdown_df)

# Detect schema and normalize 'set_piece_type' if in "LONG" schema
schema = _detect_schema(breakdown_df) if not breakdown_df.empty else "UNKNOWN"
if schema == "LONG" and "set_piece_type" in breakdown_df.columns:
    breakdown_df["set_piece_type"] = breakdown_df["set_piece_type"].map(_normalize_sp_type)

# HOPS (optional): improve file reading and error handling
def _find_latest_hops_df() -> pd.DataFrame:
    # 1) try local first
    local_patterns = HOPS_GLOB + [str(DRIVE_ROOT / "hops" / "HOPS_results_*.csv")]
    path = _latest_file(local_patterns)
    if path:
        return _read_delim(path)

    # 2) cloud fallback using service account or user OAuth
    try:
        svc_local = SERVICE or get_drive_service()
        if not svc_local:
            st.warning("No Drive credentials available to fetch HOPS.")
            return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not init Drive client: {e}")
        return pd.DataFrame()

    # Prefer searching inside Bristol Rovers FC -> Set Piece Stats -> <league>
    cfg = st.secrets.get("drive", {})
    top_folder_id = cfg.get("folder_id")
    sp_root_name = cfg.get("set_piece_root_name", "Set Piece Stats")

    parent_id = None
    try:
        if top_folder_id and "safe_folder" in globals():
            sp_root_id = drive_resolve_path(svc_local, top_folder_id, sp_root_name)
            if sp_root_id:
                parent_id = drive_resolve_path(svc_local, sp_root_id, safe_folder)
    except Exception:
        parent_id = None

    files = []
    # Try inside the league folder first
    if parent_id:
        files = list_files_from_drive(svc_local, name_contains="HOPS_results_", parent_id=parent_id)

    # Fallback to Set Piece Stats root
    if not files and top_folder_id:
        sp_root_id = drive_resolve_path(svc_local, top_folder_id, sp_root_name)
        if sp_root_id:
            files = list_files_from_drive(svc_local, name_contains="HOPS_results_", parent_id=sp_root_id)

    # Broad search if needed
    if not files:
        files = list_files_from_drive(svc_local, name_contains="HOPS_results_")

    if not files:
        return pd.DataFrame()

    # newest first
    files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    f = files[0]

    dest = DRIVE_ROOT / "hops" / f["name"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        download_file(svc_local, f["id"], dest)
    return _read_delim(dest)


def _fetch_hops_from_drive(service) -> pd.DataFrame:
    parent = LEAGUE_FOLDER_ID or st.secrets.get("drive", {}).get("folder_id")
    files = list_files_from_drive(service, name_contains="HOPS_results", parent_id=parent)
    if not files:
        st.warning("No HOPS_results files found on Drive.")
        return pd.DataFrame()
    files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
    f = files[0]
    dest_dir = league_dir if league_dir else (DRIVE_ROOT / "hops")
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f["name"]
    if not out_path.exists():
        download_file(service, f["id"], out_path)
    return _read_delim(out_path)


# thin wrapper kept for compatibility with older calls
def download_hops_from_drive(service, file_id, file_name):
    dest = DRIVE_ROOT / "hops" / file_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        download_file(service, file_id, dest)
    return _read_delim(dest)


# Get the latest HOPS data
# if you have safe_folder in scope, pass it as a hint:
# hops_df = _fetch_hops_from_drive(SERVICE, safe_folder_hint=safe_folder)  # when cloud
# otherwise fall back to your existing finder:
hops_df = _find_latest_hops_df()


SB_USERNAME, SB_PASSWORD = get_statsbomb_creds()
if not SB_USERNAME or not SB_PASSWORD:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()
    
# Check if the credentials are set
if not SB_USERNAME or not SB_PASSWORD:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME and SB_PASSWORD in the environment or add [statsbomb] user/password in secrets.")
    st.stop()
# ==== View choice ====
st.subheader(f"{league_pick} - Choose view")
view_choice = st.radio("View", ["Totals", "Corner", "Free Kick", "Throw-in"], horizontal=True, index=0)

# ==== Build base table ====
def build_base_table(view_choice: str, totals_df: pd.DataFrame, breakdown_df: pd.DataFrame, schema: str) -> pd.DataFrame:
    """Build the base table for view choice: Totals or a specific set piece breakdown."""
    if view_choice == "Totals":
        base = totals_df.copy()
        if base.empty and not breakdown_df.empty:
            # Derive totals from breakdown if needed
            base = _build_totals_from_breakdown(breakdown_df)
            # Carry metadata if present in breakdown
            for c in ["competition_id", "season_id", "league_label", "season_name", "season_norm"]:
                if c in breakdown_df.columns and c not in base.columns and breakdown_df[c].notna().any():
                    base[c] = breakdown_df[c].dropna().iloc[0]
        return base

    # Breakdown views
    if breakdown_df.empty:
        st.info("No breakdown file available for this league yet.")
        return pd.DataFrame()

    sp_map = {"Corner": "Corner", "Free Kick": "Free Kick", "Throw-in": "Throw-in"}
    sp_choice = sp_map.get(view_choice)
    if not sp_choice:
        return pd.DataFrame()

    if schema == "LONG":
        if "set_piece_type" not in breakdown_df.columns:
            st.info("Breakdown file is LONG but missing 'set_piece_type'.")
            return pd.DataFrame()
        return breakdown_df[breakdown_df["set_piece_type"] == sp_choice].copy()

    if schema == "WIDE":
        base = _slice_wide_breakdown(breakdown_df, sp_choice)
        if base.empty:
            st.info(f"No columns found for '{sp_choice}' in breakdown file.")
        return base

    st.info("Breakdown schema is not recognized (neither LONG nor WIDE).")
    return pd.DataFrame()

# Build base table
base = build_base_table(view_choice, totals_df, breakdown_df, schema)

# Merge HOPS data (ensure loaded from Drive or local if available)
if hops_df.empty:
    # Try again; _find_latest_hops_df handles cloud fallback
    hops_df = _find_latest_hops_df()
base = _with_hops(base, hops_df)

# Fetch and merge player-stats bio (primary_position, minutes, age) for this pair - prefer API values
bio_map = fetch_player_bio_map(comp_id, season_id, SB_USERNAME, SB_PASSWORD)
base = _merge_bio_prefer_api(base, bio_map)

# Guard: stop if nothing to show
if base.empty:
    st.info("No data found for this selection (after latest-season filtering). Please adjust the filters or data.")
    st.stop()

# ==== Filters ====
st.subheader("Filters")
# Teams and positions
teams, positions = _available_team_position(base)

# Slider bounds
mins_series = pd.to_numeric(base.get("player_season_minutes", pd.Series(dtype=float)), errors="coerce").fillna(0)
age_series  = pd.to_numeric(base.get("player_age", pd.Series(dtype=float)), errors="coerce")

c1, c2, c3 = st.columns([1.2, 1.2, 2.0])

with c1:
    team_pick = st.selectbox("Team", teams, index=0)

with c2:
    pos_pick = st.selectbox("Primary position", positions, index=0)

with c3:
    max_minutes = int(mins_series.max()) if not mins_series.empty else 0
    min_minutes = st.slider("Minimum minutes", 0, int(max(600, max_minutes)), 0, step=30)

# Optional age slider if we have any non NaN ages
age_min, age_max = None, None
if age_series.notna().any():
    finite_ages = age_series.dropna()
    a_min = int(np.floor(finite_ages.min()))
    a_max = int(np.ceil(finite_ages.max()))
    age_min, age_max = st.slider("Age range", a_min, a_max, (a_min, a_max))

# Apply filters once
f = base.copy()

# Team filter
if team_pick != "All teams" and "team.name" in f.columns:
    f = f[f["team.name"] == team_pick]

# Position filter
if pos_pick != "All positions" and "primary_position" in f.columns:
    f = f[f["primary_position"] == pos_pick]

# Minutes filter
if "player_season_minutes" in f.columns:
    f = f[pd.to_numeric(f["player_season_minutes"], errors="coerce").fillna(0) >= min_minutes]

# Age filter
if age_min is not None and "player_age" in f.columns:
    ages = pd.to_numeric(f["player_age"], errors="coerce")
    mask = (ages.isna()) | ((ages >= age_min) & (ages <= age_max))
    f = f[mask]

# Guard for empty result
if f.empty:
    st.warning("No rows after filters.")
    st.stop()
# ==== Display ====
# Lead columns to show
lead_cols = [c for c in ["player.name", "team.name", "primary_position", "player_age", "player_season_minutes", "HOPS"] if c in f.columns]

# Metric columns based on view choice and schema
if view_choice != "Totals" and _detect_schema(breakdown_df) == "WIDE":
    suf = SP_SUFFIX_BY_LABEL.get(view_choice, "Unknown")
    metric_cols = [f"{m}_{suf}" for m in SP_METRICS if f"{m}_{suf}" in f.columns]
else:
    metric_cols = [m for m in SP_METRICS if m in f.columns]

# Combine and de-dupe
show_cols = list(dict.fromkeys(lead_cols + metric_cols))

# Guard: if no columns survived, stop gracefully
if not show_cols:
    st.warning("No displayable columns after filtering.")
    st.stop()

# Friendly headers
table = _pretty_cols(f[show_cols]).copy()

# Sorting only by stats (exclude identifiers and bio)
blocked = {"Player", "Team", "Primary Position", "Age", "Minutes"}
stat_cols_pretty = [col for col in table.columns if col not in blocked]

# Fallback if no pure stat columns
if not stat_cols_pretty:
    stat_cols_pretty = [col for col in table.columns if col not in {"Player", "Team"}]

c_sort, c_order = st.columns([2, 1])
with c_sort:
    sort_by = st.selectbox("Sort by", stat_cols_pretty, index=0)
with c_order:
    ascending = st.checkbox("Ascending", value=False)

# Build column formatting config automatically
colcfg = {}
for col in table.columns:
    if col in {"Age", "Minutes"}:
        colcfg[col] = st.column_config.NumberColumn(format="%.0f")
    else:
        # Try to infer numeric and pick formatting
        s = pd.to_numeric(table[col], errors="coerce")
        if s.notna().any():
            # xG style metrics get 2 decimals, others default to integer if they look like counts
            if "xG" in col or "Created" in col:
                colcfg[col] = st.column_config.NumberColumn(format="%.2f")
            else:
                # if any value is non-integer after coercion, use two decimals
                if (s.dropna() % 1 != 0).any():
                    colcfg[col] = st.column_config.NumberColumn(format="%.2f")
                else:
                    colcfg[col] = st.column_config.NumberColumn(format="%.0f")

st.dataframe(
    table.sort_values(sort_by, ascending=ascending, ignore_index=True),
    use_container_width=True,
    hide_index=True,
    column_config=colcfg
)

# ==== Download CSV ====
if not table.empty:
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"{_safe_name(league_pick)}_{view_choice.replace(' ', '_')}.csv",
        mime="text/csv"
    )
else:
    st.warning("No data available for download.")

# ==== Files used / debug ====
with st.expander("Files used / debug"):
    st.write("**Latest-season mapping:**", str(latest_path))
    st.write("**League folder:**", str(league_dir))
    st.write("**Totals file:**", str(totals_path) if totals_path else "-")
    st.write("**Breakdown file:**", str(breakdown_path) if breakdown_path else "-")
    st.write("**Breakdown schema:**", _detect_schema(breakdown_df) if not breakdown_df.empty else "UNKNOWN")
    hops_used = _latest_file(HOPS_GLOB + [str(DRIVE_ROOT / "hops" / "HOPS_results_*.csv")])
    st.write("**HOPS file (latest):**", hops_used if hops_used else "-")
    st.write("**Bio rows fetched:**", 0 if bio_map is None else len(bio_map))
