# Benchmarks/set_pieces.py
from __future__ import annotations

import math
from typing import Optional, Iterable
import numpy as np
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from utils import get_statsbomb_creds
# ==============================
# Low-level HTTP helpers
# ==============================
def _get_json(url: str, user: str, pwd: str, timeout: int = 45):
    """
    GET JSON but DO NOT raise on 404; return None on 404/empty/parse errors.
    """
    try:
        r = requests.get(
            url,
            auth=HTTPBasicAuth(user, pwd),
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
    except requests.RequestException:
        return None

    if r.status_code == 404:
        return None
    try:
        r.raise_for_status()
    except requests.HTTPError:
        return None

    txt = (r.text or "").strip()
    if not txt:
        return None
    try:
        return r.json()
    except Exception:
        return None


def _first_json(urls: Iterable[str], user: str, pwd: str) -> Optional[object]:
    """
    Try a list of URLs in order; return the first non-empty JSON payload.
    """
    for u in urls:
        data = _get_json(u, user, pwd)
        if data:
            return data
    return None


# ==============================
# Matches / Events / Player Stats
# ==============================
def fetch_matches_for_pair(comp_id: int, season_id: int, user: str, pwd: str) -> pd.DataFrame:
    """
    Returns matches for a competition-season. Prefers newer endpoints if available,
    falls back to older ones. Filters to collection_status=='Complete' when present.
    """
    candidates = [
        # keep the v6 you used before
        f"https://data.statsbomb.com/api/v6/competitions/{comp_id}/seasons/{season_id}/matches",
        # fallbacks
        f"https://data.statsbomb.com/api/v5/competitions/{comp_id}/seasons/{season_id}/matches",
        f"https://data.statsbomb.com/api/v4/competitions/{comp_id}/seasons/{season_id}/matches",
    ]
    data = _first_json(candidates, user, pwd) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    if "collection_status" in df.columns:
        df = df[df["collection_status"].astype(str).str.lower() == "complete"].copy()

    # normalize match id column if needed
    # (most feeds use 'match_id'; keep as-is if already present)
    if "matchId" in df.columns and "match_id" not in df.columns:
        df = df.rename(columns={"matchId": "match_id"})

    return df.reset_index(drop=True)


def fetch_events(match_id: int, user: str, pwd: str, comp_id: Optional[int] = None, season_id: Optional[int] = None) -> pd.DataFrame:
    """
    Prefer v8 flat path:   /api/v8/events/{match_id}
    Fall back to:
      /api/v6|v5|v4/matches/{match_id}/events
      /api/v6|v5|v4/competitions/{cid}/seasons/{sid}/matches/{match_id}/events  (if comp/season given)
    Returns an empty DataFrame if nothing works.
    """
    urls = []

    # v8 (what worked in your notebook)
    urls.append(f"https://data.statsbomb.com/api/v8/events/{match_id}")

    # direct (older shapes)
    for v in ("v6", "v5", "v4"):
        urls.append(f"https://data.statsbomb.com/api/{v}/matches/{match_id}/events")

    # nested (older shapes) — only if we know comp/season
    if comp_id is not None and season_id is not None:
        for v in ("v6", "v5", "v4"):
            urls.append(
                f"https://data.statsbomb.com/api/{v}/competitions/{comp_id}/seasons/{season_id}/matches/{match_id}/events"
            )

    data = _first_json(urls, user, pwd)
    if not data:
        return pd.DataFrame()

    try:
        return pd.json_normalize(data)
    except Exception:
        # as a last resort, try a lenient constructor
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()


def fetch_player_stats(comp_id: int, season_id: int, user: str, pwd: str) -> pd.DataFrame:
    """
    Fetch player-stats for a season. Try v4, then v2.
    Normalizes to:
      player_id, player_name, team_name, player_minutes, player_birth_date, season_start_date
    """
    candidates = [
        f"https://data.statsbomb.com/api/v4/competitions/{comp_id}/seasons/{season_id}/player-stats",
        f"https://data.statsbomb.com/api/v2/competitions/{comp_id}/seasons/{season_id}/player-stats",
    ]
    data = _first_json(candidates, user, pwd)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if df.empty:
        return df

    rename_map = {
        "playerId": "player_id",
        "player_id": "player_id",
        "playerName": "player_name",
        "player_name": "player_name",
        "teamName": "team_name",
        "team_name": "team_name",
        "minutes": "player_minutes",
        "player_minutes": "player_minutes",
        "birth_date": "player_birth_date",
        "player_birth_date": "player_birth_date",
        "seasonStartDate": "season_start_date",
        "season_start_date": "season_start_date",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    keep = ["player_id", "player_name", "team_name", "player_minutes", "player_birth_date", "season_start_date"]
    existing = [c for c in keep if c in df.columns]
    return df[existing].copy() if existing else pd.DataFrame()


# ==============================
# Set-piece tagging + aggregations
# ==============================
def tag_set_pieces(events: pd.DataFrame) -> pd.DataFrame:
    """
    Tag set-piece sequences with a 14s window and 7s split into Phase 1/2.
    A new delivery (Corner / Throw-in / Free Kick or Direct FK shot) restarts the window.
    """
    ev = events.copy()

    must_have = [
        "period", "minute", "second", "index",
        "team.name", "player.name", "player.id",
        "type.name", "pass.type.name",
        "shot.type.name", "shot.outcome.name", "shot.statsbomb_xg", "shot.key_pass_id",
        "play_pattern.name", "id",
    ]
    for c in must_have:
        if c not in ev.columns:
            ev[c] = pd.NA

    ev["minute"] = pd.to_numeric(ev["minute"], errors="coerce").fillna(0)
    ev["second"] = pd.to_numeric(ev["second"], errors="coerce").fillna(0)
    if "index" not in ev.columns or ev["index"].isna().any():
        ev["index"] = np.arange(len(ev))
    ev["event_time"] = pd.to_timedelta(ev["minute"], unit="m") + pd.to_timedelta(ev["second"], unit="s")
    ev = ev.sort_values(["period", "event_time", "index"], kind="mergesort").reset_index(drop=True)

    def is_pass_delivery(r):
        return (r.get("type.name") == "Pass") and (str(r.get("pass.type.name")) in {"Corner", "Throw-in", "Free Kick"})

    def is_direct_fk_shot(r):
        return (r.get("type.name") == "Shot") and (str(r.get("shot.type.name")) == "Free Kick")

    ev["set_piece_id"] = pd.NA
    ev["set_piece_type"] = pd.NA
    ev["set_piece_phase"] = "Open Play"

    sp_id = 0
    i, n = 0, len(ev)
    while i < n:
        row = ev.iloc[i]
        if is_pass_delivery(row) or is_direct_fk_shot(row):
            sp_id += 1
            t0 = row["event_time"]
            sp_type = row["pass.type.name"] if row["type.name"] == "Pass" else row["shot.type.name"]
            ev.at[i, "set_piece_id"] = sp_id
            ev.at[i, "set_piece_type"] = sp_type
            ev.at[i, "set_piece_phase"] = "Phase 1"

            j = i + 1
            while j < n:
                r = ev.iloc[j]
                if is_pass_delivery(r) or is_direct_fk_shot(r):
                    break
                dt = (r["event_time"] - t0).total_seconds()
                if dt > 14:
                    break
                phase = "Phase 1" if dt <= 7 else "Phase 2"
                ev.at[j, "set_piece_id"] = sp_id
                ev.at[j, "set_piece_type"] = sp_type
                ev.at[j, "set_piece_phase"] = phase
                j += 1
            i = j
            continue
        i += 1

    return ev


def aggregate_match_player_setpieces(ev_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Per-match player aggregation of SP contributions (xG, xG Created, shots, goals,
    first contacts, aerials won, blocks, clearances).
    """
    sp = ev_tagged[ev_tagged["set_piece_id"].notna()].copy()
    if sp.empty:
        return pd.DataFrame()

    # Shots, xG
    shots = sp[sp["type.name"] == "Shot"].copy()
    shots["is_goal"] = (shots["shot.outcome.name"] == "Goal").astype(int)
    shots["xg"] = pd.to_numeric(shots["shot.statsbomb_xg"], errors="coerce").fillna(0.0)
    shooting = shots.groupby(["player.id", "player.name", "team.name"]).agg(
        xG_from_SP=("xg", "sum"),
        SP_shots=("id", "count"),
        SP_goals=("is_goal", "sum"),
    ).reset_index()

    # xG created via key pass
    pass_map = (
        sp[sp["type.name"] == "Pass"][["id", "player.id", "player.name", "team.name"]]
        .dropna(subset=["id"])
        .rename(columns={
            "id": "pass_id",
            "player.id": "passer_id",
            "player.name": "passer_name",
            "team.name": "passer_team",
        })
    )
    shots_kp = shots.dropna(subset=["shot.key_pass_id"]).merge(
        pass_map, left_on="shot.key_pass_id", right_on="pass_id", how="left"
    )
    xg_created = shots_kp.groupby(["passer_id", "passer_name", "passer_team"]).agg(
        xGCreated_from_SP=("xg", "sum"),
        SP_key_passes=("id", "count"),
    ).reset_index().rename(columns={
        "passer_id": "player.id",
        "passer_name": "player.name",
        "passer_team": "team.name",
    })

    # Defensive actions
    parts = []
    clearances = sp[sp["type.name"] == "Clearance"]
    if not clearances.empty:
        parts.append(clearances.groupby(["player.id", "player.name", "team.name"]).size().rename("SP_clearances"))
    blocks = sp[sp["type.name"] == "Block"]
    if not blocks.empty:
        parts.append(blocks.groupby(["player.id", "player.name", "team.name"]).size().rename("SP_blocks"))
    def_table = (
        pd.concat(parts, axis=1).fillna(0).reset_index()
        if parts else pd.DataFrame(columns=["player.id", "player.name", "team.name", "SP_clearances", "SP_blocks"])
    )

    # Aerial duels won
    duels = sp[sp["type.name"] == "Duel"].copy()
    duels["aerial_flag"] = duels["duel.type.name"].fillna("").str.contains("Aerial", case=False)
    duels["won_flag"] = duels["duel.outcome.name"].fillna("") == "Won"
    aerials = duels[duels["aerial_flag"] & duels["won_flag"]]
    aerials_summary = aerials.groupby(["player.id", "player.name", "team.name"]).size().reset_index(name="SP_aerials_won")

    # First contact: second event in each SP sequence
    first_contacts = []
    for sid, g in sp.groupby("set_piece_id", sort=False):
        g2 = g.sort_values(["period", "event_time", "index"], kind="mergesort")
        if len(g2) >= 2:
            fc = g2.iloc[1]
            first_contacts.append([fc.get("player.id"), fc.get("player.name"), fc.get("team.name")])
    fc_df = pd.DataFrame(first_contacts, columns=["player.id", "player.name", "team.name"])
    fc_summary = fc_df.groupby(["player.id", "player.name", "team.name"]).size().reset_index(name="SP_first_contacts_won")

    from functools import reduce
    dfs = [shooting, xg_created, def_table, aerials_summary, fc_summary]
    out = reduce(
        lambda a, b: a.merge(b, on=["player.id", "player.name", "team.name"], how="outer"),
        [d for d in dfs if not d.empty]
    ).fillna(0)

    return out


# ==============================
# Season builders
# ==============================
def build_season_setpieces(comp_id: int, season_id: int, user: str, pwd: str):
    """
    Aggregate per-player set-piece totals across all matches in the season.
    Returns:
      season_totals (DataFrame), season_by_type (placeholder), match_count (int)
    """
    matches = fetch_matches_for_pair(comp_id, season_id, user, pwd)
    rows = []
    for m in matches.itertuples(index=False):
        mid = int(getattr(m, "match_id"))
        ev = fetch_events(mid, user, pwd, comp_id=comp_id, season_id=season_id)
        if ev.empty:
            continue
        tagged = tag_set_pieces(ev)
        match_tbl = aggregate_match_player_setpieces(tagged)
        if not match_tbl.empty:
            rows.append(match_tbl)

    if not rows:
        return pd.DataFrame(), pd.DataFrame(), 0

    season_totals = (
        pd.concat(rows, ignore_index=True)
        .groupby(["player.id", "player.name", "team.name"], as_index=False)
        .sum()
    )
    season_by_type = pd.DataFrame()  # reserved
    return season_totals, season_by_type, len(matches)


def attach_minutes(season_totals: pd.DataFrame, comp_id: int, season_id: int, user: str, pwd: str) -> pd.DataFrame:
    """
    Legacy no-op kept for backwards compatibility with old pages.
    Prefer enrich_with_player_stats().
    """
    return season_totals


# ==============================
# Age & minutes enrichment
# ==============================
def _compute_age_years(birth_date, season_start) -> float:
    try:
        b = pd.to_datetime(birth_date)
        s = pd.to_datetime(season_start)
        return round((s - b).days / 365.25, 1)
    except Exception:
        return math.nan


def enrich_with_player_stats(
    season_totals: pd.DataFrame,
    comp_id: int,
    season_id: int,
    latest_season: bool,
    user: str,
    pwd: str,
) -> pd.DataFrame:
    """
    Join minutes & age from player-stats.
    Enforce minutes threshold: latest season -> 0, older seasons -> 600.
    """
    ps = fetch_player_stats(comp_id, season_id, user, pwd)
    out = season_totals.copy()

    if ps.empty:
        out["player_season_minutes"] = 0
        out["player_age"] = np.nan
        return out

    # Age at season start (if available)
    if "player_birth_date" in ps.columns:
        if "season_start_date" not in ps.columns:
            ps["season_start_date"] = pd.NaT
        ps["player_age"] = ps.apply(
            lambda r: _compute_age_years(r.get("player_birth_date"), r.get("season_start_date")), axis=1
        )
    else:
        ps["player_age"] = np.nan

    if "player_id" not in ps.columns:
        out["player_season_minutes"] = 0
        out["player_age"] = np.nan
        return out

    if "player_minutes" not in ps.columns:
        ps["player_minutes"] = 0

    ps_small = ps[["player_id", "player_minutes", "player_age"]].rename(
        columns={"player_id": "player.id", "player_minutes": "player_season_minutes"}
    )

    out = out.merge(ps_small, on="player.id", how="left")
    out["player_season_minutes"] = pd.to_numeric(out["player_season_minutes"], errors="coerce").fillna(0).astype(float)

    min_required = 0 if latest_season else 600
    out = out[out["player_season_minutes"] >= min_required].copy()

    return out
