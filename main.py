# main.py
import os
import json
import ast
from typing import Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from utils import get_statsbomb_creds




st.set_page_config(layout="wide", page_title="Bristol Rovers Benchmark App")

# ---------- ENV ----------
SB_USER, SB_PASS = get_statsbomb_creds()
if not SB_USER or not SB_PASS:
    st.error("StatsBomb credentials are missing. Set SB_USERNAME/SB_PASSWORD in your environment or add [statsbomb] user/password in secrets.")
    st.stop()

# ---------- CONFIG ----------
CLUB_NAME = "Bristol Rovers"
SEASON_CHOICES = {
    "2025-26 (League Two)": {"competition_id": 5, "season_id": 318, "name": "League Two"},
    "2024-25 (League One)": {"competition_id": 4, "season_id": 317, "name": "League One"},
    "2023-24 (League One)": {"competition_id": 4, "season_id": 281, "name": "League One"},
}
DEFAULT_SEASON_KEY = "2025-26 (League Two)"

# ---------- FETCH ----------
@st.cache_data(show_spinner=False)
def fetch_matches(comp_id: int, season_id: int) -> pd.DataFrame:
    url = f"https://data.statsbomb.com/api/v6/competitions/{comp_id}/seasons/{season_id}/matches"
    r = requests.get(url, auth=HTTPBasicAuth(SB_USER, SB_PASS), timeout=40)
    r.raise_for_status()
    data = r.json() or []
    return pd.DataFrame(data)

# ---------- HELPERS ----------
def _extract_name_from_mapping(d: dict) -> str:
    for key in ("name", "team_name", "home_team_name", "away_team_name"):
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return val
    # last resort: stringify
    return str(d)

def _maybe_parse_dict_string(s: str):
    s = s.strip()
    if not s or (not s.startswith("{")) or (not s.endswith("}")):
        return None
    # try JSON first, then Python literal
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def _get_team_name(val) -> str:
    """
    Accepts:
      - dicts like {"id":..., "name":"Bristol Rovers", ...}
      - strings that look like such dicts
      - plain strings ("Bristol Rovers")
    Returns a clean team name.
    """
    if isinstance(val, dict):
        return _extract_name_from_mapping(val)

    if isinstance(val, str):
        # dict-as-string?
        parsed = _maybe_parse_dict_string(val)
        if isinstance(parsed, dict):
            return _extract_name_from_mapping(parsed)
        # already a plain string name
        return val

    if pd.isna(val):
        return ""
    return str(val)

def _best_datetime(row: pd.Series) -> Optional[pd.Timestamp]:
    for col in ("match_date_time_utc", "match_date_utc", "match_date", "kickoff_time", "kick_off"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            dt = pd.to_datetime(row[col], errors="coerce", utc=True)
            if pd.notna(dt):
                return dt.tz_localize(None)
    return None

def _result_for_club(row: pd.Series, club: str, home_name_col: str, away_name_col: str) -> str:
    hs, as_ = row.get("home_score"), row.get("away_score")
    if pd.isna(hs) or pd.isna(as_):
        return ""
    try:
        hs, as_ = int(hs), int(as_)
    except Exception:
        return ""
    is_home = club.lower() in str(row.get(home_name_col, "")).lower()
    my_goals, opp_goals = (hs, as_) if is_home else (as_, hs)
    if my_goals > opp_goals: return "W"
    if my_goals < opp_goals: return "L"
    return "D"

def _color_result(result: str) -> str:
    color = "#16a34a" if result == "W" else ("#ef4444" if result == "L" else "#f59e0b")
    text = result if result in ("W", "D", "L") else ""
    return f"<span style='display:inline-block;padding:2px 8px;border-radius:999px;background:{color};color:white;font-weight:600'>{text}</span>"

# ---------- CLEAN ----------
def tidy_matches(df_raw: pd.DataFrame, competition_label: str) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw

    df = df_raw.copy()

    # Filter to completed if available; fall back if it wipes everything
    df_try = df.copy()
    if "collection_status" in df_try.columns:
        mask = df_try["collection_status"].astype(str).str.lower() == "complete"
        if mask.any():
            df_try = df_try.loc[mask].copy()
    if df_try.empty:
        df_try = df.copy()

    # Normalize team names (works for dicts, dict-strings, or plain strings)
    if "home_team_name" in df_try.columns and "away_team_name" in df_try.columns:
        df_try["home_team_name_norm"] = df_try["home_team_name"].map(_get_team_name)
        df_try["away_team_name_norm"] = df_try["away_team_name"].map(_get_team_name)
    else:
        df_try["home_team_name_norm"] = df_try.get("home_team", "").map(_get_team_name)
        df_try["away_team_name_norm"] = df_try.get("away_team", "").map(_get_team_name)

    # Keep only BRFC matches
    mask_club = (
        df_try["home_team_name_norm"].str.contains(CLUB_NAME, case=False, na=False) |
        df_try["away_team_name_norm"].str.contains(CLUB_NAME, case=False, na=False)
    )
    df_try = df_try.loc[mask_club].copy()
    if df_try.empty:
        return df_try

    # Display fields
    if {"home_score", "away_score"} <= set(df_try.columns):
        df_try["Score"] = (
            df_try["home_score"].astype("Int64").astype(str) + "–" +
            df_try["away_score"].astype("Int64").astype(str)
        )
    else:
        df_try["Score"] = ""

    df_try["Result"] = df_try.apply(
        lambda r: _result_for_club(r, CLUB_NAME, "home_team_name_norm", "away_team_name_norm"),
        axis=1,
    )
    df_try["Fixture"] = df_try["home_team_name_norm"] + " vs " + df_try["away_team_name_norm"]
    df_try["Competition"] = competition_label

    df_try["Date"] = df_try.apply(_best_datetime, axis=1)
    df_try["Date"] = pd.to_datetime(df_try["Date"], errors="coerce")
    df_try["DateStr"] = df_try["Date"].dt.strftime("%d %b %Y")

    df_try = df_try.sort_values("Date", ascending=False).reset_index(drop=True)
    keep = ["Date", "DateStr", "Fixture", "Score", "Result", "Competition"]
    return df_try[[c for c in keep if c in df_try.columns]]

# ---------- UI ----------
st.markdown("<h1 style='text-align:center;'>📊 Bristol Rovers Benchmark App</h1>", unsafe_allow_html=True)

sel = st.selectbox(
    "Season",
    list(SEASON_CHOICES.keys()),
    index=list(SEASON_CHOICES.keys()).index(DEFAULT_SEASON_KEY),
)
meta = SEASON_CHOICES[sel]

try:
    raw = fetch_matches(meta["competition_id"], meta["season_id"])
    matches = tidy_matches(raw, meta["name"])
except Exception as e:
    st.error(f"Failed to load matches: {e}")
    matches = pd.DataFrame()

# Latest result
st.markdown("### 🔔 Latest Result")
if matches.empty:
    st.info("No completed matches for the selected season.")
else:
    latest = matches.iloc[0]
    st.markdown(
        f"""
        <div style="max-width:780px;margin:0 auto 14px auto;border:1px solid #eee;background:#fff;border-radius:12px;padding:16px;box-shadow:0 4px 16px rgba(0,0,0,0.04);text-align:center;">
          <div style="font-weight:600;color:#6b7280">{latest['DateStr']} • {latest['Competition']}</div>
          <div style="margin-top:6px;font-size:18px;font-weight:600">{latest['Fixture']}</div>
          <div style="margin-top:10px;font-size:22px;font-weight:800;letter-spacing:1px">{latest['Score']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Recent form
st.markdown("### 📊 Recent Form (last 5)")
if matches.empty:
    st.markdown("—")
else:
    emo = {"W": "🟩", "D": "🟨", "L": "🟥"}
    st.markdown(
        f"<div style='font-size:22px'>{' '.join(emo.get(x, '⬜') for x in matches['Result'].head(5))}</div>",
        unsafe_allow_html=True
    )

# Last 10 matches
st.markdown("### 🗓️ Last 10 Matches")
if matches.empty:
    st.markdown("—")
else:
    table = matches.head(10).copy().rename(columns={"DateStr": "Date"})
    table["Result"] = table["Result"].map(_color_result)
    table = table[["Date", "Fixture", "Score", "Result", "Competition"]]
    st.write(table.to_html(escape=False, index=False), unsafe_allow_html=True)

# Footer nav
st.write(""); st.write(""); st.divider()
c1, c2, c3,c4,c5 = st.columns(5)
with c1:
    if st.button("Players", use_container_width=True):
        st.switch_page("pages/1_Players.py")
with c2:
    if st.button("Teams", use_container_width=True):
        st.switch_page("pages/2_Teams.py")
with c3:
    if st.button("Similar Players", use_container_width=True):
        st.switch_page("pages/3_Similar_Players.py")
with c4:
    if st.button("Teams Set Pieces Stats", use_container_width=True):
        st.switch_page("pages/4_Teams_Set_Pieces.py")
with c5:
    if st.button("Players Set Pieces Stats", use_container_width=True):
        st.switch_page("pages/4_Players_Set_Pieces.py")
st.caption("Tip: you can also use the **Pages** sidebar to switch views.")
