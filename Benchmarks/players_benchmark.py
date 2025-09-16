# Benchmarks/players_benchmark.py
from typing import Tuple, Callable, Optional, Dict, List
import numpy as np
import pandas as pd
from utils import get_statsbomb_creds
# --- Metrics where LOWER is better (used for sign logic/percentiles) ---
NEGATIVE_FEATURES: set[str] = {
    "player_season_dribbled_past_90",
    "player_season_dispossessions_90",
    "player_season_turnovers_90",
    "player_season_failed_dribbles_90",
    "player_season_yellow_cards_90",
    "player_season_second_yellow_cards_90",
    "player_season_red_cards_90",
    "player_season_errors_90",
}

# --- Columns we explicitly EXCLUDE from numeric comparisons/universe ---
EXCLUDED_NUMERIC: set[str] = {
    "player_season_most_recent_match",
    "account_id",
    "player_id",
    "team_id",
    "competition_id",
    "season_id",
    "country_id",
    "player_weight",
    "player_height",

    # GK-only columns
    "player_season_shots_faced_90",
    "player_season_goals_faced_90",
    "player_season_np_xg_faced_90",
    "player_season_np_psxg_faced_90",
    "player_season_save_ratio",
    "player_season_xs_ratio",
    "player_season_gsaa_90",
    "player_season_gsaa_ratio",
    "player_season_ot_shots_faced_90",
    "player_season_npot_psxg_faced_90",
    "player_season_ot_shots_faced_ratio",
    "player_season_np_optimal_gk_dlength",

    # Misc. not-for-compare
    "player_season_clcaa",
    "player_season_da_aggressive_distance",
    "player_season_penalties_faced_90",
    "player_season_penalties_conceded_90",
    "player_season_360_minutes",
}

# What we show in the little bio cards
BIO_FIELDS: List[str] = ["player_name", "team_name", "birth_date", "Profile", "player_season_minutes"]


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns minus those we exclude."""
    cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if c not in EXCLUDED_NUMERIC]


def get_player_bio(df: pd.DataFrame, player_name: str, avg_name: str = "Average Player") -> pd.DataFrame:
    """
    Return a tiny bio dataframe (1 row) for the given player.
    For the synthetic average row, Minutes = league/profile mean.
    Minutes are rounded to integers (display-friendly).
    """
    if player_name == avg_name:
        minutes_mean = pd.to_numeric(df.get("player_season_minutes", pd.Series(dtype=float)), errors="coerce").mean()
        bio = pd.DataFrame([{
            "player_name": avg_name,
            "team_name": "League Average",
            "birth_date": "",
            "Profile": "-",
            "player_season_minutes": minutes_mean,
        }])
    else:
        bio = df.loc[df["player_name"] == player_name, BIO_FIELDS].drop_duplicates().copy()

    if "player_season_minutes" in bio.columns:
        bio["player_season_minutes"] = (
            pd.to_numeric(bio["player_season_minutes"], errors="coerce")
            .round(0)
            .astype("Int64")
        )
    return bio


# ---------- helpers for league weighted percentiles ----------

# Benchmarks/players_benchmark.py

# Benchmarks/players_benchmark.py

def _pct_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Percentile rank in [0,1] with 1.0 = best.
    Robust to NaNs and ties.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.empty or s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)

    # Best should get rank=1.  Use ascending=not hib so "best" is always 1.
    ranks = s.rank(ascending=not higher_is_better,
                   method="average", na_option="keep")

    n = ranks.notna().sum()
    if n <= 1:
        # singleton cohort -> treat as 1.0
        return pd.Series(1.0, index=s.index)

    # 1.0 for best (rank=1), 1/n for worst (rank=n)
    pct = (n - ranks + 1) / n
    return pct


def _league_weighted_pct(raw_pct: float, weight: float) -> float:
    """
    Shrink percentile toward 0.5 by league strength weight in [0,1].
    0 -> everything is 0.5; 1 -> keep raw percentile.
    """
    return 0.5 + weight * (raw_pct - 0.5)


def _build_weight_lookup(
    league_weight_func: Optional[Callable[[str, str], float]],
    df: pd.DataFrame
) -> Dict[tuple, float]:
    """
    Build a cache {(country, competition): weight}. If func is None or the
    required columns are missing, return {}.
    """
    if league_weight_func is None:
        return {}
    if "country_name" not in df.columns or "competition_name" not in df.columns:
        return {}
    cache: Dict[tuple, float] = {}
    sub = df[["country_name", "competition_name"]].drop_duplicates()
    for cty, comp in sub.itertuples(index=False):
        if pd.isna(cty) or pd.isna(comp):
            continue
        cache[(str(cty), str(comp))] = float(league_weight_func(cty, comp))
    return cache


# ---------------- NEW: metric selection helpers ----------------

def _resolve_metric_list(
    *,
    player_profile: str,
    metrics_group_name: Optional[str],
    include_metrics: Optional[List[str]],
    get_metrics_for_profile: Callable[[str], List[str]],
    get_metrics_for_group: Optional[Callable[[str], List[str]]] = None,
    detailed_profile_metrics: Optional[Dict[str, List[str]]] = None,
    numeric_cols: List[str],
) -> List[str]:
    """
    Decide which metrics to display:
      1) If include_metrics is provided → intersect with numeric columns.
      2) Else if metrics_group_name provided → pull from get_metrics_for_group() or detailed_profile_metrics dict.
      3) Else → default to get_metrics_for_profile(player_profile).
    Always intersect with numeric columns to avoid non-numeric / missing.
    """
    if include_metrics:
        base = include_metrics
    elif metrics_group_name:
        if get_metrics_for_group is not None:
            base = get_metrics_for_group(metrics_group_name) or []
        elif detailed_profile_metrics is not None and metrics_group_name in detailed_profile_metrics:
            base = detailed_profile_metrics[metrics_group_name]
        else:
            # fallback to role default if group lookup not found
            base = get_metrics_for_profile(player_profile) or []
    else:
        base = get_metrics_for_profile(player_profile) or []

    # keep only numeric and present
    return [m for m in base if m in numeric_cols]


def compare_players_profiled(
    df: pd.DataFrame,
    player1: str,
    player2: Optional[str] = None,
    *,
    metrics_mapping: dict,
    get_metrics_for_profile: Callable[[str], List[str]],
    # NEW: choose any metrics group by name (e.g., "Runner", "Box to Box 8", "Wing Back", etc.)
    metrics_group_name: Optional[str] = None,
    # NEW: resolve named metric groups (if you prefer a function) OR pass detailed_profile_metrics below
    get_metrics_for_group: Optional[Callable[[str], List[str]]] = None,
    # NEW: direct dict alternative for group lookup (e.g., DETAILED_PROFILE_METRICS from Config)
    detailed_profile_metrics: Optional[Dict[str, List[str]]] = None,
    # Keep default fairness: percentiles scoped to league+season+player’s own Profile
    percentile_scope: str = "profile_league",  # "profile_league" or "league"
    # NEW: force the cohort profile used for percentiles (default = player1's Profile)
    cohort_profile: Optional[str] = None,
    league_weight_func: Optional[Callable[[str, str], float]] = None,
    include_metrics: Optional[List[str]] = None,  # explicit metric keys to keep
    avg_name: str = "Average Player",
    decimals: int = 2,
) -> Tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    """
    Returns (table_df, player1_profile, bio1, bio2).

    - Columns (metrics) can come from:
        a) the player's default profile (legacy, if no overrides),
        b) a named metrics group (metrics_group_name),
        c) include_metrics explicit list.
      Regardless of the columns chosen, the percentile *cohort* stays pinned to
      league+season and the *cohort_profile* (default = player1's Profile).

    - Sorting uses weighted percentile gap within the cohort, then raw diff.
    """
    id_col = "player_name"
    if player1 not in set(df.get(id_col, pd.Series([], dtype=object))):
        raise ValueError(f"Player '{player1}' not found.")

    # Pull P1 context
    p1_row = df.loc[df[id_col] == player1].iloc[0]
    p1_comp = p1_row.get("competition_id", None)
    p1_season = p1_row.get("season_id", None)
    p1_profile = str(p1_row.get("Profile", "")) if "Profile" in p1_row else ""
    cohort_prof = str(cohort_profile) if cohort_profile else p1_profile

    # numeric universe
    all_num_cols = _numeric_cols(df)

    # Decide which metrics to display
    whitelist = _resolve_metric_list(
        player_profile=p1_profile,
        metrics_group_name=metrics_group_name,
        include_metrics=include_metrics,
        get_metrics_for_profile=get_metrics_for_profile,
        get_metrics_for_group=get_metrics_for_group,
        detailed_profile_metrics=detailed_profile_metrics,
        numeric_cols=all_num_cols,
    )

    work = df.copy()
    player2_label = player2 if player2 else avg_name

    # Ensure scoped Average row exists (league + season, and profile if requested)
    def _ensure_avg_row() -> None:
        if (work[id_col] == avg_name).any():
            return
        comp_ok = work.get("competition_id", pd.Series(True, index=work.index)) == p1_comp
        season_ok = work.get("season_id", pd.Series(True, index=work.index)) == p1_season
        mask = comp_ok & season_ok
        if percentile_scope == "profile_league" and "Profile" in work.columns:
            mask = mask & (work["Profile"] == cohort_prof)
        if not mask.any():
            # last resort: global mean
            mask = pd.Series(True, index=work.index)

        means = work.loc[mask, all_num_cols].mean(numeric_only=True)
        avg_row = {c: np.nan for c in work.columns}
        avg_row[id_col] = avg_name
        for m, v in means.items():
            avg_row[m] = v
        work.loc[len(work)] = avg_row

    _ensure_avg_row()

    def _get_row_by_name(name: str, cols: List[str]) -> pd.Series:
        rows = work.loc[work[id_col] == name, cols]
        if rows.empty and name == avg_name:
            _ensure_avg_row()
            rows = work.loc[work[id_col] == name, cols]
        if rows.empty:
            raise ValueError(f"Player '{name}' not found in dataframe.")
        return rows.iloc[0]

    # if no metrics survive, short-circuit to empty-but-shaped
    if not whitelist:
        empty = pd.DataFrame({"Metric": [], player1: [], player2_label: []})
        return empty, p1_profile, get_player_bio(work, player1, avg_name), get_player_bio(work, player2_label, avg_name)

    row1 = _get_row_by_name(player1, whitelist)
    row2 = _get_row_by_name(player2_label, whitelist)

    out = pd.DataFrame({
        "Metric": whitelist,
        player1: row1.values,
        player2_label: row2.values,
    })

    # Weighted percentile sorting
    weight_cache = _build_weight_lookup(league_weight_func, work)

    def _ctx(name: str) -> dict:
        if name == avg_name:
            return {
                "comp_id": p1_comp,
                "season_id": p1_season,
                "profile": cohort_prof,  # force cohort profile
                "country": p1_row.get("country_name"),
                "comp_name": p1_row.get("competition_name"),
                "index": None,
            }
        rr = work.loc[work[id_col] == name]
        idx = None if rr.empty else int(rr.index[0])
        r = rr.iloc[0] if not rr.empty else p1_row
        # **Key**: for fair comparison, use the *same* cohort profile as player1
        return {
            "comp_id": p1_comp,
            "season_id": p1_season,
            "profile": cohort_prof,
            "country": r.get("country_name"),
            "comp_name": p1_row.get("competition_name"),
            "index": idx,
        }

    def _group_mask(ctx: dict) -> pd.Series:
        mask = (work["competition_id"] == ctx["comp_id"]) & (work["season_id"] == ctx["season_id"])
        if percentile_scope == "profile_league" and "Profile" in work.columns:
            mask = mask & (work["Profile"] == ctx["profile"])
        # exclude synthetic avg from the distribution
        return mask & (work[id_col] != avg_name)

    ctx1 = _ctx(player1)
    ctx2 = _ctx(player2_label)
    mask1 = _group_mask(ctx1)
    mask2 = _group_mask(ctx2)

    w1 = weight_cache.get((ctx1["country"], ctx1["comp_name"]), 0.70)
    w2 = w1  # same league shrink for fair cohort compare

    def _weighted_pct_for(ctx: dict, mask: pd.Series, metric: str) -> float:
        # By definition, the “Average Player” is 0.5 after shrink
        if player2_label == avg_name and ctx is ctx2:
            return 0.5
        series = work.loc[mask, metric]
        # If the player is not part of the cohort (e.g., different club/league row), we can't rank him inside
        if series.empty or ctx["index"] is None or ctx["index"] not in series.index:
            return np.nan
        hib = metric not in NEGATIVE_FEATURES
        pr = _pct_rank(series, higher_is_better=hib)
        return float(pr.loc[ctx["index"]])

    wpct_delta: list[float] = []
    for m in whitelist:
        a_raw = _weighted_pct_for(ctx1, mask1, m)
        b_raw = _weighted_pct_for(ctx2, mask2, m)
        a = np.nan if pd.isna(a_raw) else _league_weighted_pct(a_raw, w1)
        b = np.nan if pd.isna(b_raw) else _league_weighted_pct(b_raw, w2)
        wpct_delta.append(np.nan if (pd.isna(a) or pd.isna(b)) else abs(a - b))

    out["wpct_delta"] = wpct_delta

    # sort: weighted percentile gap desc, fallback to raw abs diff
    if out["wpct_delta"].notna().any():
        out = out.sort_values("wpct_delta", ascending=False)
    else:
        out["abs_diff"] = (row1 - row2).abs().reindex(out["Metric"]).values
        out = out.sort_values("abs_diff", ascending=False).drop(columns=["abs_diff"])

    out = out.drop(columns=["wpct_delta"]).reset_index(drop=True)

    # label metrics and pretty numbers
    out["Metric"] = out["Metric"].map(metrics_mapping).fillna(out["Metric"])
    fmt = f"{{:,.{decimals}f}}"
    out[player1] = out[player1].map(lambda x: "" if pd.isna(x) else fmt.format(x))
    out[player2_label] = out[player2_label].map(lambda x: "" if pd.isna(x) else fmt.format(x))

    # bios
    bio1 = get_player_bio(work, player1, avg_name)
    bio2 = get_player_bio(work, player2_label, avg_name)

    return out, p1_profile, bio1, bio2
