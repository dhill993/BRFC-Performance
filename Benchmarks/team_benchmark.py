# Benchmarks/team_benchmark.py

from __future__ import annotations
from typing import Dict, Iterable, Optional, Tuple, Sequence, List, Set
import numpy as np
import pandas as pd
from Config.config_metrics import team_metrics  # optional: friendly labels mapping
from utils import get_statsbomb_creds

# ----------------------------- Core helpers ----------------------------------

def _detect_team_col(df: pd.DataFrame) -> str:
    """Pick a sensible team-name column."""
    for c in df.columns:
        lc = str(c).lower()
        if lc in ("team_name", "team", "name"):
            return c
    return df.columns[0]


def _metric_columns(df: pd.DataFrame, minutes_col: str) -> List[str]:
    """
    Return numeric metric columns that appear after minutes_col.
    Assumes your data is laid out as: [meta..., minutes_col, metrics...]
    """
    if minutes_col not in df.columns:
        raise ValueError(f"Column '{minutes_col}' not found in dataframe.")
    start_idx = list(df.columns).index(minutes_col)
    metric_candidates = df.columns[start_idx + 1 :]
    return [c for c in metric_candidates if pd.api.types.is_numeric_dtype(df[c])]


def _default_negative_features(cols: Iterable[str]) -> Set[str]:
    """
    Heuristic for lower is better team metrics.
    Expanded to catch defence against names, corners against, and discipline.
    """
    cols = list(cols)
    cols_l = [c.lower() for c in cols]
    s: Set[str] = set()
    for c, lc in zip(cols, cols_l):
        # generic against tokens
        if any(k in lc for k in (
            "conceded", "allowed", "against", "faced",  # defence against
            "opp_", "opponent",                         # explicit opponent prefixes
            "pressures_against", "shots_against"
        )):
            s.add(c)
        # corners conceded or against
        if ("corner" in lc or "corners" in lc) and any(t in lc for t in ("conceded", "against", "allowed", "faced")):
            s.add(c)
        # discipline
        if any(k in lc for k in ("red_cards", "yellow_cards", "cards", "fouls")):
            s.add(c)
        # PPDA lower is better
        if "ppda" in lc:
            s.add(c)

    # explicit safety nets for common column names
    for c in (
        "team_season_ppda",
        "team_season_corners_conceded_pg",
        "team_season_corners_against_pg",
        "team_season_yellow_cards_pg",
        "team_season_second_yellow_cards_pg",
        "team_season_red_cards_pg",
    ):
        if c in cols:
            s.add(c)
    return s


def _rank_series(ser: pd.Series, lower_is_better: bool) -> pd.Series:
    """Rank with 1 = best. If lower_is_better, invert ascending."""
    return ser.rank(ascending=lower_is_better, method="min", na_option="bottom")


def _percentile_from_rank(rank_val: float, n: int) -> float:
    """
    Convert rank (1 best) to percentile in [0,1] where 1 = best.
    Fixed: NaN rank produces NaN percentile, not 1.0.
    """
    if n <= 0:
        return np.nan
    if pd.isna(rank_val):
        return np.nan
    if n == 1:
        return 1.0
    return 1.0 - ((float(rank_val) - 1.0) / (n - 1.0))


def ensure_avg_row(
    df: pd.DataFrame,
    team_col: str,
    metric_cols: List[str],
    *,
    minutes_col: str,
    avg_name: str,
    avg_aliases: Tuple[str, ...],
) -> Tuple[pd.DataFrame, str]:
    """
    Ensure an Average row exists (mean of real teams). Returns (work_df, avg_label).
    """
    work_df = df.copy()
    has_avg = work_df[team_col].isin(avg_aliases).any()
    if not has_avg:
        league_mask = ~work_df[team_col].isin(avg_aliases)
        league_means = work_df.loc[league_mask, metric_cols].mean(numeric_only=True)
        avg_row = {team_col: avg_name, minutes_col: np.nan}
        # fill other non-metric columns as NaN
        for c in work_df.columns:
            if c not in (team_col, minutes_col) and c not in metric_cols:
                avg_row[c] = np.nan
        for c in metric_cols:
            avg_row[c] = league_means.get(c, np.nan)
        work_df = pd.concat([work_df, pd.DataFrame([avg_row])], ignore_index=True)
        avg_label = avg_name
    else:
        # use the first average alias as canonical label
        avg_label = work_df.loc[work_df[team_col].isin(avg_aliases), team_col].iloc[0]
    return work_df, avg_label


# ----------------------------- Grouping logic --------------------------------

def _infer_group_from_key_and_label(key: str, label: str) -> str:
    """
    Lightweight grouping for UI sections:
      - Defence for conceded, allowed, against, faced, PPDA, xGA against, saves%, etc.
      - Discipline for cards and fouls.
      - Attack otherwise.
    """
    k = (key or "").lower()
    l = (label or "").lower()

    defence_tokens = [
        "conceded", "allowed", "against", "faced",
        "opp_", "opponent", "defence", "defense", "save", "saves",
        "ga", "xga", "ppda", "pressures_against", "shots_against", "corner"
    ]
    discipline_tokens = ["yellow", "red", "cards", "foul", "discipline"]

    if any(t in k or t in l for t in discipline_tokens):
        return "Discipline"
    if any(t in k or t in l for t in defence_tokens):
        return "Defence"
    return "Attack"


def _make_group_series(keys: Sequence[str], mapping: Optional[Dict[str, str]]) -> pd.Series:
    """Build a Group series aligned to metric keys."""
    labels = [mapping.get(k, k) if mapping else k for k in keys]
    return pd.Series([_infer_group_from_key_and_label(k, lbl) for k, lbl in zip(keys, labels)], index=range(len(keys)))


# ------------------------------ Public API -----------------------------------

def build_team_comparison(
    df: pd.DataFrame,
    team1: str,
    team2: Optional[str] = None,
    *,
    minutes_col: str = "team_season_minutes",
    avg_name: str = "Average",
    avg_aliases: Tuple[str, ...] = ("Average", "Benchmark"),
    decimals: int = 2,
    metrics_mapping: Optional[Dict[str, str]] = None,
    negative_features: Optional[Iterable[str]] = None,
    include_metrics: Optional[Iterable[str]] = None,
    include_groups: Optional[Iterable[str]] = None,
    pretty: bool = True,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
      - MetricKey, Metric, Group
      - <team1>, <target>, Comparison (+/-/= raw), ranks and percentiles
    """
    team_col = _detect_team_col(df)
    metric_cols_all = _metric_columns(df, minutes_col)

    # subset to metric keys if provided
    if include_metrics:
        include_set = set(include_metrics)
        metric_cols = [c for c in metric_cols_all if c in include_set] or metric_cols_all
    else:
        metric_cols = metric_cols_all

    # negatives
    neg_set = set(negative_features) if negative_features is not None else _default_negative_features(metric_cols)

    # Average row
    work_df, avg_label = ensure_avg_row(
        df, team_col, metric_cols, minutes_col=minutes_col, avg_name=avg_name, avg_aliases=avg_aliases
    )
    target_label = team2 if team2 else avg_label

    # Guards
    all_teams = set(work_df[team_col])
    if team1 not in all_teams:
        raise ValueError(f"Team1 '{team1}' not found in '{team_col}'.")
    if target_label not in all_teams:
        raise ValueError(f"Target '{target_label}' not found in '{team_col}'.")

    # Raw rows
    row1_raw = work_df.loc[work_df[team_col] == team1, metric_cols].iloc[0]
    row2_raw = work_df.loc[work_df[team_col] == target_label, metric_cols].iloc[0]

    # Base output
    out = pd.DataFrame({
        "MetricKey": metric_cols,
        team1: row1_raw.values,
        target_label: row2_raw.values
    })

    # Group column
    groups = _make_group_series(metric_cols, metrics_mapping)
    out["Group"] = groups.values

    # Optional group filter
    if include_groups:
        keep = set(g.lower() for g in include_groups)
        out = out[out["Group"].str.lower().isin(keep)]
        metric_cols = out["MetricKey"].tolist()  # keep alignment

    # Sign compare
    def _sign(key: str, a: float, b: float) -> str:
        if pd.isna(a) or pd.isna(b):
            return ""
        if key in neg_set:
            return "+" if a < b else "-" if a > b else "="
        return "+" if a > b else "-" if a < b else "="

    out["Comparison"] = out.apply(lambda r: _sign(r["MetricKey"], r[team1], r[target_label]), axis=1)

    # League-only for rank and percentile
    league_mask = ~work_df[team_col].isin(avg_aliases)
    league = work_df.loc[league_mask, [team_col] + metric_cols].copy()

    team1_ranks, team2_ranks, team1_pcts, team2_pcts = [], [], [], []
    for m in metric_cols:
        sub = league[[team_col, m]].dropna(subset=[m])
        if sub.empty:
            team1_ranks.append(np.nan); team2_ranks.append(np.nan)
            team1_pcts.append(np.nan);  team2_pcts.append(np.nan)
            continue
        lower_is_better = (m in neg_set)
        sub["__rank"] = _rank_series(sub[m], lower_is_better)
        n = int(sub.shape[0])

        r1 = sub.loc[sub[team_col] == team1, "__rank"]
        if r1.empty:
            team1_ranks.append(np.nan); team1_pcts.append(np.nan)
        else:
            r1v = float(r1.iloc[0]); team1_ranks.append(r1v); team1_pcts.append(_percentile_from_rank(r1v, n))

        if target_label in avg_aliases or target_label == avg_name:
            team2_ranks.append(np.nan); team2_pcts.append(np.nan)
        else:
            r2 = sub.loc[sub[team_col] == target_label, "__rank"]
            if r2.empty:
                team2_ranks.append(np.nan); team2_pcts.append(np.nan)
            else:
                r2v = float(r2.iloc[0]); team2_ranks.append(r2v); team2_pcts.append(_percentile_from_rank(r2v, n))

    out[f"{team1}_rank"] = team1_ranks
    out[f"{target_label}_rank"] = team2_ranks
    out[f"{team1}_pct"] = team1_pcts
    out[f"{target_label}_pct"] = team2_pcts

    # Sort by absolute raw diff within groups
    base_series = row1_raw.reindex(out["MetricKey"]).astype(float) - row2_raw.reindex(out["MetricKey"]).astype(float)
    out["abs_diff"] = base_series.abs().values
    out = out.sort_values(["Group", "abs_diff"], ascending=[True, False]).drop(columns=["abs_diff"]).reset_index(drop=True)

    # Friendly labels
    if metrics_mapping:
        out["Metric"] = out["MetricKey"].map(metrics_mapping).fillna(out["MetricKey"])
    else:
        out["Metric"] = out["MetricKey"]

    # Pretty values (leave rank and pct numeric)
    if pretty:
        fmt = f"{{:,.{decimals}f}}"
        out[team1] = out[team1].map(lambda x: "" if pd.isna(x) else fmt.format(x))
        out[target_label] = out[target_label].map(lambda x: "" if pd.isna(x) else fmt.format(x))

    # Final order
    cols = ["MetricKey", "Metric", "Group", team1, target_label, "Comparison",
            f"{team1}_rank", f"{target_label}_rank", f"{team1}_pct", f"{target_label}_pct"]
    out = out[[c for c in cols if c in out.columns]]
    return out


def build_team_percentiles_payload(
    df: pd.DataFrame,
    team: str,
    *,
    minutes_col: str = "team_season_minutes",
    avg_aliases: Tuple[str, ...] = ("Average", "Benchmark"),
    metrics_mapping: Optional[Dict[str, str]] = None,
    negative_features: Optional[Iterable[str]] = None,
    include_metrics: Optional[Iterable[str]] = None,
    include_groups: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Tidy dataframe for a single team:
      - MetricKey, Metric, Group
      - raw_value, rank (1 best), percentile [0..1]
    """
    team_col = _detect_team_col(df)
    metric_cols_all = _metric_columns(df, minutes_col)

    # subset to metric keys if provided
    if include_metrics:
        include_set = set(include_metrics)
        metric_cols = [c for c in metric_cols_all if c in include_set] or metric_cols_all
    else:
        metric_cols = metric_cols_all

    neg_set = set(negative_features) if negative_features is not None else _default_negative_features(metric_cols)

    if team not in set(df[team_col]):
        raise ValueError(f"Team '{team}' not found in '{team_col}'.")

    # league only (exclude Average-type rows)
    league = df.loc[~df[team_col].isin(avg_aliases), [team_col] + metric_cols].copy()
    row = league.loc[league[team_col] == team, metric_cols]
    if row.empty:
        row = df.loc[df[team_col] == team, metric_cols]
    row = row.iloc[0]

    # Build base rows
    out_rows = []
    for m in metric_cols:
        sub = league[[team_col, m]].dropna(subset=[m])
        if sub.empty:
            out_rows.append({"MetricKey": m, "raw_value": np.nan, "rank": np.nan, "percentile": np.nan})
            continue

        lower_is_better = (m in neg_set)
        sub["__rank"] = _rank_series(sub[m], lower_is_better)
        n = int(sub.shape[0])
        r = sub.loc[sub[team_col] == team, "__rank"]
        if r.empty:
            out_rows.append({"MetricKey": m, "raw_value": row[m], "rank": np.nan, "percentile": np.nan})
        else:
            rv = float(r.iloc[0])
            out_rows.append({
                "MetricKey": m,
                "raw_value": row[m],
                "rank": rv,
                "percentile": _percentile_from_rank(rv, n),
            })

    out = pd.DataFrame(out_rows)

    # Attach labels and groups
    if metrics_mapping:
        out["Metric"] = out["MetricKey"].map(metrics_mapping).fillna(out["MetricKey"])
    else:
        out["Metric"] = out["MetricKey"]
    out["Group"] = _make_group_series(out["MetricKey"].tolist(), metrics_mapping).values

    # Optional group filter
    if include_groups:
        keep = set(g.lower() for g in include_groups)
        out = out[out["Group"].str.lower().isin(keep)]

    # Nice order
    out = out.sort_values(["Group", "percentile"], ascending=[True, False]).reset_index(drop=True)
    return out


def render_team_comparison_table_ranked(df: pd.DataFrame, team1: str, team2: str) -> None:
    """
    Render an HTML comparison table using rank-based coloring for team columns
    and outcome coloring for the last arrow column.

    CSS classes used (define these in your styles.css):
      - rank-green / rank-amber / rank-red
      - comp-green / comp-red / comp-equal
    """
    def rank_class(r):
        if pd.isna(r):
            return ""
        r = int(r)
        if 1 <= r <= 6:
            return "rank-green"
        if 7 <= r <= 17:
            return "rank-amber"
        return "rank-red"

    arrow_map = {"+": "⬆️", "-": "⬇️", "=": "↔"}

    def comp_class(sign: str) -> str:
        if sign == "+":
            return "comp-green"
        if sign == "-":
            return "comp-red"
        return "comp-equal"

    r1_col = f"{team1}_rank"
    r2_col = f"{team2}_rank" if f"{team2}_rank" in df.columns else None

    # Safer grouping when Group is missing
    group_key = df["Group"] if "Group" in df.columns else pd.Series([""] * len(df), index=df.index)

    from streamlit import markdown

    for gname, chunk in df.groupby(group_key):
        if gname:
            markdown(f"#### {gname}")
        html = [
            "<table class='comparison-table'>",
            "<thead>",
            f"<tr><th>📌 Metric</th><th>🔵 {team1}</th><th>🟡 {team2}</th><th>↕</th></tr>",
            "</thead><tbody>",
        ]
        for _, row in chunk.iterrows():
            sign = row.get("Comparison", "")
            arrow = arrow_map.get(sign, sign)
            outcome_cls = comp_class(sign)

            cls1 = rank_class(row.get(r1_col, np.nan))
            cls2 = "" if team2.lower() == "average" else rank_class(row.get(r2_col, np.nan))

            html.append("<tr>")
            html.append(f"<td>{row['Metric']}</td>")
            html.append(f"<td class='{cls1}'>{row[team1]}</td>")
            html.append(f"<td class='{cls2}'>{row[team2]}</td>")
            html.append(f"<td class='comp-cell {outcome_cls}'>{arrow}</td>")
            html.append("</tr>")
        html.append("</tbody></table>")
        markdown("".join(html), unsafe_allow_html=True)
