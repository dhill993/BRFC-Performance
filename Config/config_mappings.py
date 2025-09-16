# Config/config_metrics.py

# --- Columns we expect to pull/keep from SB ---
statbomb_metrics_needed = [
    "player_name","team_name","season_name","competition_name","birth_date","Age",
    "player_season_minutes","primary_position",

    # raw stats we use directly or to derive others
    "player_season_aerial_ratio",
    "player_season_ball_recoveries_90",
    "player_season_blocks_per_shot",
    "player_season_carries_90",
    "player_season_crossing_ratio",
    "player_season_deep_progressions_90",
    "player_season_defensive_action_regains_90",
    "player_season_defensive_actions_90",
    "player_season_dribble_faced_ratio",
    "player_season_dribbles_90",
    "player_season_failed_dribbles_90",
    "player_season_dribbled_past_90",
    "player_season_np_shots_90",
    "player_season_np_xg_90",
    "player_season_np_xg_per_shot",
    "player_season_npg_90",
    "player_season_npxgxa_90",
    "player_season_obv_90",
    "player_season_obv_defensive_action_90",
    "player_season_obv_dribble_carry_90",
    "player_season_obv_pass_90",
    "player_season_obv_shot_90",
    "player_season_op_f3_passes_90",
    "player_season_op_key_passes_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_season_op_passes_into_box_90",
    "player_season_padj_clearances_90",
    "player_season_padj_interceptions_90",
    "player_season_padj_pressures_90",
    "player_season_padj_tackles_90",
    "player_season_passing_ratio",
    "player_season_shot_on_target_ratio",
    "player_season_shot_touch_ratio",
    "player_season_touches_inside_box_90",
    "player_season_xgbuildup_90",
    "player_season_op_xa_90",
    "player_season_pressured_passing_ratio",
    "player_season_forward_pass_ratio",
    "player_season_forward_pass_proportion",
    "player_season_fouls_won_90",
    "player_season_pressures_90",
    "player_season_counterpressures_90",
    "player_season_aggressive_actions_90",
    "player_season_op_assists_90",  # for scoring contribution
]

# --- Human-friendly labels for columns shown in UI ---
metrics_mapping = {
    "player_name": "Player Name",
    "team_name": "Team",
    "season_name": "Season",
    "competition_name": "League",
    "player_season_minutes": "Minutes",
    "primary_position": "Position",

    "player_season_aerial_ratio": "Aerial Win %",
    "player_season_ball_recoveries_90": "Ball Recoveries",
    "player_season_blocks_per_shot": "Blocks/Shots",
    "player_season_carries_90": "Carries",
    "player_season_crossing_ratio": "Crossing %",
    "player_season_deep_progressions_90": "Deep Progressions",
    "player_season_defensive_action_regains_90": "Defensive Regains",
    "player_season_defensive_actions_90": "Defensive Actions",
    "player_season_dribble_faced_ratio": "Dribbles Stopped % (SB)",
    "player_season_dribbles_90": "Dribbles",
    "player_season_np_shots_90": "Shots",
    "player_season_np_xg_90": "xG",
    "player_season_np_xg_per_shot": "xG/Shot",
    "player_season_npg_90": "NP Goals",
    "player_season_npxgxa_90": "OP xG Assisted",
    "player_season_obv_90": "OBV",
    "player_season_obv_defensive_action_90": "DA OBV",
    "player_season_obv_dribble_carry_90": "OBV D&C",
    "player_season_obv_pass_90": "Pass OBV",
    "player_season_obv_shot_90": "Shot OBV",
    "player_season_op_f3_passes_90": "OP F3 Passes",
    "player_season_op_key_passes_90": "OP Key Passes",
    "player_season_op_passes_into_and_touches_inside_box_90": "PINTIN",
    "player_season_op_passes_into_box_90": "OP Passes Into Box",
    "player_season_padj_clearances_90": "PADJ Clearances",
    "player_season_padj_interceptions_90": "PADJ Interceptions",
    "player_season_padj_pressures_90": "PADJ Pressures",
    "player_season_padj_tackles_90": "PADJ Tackles",
    "player_season_passing_ratio": "Passing %",
    "player_season_shot_on_target_ratio": "Shooting %",
    "player_season_shot_touch_ratio": "Shot Touch %",
    "player_season_touches_inside_box_90": "Touches in Box",
    "player_season_xgbuildup_90": "xG Buildup",
    "player_season_op_xa_90": "OP xG Assisted",
    "player_season_pressured_passing_ratio": "PR Pass %",
    "player_season_forward_pass_ratio": "Pass Forward %",
    "player_season_forward_pass_proportion": "Pass Forward %",
    "player_season_fouls_won_90": "Fouls Won",
    "player_season_pressures_90": "Pressures",
    "player_season_counterpressures_90": "Counterpressures",
    "player_season_aggressive_actions_90": "Aggressive Actions",

    # Derived columns (you compute in main ETL)
    "player_dribbles_stopped_ratio": "Dribbles Stopped %",
    "player_successful_dribbles_90": "Successful Dribbles",
    "player_successful_crosses_90": "Successful Crosses",
    "player_op_xa_90": "OP xG Assisted",  # alias
    "player_scoring_contribution_90": "Scoring Contribution",
}

# --- Position -> (legacy) Profile mapping (kept as-is for compatibility) ---
position_mapping = {
    "Full Back": "Full Back", "Left Back": "Full Back", "Right Back": "Full Back",
    "Left Wing Back": "Full Back", "Right Wing Back": "Full Back",

    "Centre Back": "Outside Centre Back", "Right Centre Back": "Outside Centre Back",
    "Left Centre Back": "Outside Centre Back",

    "Number 8": "Number 8",
    "Left Defensive Midfielder": "Number 8", "Right Defensive Midfielder": "Number 8",
    "Defensive Midfielder": "Number 8", "Centre Defensive Midfielder": "Number 8",
    "Left Centre Midfield": "Number 8", "Left Centre Midfielder": "Number 8",
    "Right Centre Midfield": "Number 8", "Right Centre Midfielder": "Number 8",
    "Centre Midfield": "Number 8", "Left Attacking Midfield": "Number 8",
    "Right Attacking Midfield": "Number 8", "Right Attacking Midfielder": "Number 8",
    "Attacking Midfield": "Number 8",

    "Secondary Striker": "Number 10", "Centre Attacking Midfielder": "Number 10",
    "Left Attacking Midfielder": "Number 10",

    "Winger": "Winger", "Right Midfielder": "Winger", "Left Midfielder": "Winger",
    "Left Wing": "Winger", "Right Wing": "Winger",

    "Centre Forward": "Runner", "Left Centre Forward": "Runner", "Right Centre Forward": "Runner",
}

# ----------------------------
# Detailed profile metric sets
# ----------------------------

# Full Back → Wing Back
WB_METRICS = [
    "player_dribbles_stopped_ratio",
    "player_season_carries_90",
    "player_op_xa_90",
    "player_successful_dribbles_90",
    "player_successful_crosses_90",
    "player_season_ball_recoveries_90",
    "player_season_op_passes_into_box_90",
    "player_season_pressured_passing_ratio",
    "player_season_padj_interceptions_90",
    "player_season_aerial_ratio",
]

# Centre Back → split Outside vs Middle
OUTSIDE_CB_METRICS = [
    "player_season_aerial_ratio",
    "player_dribbles_stopped_ratio",
    "player_season_padj_tackles_90",
    "player_season_padj_interceptions_90",
    "player_season_padj_clearances_90",
    "player_season_forward_pass_proportion",
    "player_season_defensive_action_regains_90",
    "player_season_obv_defensive_action_90",
    "player_season_ball_recoveries_90",
    "player_season_pressured_passing_ratio",
    "player_season_carries_90",
    "player_successful_crosses_90",
    "player_season_dribbles_90",
    "player_season_op_passes_into_box_90",
    "player_season_op_f3_passes_90",
]

MIDDLE_CB_METRICS = [
    "player_season_aerial_ratio",
    "player_dribbles_stopped_ratio",
    "player_season_padj_tackles_90",
    "player_season_padj_interceptions_90",
    "player_season_padj_clearances_90",
    "player_season_forward_pass_proportion",
    "player_season_defensive_action_regains_90",
    "player_season_obv_defensive_action_90",
    "player_season_ball_recoveries_90",
    "player_season_pressured_passing_ratio",
]

# Centre Midfield → Ball-Playing 6 / Destroyer 6 / Box-to-Box 8 / Technical 8
BALL_PLAYING_6_METRICS = [
    "player_season_padj_tackles_90",
    "player_season_padj_interceptions_90",
    "player_season_forward_pass_proportion",
    "player_season_ball_recoveries_90",
    "player_season_xgbuildup_90",
    "player_dribbles_stopped_ratio",
    "player_season_op_f3_passes_90",
    "player_season_pressured_passing_ratio",
    "player_season_obv_pass_90",
    "player_season_deep_progressions_90",
]

DESTROYER_6_METRICS = [
    "player_season_padj_tackles_90",
    "player_season_padj_interceptions_90",
    "player_season_forward_pass_proportion",
    "player_season_ball_recoveries_90",
    "player_season_aerial_ratio",
    "player_dribbles_stopped_ratio",
    "player_season_obv_defensive_action_90",
    "player_season_pressured_passing_ratio",
    "player_season_defensive_action_regains_90",
    "player_season_deep_progressions_90",
]

BOX_TO_BOX_8_METRICS = [
    "player_season_np_xg_90",
    "player_season_op_key_passes_90",
    "player_season_op_xa_90",
    "player_season_padj_interceptions_90",
    "player_season_padj_tackles_90",
    "player_season_aerial_ratio",
    "player_season_pressures_90",
    "player_season_ball_recoveries_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_scoring_contribution_90",
]

TECHNICAL_8_METRICS = [
    "player_season_np_xg_90",
    "player_season_np_shots_90",
    "player_season_op_xa_90",
    "player_season_op_key_passes_90",
    "player_season_obv_pass_90",
    "player_season_dribbles_90",               # (proxy for dribble involvement)
    "player_season_xgbuildup_90",
    "player_season_carries_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_scoring_contribution_90",
]

# Attacking Midfielder → No10
NO10_METRICS = [
    "player_season_np_shots_90",
    "player_season_np_xg_90",
    "player_scoring_contribution_90",
    "player_season_pressured_passing_ratio",
    "player_season_op_key_passes_90",
    "player_successful_dribbles_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_season_op_xa_90",
    "player_season_carries_90",
    "player_season_shot_on_target_ratio",
]

# Winger → Inverted / Traditional
WINGER_INVERTED_METRICS = [
    "player_season_np_xg_90",
    "player_season_np_shots_90",
    "player_season_op_key_passes_90",
    "player_season_dribbles_90",
    "player_successful_dribbles_90",
    "player_season_obv_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_successful_crosses_90",
    "player_season_op_xa_90",
    "player_season_obv_dribble_carry_90",
]

WINGER_TRADITIONAL_METRICS = [
    "player_season_np_xg_90",
    "player_season_np_shots_90",
    "player_season_op_key_passes_90",
    "player_season_dribbles_90",
    "player_successful_dribbles_90",
    "player_season_carries_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_successful_crosses_90",
    "player_season_obv_dribble_carry_90",
    "player_season_op_xa_90",
]

# Centre Forward → Target Man / Runner
TARGET_MAN_METRICS = [
    "player_season_npg_90",
    "player_season_np_shots_90",
    "player_season_shot_on_target_ratio",
    "player_season_np_xg_90",
    "player_season_np_xg_per_shot",
    "player_season_shot_touch_ratio",
    "player_season_aerial_ratio",
    "player_season_touches_inside_box_90",
    "player_season_carries_90",
    "player_season_fouls_won_90",
]

RUNNER_METRICS = [
    "player_season_npg_90",
    "player_season_np_shots_90",
    "player_season_shot_on_target_ratio",
    "player_season_np_xg_90",
    "player_season_np_xg_per_shot",
    "player_season_shot_touch_ratio",
    "player_season_aggressive_actions_90",
    "player_season_fouls_won_90",
    "player_season_pressures_90",
    "player_season_counterpressures_90",
]

# ----------------------------
# Legacy (coarse) profile sets – kept for backward compatibility
# ----------------------------
PROFILE_METRICS = {
    "Full Back": WB_METRICS,
    "Outside Centre Back": OUTSIDE_CB_METRICS,
    "Number 8": BOX_TO_BOX_8_METRICS,   # legacy fallback
    "Number 10": NO10_METRICS,          # legacy fallback
    "Winger": WINGER_INVERTED_METRICS,  # legacy fallback
    "Runner": RUNNER_METRICS,
}

# ----------------------------
# New detailed profile sets
# ----------------------------
DETAILED_PROFILE_METRICS = {
    # Full Back
    "Wing Back": WB_METRICS,

    # Centre Back
    "Outside Centre Back": OUTSIDE_CB_METRICS,
    "Middle Centre Back": MIDDLE_CB_METRICS,

    # Centre Midfield
    "Ball-Playing 6": BALL_PLAYING_6_METRICS,
    "Destroyer 6": DESTROYER_6_METRICS,
    "Box to Box 8": BOX_TO_BOX_8_METRICS,
    "Technical 8": TECHNICAL_8_METRICS,

    # Attacking Midfielder
    "No10": NO10_METRICS,

    # Winger
    "Inverted Winger": WINGER_INVERTED_METRICS,
    "Traditional Winger": WINGER_TRADITIONAL_METRICS,

    # Centre Forward
    "Target Man": TARGET_MAN_METRICS,
    "Runner": RUNNER_METRICS,
}

# Optional helper for UI groupings (family -> profile types)
PROFILE_FAMILY_TO_TYPES = {
    "Full Back": ["Wing Back"],
    "Centre Back": ["Outside Centre Back", "Middle Centre Back"],
    "Centre Midfield": ["Ball-Playing 6", "Destroyer 6", "Box to Box 8", "Technical 8"],
    "Attacking Midfielder": ["No10"],
    "Winger": ["Inverted Winger", "Traditional Winger"],
    "Centre Forward": ["Target Man", "Runner"],
}

def get_metrics_for_profile(profile: str) -> list[str]:
    """
    Return the metric list for a given profile name.
    Checks detailed sets first, then legacy sets as fallback,
    finally falls back to a sensible generic CM set (Box-to-Box 8).
    """
    if profile in DETAILED_PROFILE_METRICS:
        return DETAILED_PROFILE_METRICS[profile]
    if profile in PROFILE_METRICS:
        return PROFILE_METRICS[profile]
    return BOX_TO_BOX_8_METRICS
