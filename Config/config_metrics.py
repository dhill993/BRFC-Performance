# Config/config_metrics.py
import numpy as np
import pandas as pd

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
    "player_season_op_xa_90": "OP XG Assisted",
    "player_season_pressured_passing_ratio": "PR Pass %",
    "player_season_forward_pass_ratio": "Pass Forward %",
    "player_season_forward_pass_proportion": "Pass Forward %",
    "player_season_fouls_won_90": "Fouls Won",
    "player_season_pressures_90": "Pressures",
    "player_season_counterpressures_90": "Counterpressures",
    "player_season_aggressive_actions_90": "Aggressive Actions",

    # Derived columns
    "player_dribbles_stopped_ratio": "Dribbles Stopped %",
    "player_successful_dribbles_90": "Successful Dribbles",
    "player_successful_crosses_90": "Successful Crosses",
    "player_op_xa_90": "OP XG Assisted",
    "player_scoring_contribution_90": "Scoring Contribution",
}

team_metrics = {
    'team_name': 'Team',
    'competition_name': 'Competition',
    'season_name': 'Season',
    'team_season_matches': 'Matches',
    'team_season_minutes': 'Minutes',
    'team_season_gd': 'Goal Difference',
    'team_season_xgd': 'xG Difference',
    'team_season_np_shots_pg': 'Non Penalty Shots',
    'team_season_op_shots_pg': 'Open Play Shots',
    'team_season_op_shots_outside_box_pg': 'Open Play Shots Outside Box',
    'team_season_sp_shots_pg': 'Set Piece Shots',
    'team_season_np_xg_pg': 'Non Penalty xG',
    'team_season_op_xg_pg': 'Open Play xG',
    'team_season_sp_xg_pg': 'Set Piece xG',
    'team_season_np_xg_per_shot': 'Non Penalty xG/Shot',
    'team_season_np_shot_distance': 'Non Penalty Shot Distance',
    'team_season_op_shot_distance': 'Open Play Shot Distance',
    'team_season_sp_shot_distance': 'Set Piece Shot Distance',
    'team_season_possessions': 'Possessions',
    'team_season_possession': 'Possession %',
    'team_season_directness': 'Directness',
    'team_season_pace_towards_goal': 'Pace Towards Goal',
    'team_season_gk_pass_distance': 'Goalkeeper Pass Distance',
    'team_season_gk_long_pass_ratio': 'Goalkeeper Long Pass %',
    'team_season_box_cross_ratio': 'Box Cross %',
    'team_season_passes_inside_box_pg': 'Passes Inside Box',
    'team_season_defensive_distance': 'Defensive Distance',
    'team_season_ppda': 'PPDA',
    'team_season_defensive_distance_ppda': 'Defensive Distance PPDA',
    'team_season_opp_passing_ratio': 'Open Play Passing %',
    'team_season_opp_final_third_pass_ratio': 'Open Play Final Third Pass %',
    'team_season_np_shots_conceded_pg': 'Non Penalty Shots Conceded',
    'team_season_op_shots_conceded_pg': 'Open Play Shots Conceded',
    'team_season_op_shots_conceded_outside_box_pg': 'Open Play Conceded Outside Box',
    'team_season_sp_shots_conceded_pg': 'Set Piece Shots Conceded',
    'team_season_np_xg_conceded_pg': 'Non Penalty xG Conceded',
    'team_season_op_xg_conceded_pg': 'Open Play xG Conceded',
    'team_season_sp_xg_conceded_pg': 'Set Piece xG Conceded',
    'team_season_np_xg_per_shot_conceded': 'Non Penalty xG/Shot Conceded',
    'team_season_np_shot_distance_conceded': 'Non Penalty Shot Distance Conceded',
    'team_season_op_shot_distance_conceded': 'Open Play Shot Distance Conceded',
    'team_season_sp_shot_distance_conceded': 'Set Piece Shot Distance Conceded',
    'team_season_deep_completions_conceded_pg': 'Deep Completions Conceded',
    'team_season_passes_inside_box_conceded_pg': 'Passes Inside Box Conceded',
    'team_season_corners_pg': 'Corners',
    'team_season_corner_xg_pg': 'Corner xG',
    'team_season_xg_per_corner': 'xG/Corner',
    'team_season_free_kicks_pg': 'Free Kicks',
    'team_season_free_kick_xg_pg': 'Free Kick xG',
    'team_season_xg_per_free_kick': 'xG/Free kick',
    'team_season_direct_free_kicks_pg': 'Direct Free Kicks',
    'team_season_direct_free_kick_xg_pg': 'Direct Free Kick xG',
    'team_season_xg_per_direct_free_kick': 'xG/Direct Free Kick',
    'team_season_throw_ins_pg': 'Throw-Ins',
    'team_season_throw_in_xg_pg': 'Throw-In xG',
    'team_season_xg_per_throw_in': 'xG/Throw-In',
    'team_season_ball_in_play_time': 'Ball in Play Time',
    'team_season_counter_attacking_shots_pg': 'Counter Attack Shots',
    'team_season_high_press_shots_pg': 'High Press Shots',
    'team_season_shots_in_clear_pg': 'Shots in Clear',
    'team_season_counter_attacking_shots_conceded_pg': 'Counter Attack Shots Conceded',
    'team_season_shots_in_clear_conceded_pg': 'Shots in Clear Conceded',
    'team_season_aggressive_actions_pg': 'Agressive Actions',
    'team_season_aggression': 'Aggression',
    'team_season_goals_pg': 'Goals',
    'team_season_own_goals_pg': 'Own Goals',
    'team_season_penalty_goals_pg': 'Penalty Goals',
    'team_season_goals_conceded_pg': 'Goals Conceded',
    'team_season_opposition_own_goals_pg': 'Opposition Own Goals',
    'team_season_penalty_goals_conceded_pg': 'Penalty Goals Conceded',
    'team_season_shots_from_corners_pg': 'Shots from Corner',
    'team_season_goals_from_corners_pg': 'Goals from Corner',
    'team_season_shots_from_free_kicks_pg': 'Shots from Free Kick',
    'team_season_goals_from_free_kicks_pg': 'Goals from Free Kick',
    'team_season_direct_free_kick_goals_pg': 'Direct Free Kick Goals',
    'team_season_shots_from_direct_free_kicks_pg': 'Direct Free Kick Shots',
    'team_season_shots_from_throw_ins_pg': 'Throw-Ins Shots',
    'team_season_goals_from_throw_ins_pg': 'Throw-Ins Goals',
    'team_season_direct_free_kick_goals_conceded_pg': 'Direct Free Kick Goals Conceded',
    'team_season_shots_from_direct_free_kicks_conceded_pg': 'Direct Free Kick Shots Conceded',
    'team_season_corners_conceded_pg': 'Corners Conceded',
    'team_season_corner_xg_conceded_pg': 'Corners xG Conceded',
    'team_season_shots_from_corners_conceded_pg': 'Shots from Corners Conceded',
    'team_season_goals_from_corners_conceded_pg': 'Goals from Corners Conceded',
    'team_season_free_kicks_conceded_pg': 'Free Kicks Conceded',
    'team_season_free_kick_xg_conceded_pg': 'Free Kicks xG Conceded',
    'team_season_shots_from_free_kicks_conceded_pg': 'Free Kicks Shots Conceded',
    'team_season_goals_from_free_kicks_conceded_pg': 'Free Kicks Goals Conceded',
    'team_season_direct_free_kicks_conceded_pg': 'Direct Free Kicks Conceded',
    'team_season_direct_free_kick_xg_conceded_pg': 'Direct Free Kicks xG Conceded',
    'team_season_throw_ins_conceded_pg': 'Throw-Ins Conceded',
    'team_season_throw_in_xg_conceded_pg': 'Throw-Ins xG Conceded',
    'team_season_shots_from_throw_ins_conceded_pg': 'Throw-Ins Shots Conceded',
    'team_season_goals_from_throw_ins_conceded_pg': 'Throw-Ins Goals Conceded',
    'team_season_corner_shot_ratio': 'Corner Shot %',
    'team_season_corner_goal_ratio': 'Corner Goal %',
    'team_season_free_kick_shot_ratio': 'Free Kick Shot %',
    'team_season_free_kick_goal_ratio': 'Free Kick Goal %',
    'team_season_direct_free_kick_goal_ratio': 'Direct Free Kick Goal %',
    'team_season_throw_in_shot_ratio': 'Throw-In Shot %',
    'team_season_throw_in_goal_ratio': 'Throw-In Goal %',
    'team_season_xg_per_corner_conceded': 'xG/Corner Conceded',
    'team_season_corner_shot_ratio_conceded': 'Corner Shot Conceded %',
    'team_season_corner_goal_ratio_conceded': 'Corner Goal Conced %',
    'team_season_xg_per_free_kick_conceded': 'xG/Free Kick Conceded',
    'team_season_free_kick_shot_ratio_conceded': 'Free Kick Shot Conceded %',
    'team_season_free_kick_goal_ratio_conceded': 'Free Kick Goal Conceded %',
    'team_season_xg_per_direct_free_kick_conceded': 'xG Conceded/Direct Free Kick',
    'team_season_direct_free_kick_goal_ratio_conceded': 'Direct Free Kick Goal Conceded %',
    'team_season_xg_per_throw_in_conceded': 'xG Conceded/Throw-In',
    'team_season_throw_in_shot_ratio_conceded': 'Throw-In Shot Conceded %',
    'team_season_throw_in_goal_ratio_conceded': 'Throw-In Goal Conceded %',
    'team_season_direct_free_kick_shot_ratio': 'Direct Free Kick Shot %',
    'team_season_direct_free_kick_shot_ratio_conceded': 'Direct Free Kick Shot Conceded %',
    'team_season_sp_pg': 'Set Piece',
    'team_season_xg_per_sp': 'xG/Set Piece',
    'team_season_sp_shot_ratio': 'Set Piece Shot %',
    'team_season_sp_goals_pg': 'Set Piece Goals',
    'team_season_sp_goal_ratio': 'Set Piece Goals %',
    'team_season_sp_pg_conceded': 'Set Piece Conceded',
    'team_season_xg_per_sp_conceded': 'xG/Set Piece Conceded',
    'team_season_sp_shot_ratio_conceded': 'Set Piece Shot Conceded %',
    'team_season_sp_goals_pg_conceded': 'Set Piece Goals Conceded',
    'team_season_sp_goal_ratio_conceded': 'Set Piece Goals Conceded %',
    'team_season_penalties_won_pg': 'Penalties Won',
    'team_season_penalties_conceded_pg': 'Penalties Conceded',
    'team_season_completed_dribbles_pg': 'Dribbles Completed',
    'team_season_failed_dribbles_pg': 'Dribbles Failed',
    'team_season_total_dribbles_pg': 'Total Dribbles',
    'team_season_dribble_ratio': 'Dribble %',
    'team_season_completed_dribbles_conceded_pg': 'Dribbles Completed Conceded',
    'team_season_failed_dribbles_conceded_pg': 'Dribbles Failed Conceded',
    'team_season_total_dribbles_conceded_pg': 'Total Dribbles Conceded',
    'team_season_opposition_dribble_ratio': 'Opposition Dribble %',
    'team_season_high_press_shots_conceded_pg': 'High Press Shots Conceded',
    'team_season_gd_pg': 'Goal Difference',
    'team_season_np_gd_pg': 'Non Penalty Goal Difference',
    'team_season_xgd_pg': 'xG Difference',
    'team_season_np_xgd_pg': 'Non Penalty xG Difference',
    'team_season_deep_completions_pg': 'Deep Completions',
    'team_season_passing_ratio': 'Pressing %',
    'team_season_pressures_pg': 'Pressures',
    'team_season_counterpressures_pg': 'Counter Pressures',
    'team_season_pressure_regains_pg': 'Pressure Regains',
    'team_season_counterpressure_regains_pg': 'Counter Pressure Regains',
    'team_season_defensive_action_regains_pg': 'Defensive Action Regains',
    'team_season_yellow_cards_pg': 'Yellow Cards',
    'team_season_second_yellow_cards_pg': 'Second Yellow Cards',
    'team_season_red_cards_pg': 'Red Cards',
    'team_season_fhalf_pressures_pg': 'Final Half Pressures',
    'team_season_fhalf_counterpressures_pg': 'Final Half Counter Pressures',
    'team_season_fhalf_pressures_ratio': 'Final Half Pressures',
    'team_season_fhalf_counterpressures_ratio': 'Final Half Counter Pressures %',
    'team_season_crosses_into_box_pg': 'Crosses into Box',
    'team_season_successful_crosses_into_box_pg': 'Successful Crosses into Box',
    'team_season_successful_box_cross_ratio': 'Successful Crosses into Box %',
    'team_season_deep_progressions_pg': 'Deep Progressions',
    'team_season_deep_progressions_conceded_pg': 'Deep Progressions Conceded',
    'team_season_obv_pg': 'OBV',
    'team_season_obv_pass_pg': 'Pass OBV',
    'team_season_obv_shot_pg': 'Shot OBV',
    'team_season_obv_defensive_action_pg': 'DA OBV',
    'team_season_obv_dribble_carry_pg': 'DC OBV',
    'team_season_obv_gk_pg': 'GK OBV',
    'team_season_obv_conceded_pg': 'OBV Conceded',
    'team_season_obv_pass_conceded_pg': 'Pass OBV Conceded',
    'team_season_obv_shot_conceded_pg': 'Shot OBV Conceded',
    'team_season_obv_defensive_action_conceded_pg': 'DA Conceded OBV',
    'team_season_obv_dribble_carry_conceded_pg': 'DC Conceded OBV',
    'team_season_obv_gk_conceded_pg': 'GK Conceded OBV',
    'team_season_passes_pg': 'Passes',
    'team_season_successful_passes_pg': 'Successful Passes',
    'team_season_passes_conceded_pg': 'Passes Conceded',
    'team_season_successful_passes_conceded_pg': 'Successful Passes Conceded',
    'team_season_op_passes_pg': 'Open Play Passes',
    'team_season_op_passes_conceded_pg': 'Open Play Passes Conceded',
}



# --- Position -> Profile mapping ---
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

# --- Metric sets for each profile ---
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

NO6_NO8_METRICS = [
    # Defensive / work
    "player_season_padj_tackles_90",
    "player_season_padj_interceptions_90",
    "player_season_ball_recoveries_90",
    "player_season_pressures_90",
    "player_dribbles_stopped_ratio",
    "player_season_aerial_ratio",

    # Passing
    "player_season_pressured_passing_ratio",
    "player_season_forward_pass_proportion",
    "player_season_obv_pass_90",

    # Creation / progression
    "player_season_op_key_passes_90",
    "player_season_npxgxa_90",
    "player_season_xgbuildup_90",
    "player_season_deep_progressions_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_season_carries_90",

    # Output
    "player_season_np_xg_90",
    "player_season_np_shots_90",
    "player_season_shot_on_target_ratio",
    "player_scoring_contribution_90",
    "player_season_obv_defensive_action_90",
]

WINGER_METRICS = [
    "player_season_np_xg_90",
    "player_season_np_shots_90",
    "player_season_shot_on_target_ratio",
    "player_season_op_key_passes_90",
    "player_season_npxgxa_90",
    "player_season_op_passes_into_and_touches_inside_box_90",
    "player_season_dribbles_90",
    "player_successful_dribbles_90",
    "player_season_carries_90",
    "player_successful_crosses_90",
    "player_season_obv_90",
    "player_season_obv_dribble_carry_90",
]

RUNNER_METRICS = [
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
    "player_season_pressures_90",
    "player_season_counterpressures_90",
    "player_season_aggressive_actions_90",
    "player_scoring_contribution_90",
]

# Master mapping from Profile -> metric list
PROFILE_METRICS = {
    "Full Back": WB_METRICS,
    "Outside Centre Back": OUTSIDE_CB_METRICS,
    "Number 8": NO6_NO8_METRICS,
    "Number 10": NO6_NO8_METRICS,
    "Winger": WINGER_METRICS,
    "Runner": RUNNER_METRICS,
}

def get_metrics_for_profile(profile: str) -> list[str]:
    """Return the metric list for a given profile (fallback to a sensible default)."""
    return PROFILE_METRICS.get(profile, NO6_NO8_METRICS)

# -------------------------
# Derived columns for all players
# -------------------------
def add_derived_player_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """
    Adds generic derived columns for all players:
      - player_dribbles_stopped_ratio
      - player_successful_dribbles_90
      - player_successful_crosses_90
      - player_op_xa_90
      - player_scoring_contribution_90
    """
    def safe_div(a, b):
        a = a.astype(float)
        b = b.astype(float)
        return np.where((b == 0) | ~np.isfinite(b), np.nan, a / b)

    def first_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # 1) Dribbles stopped %
    faced_90_col = first_col(players, [
        "player_season_dribbles_faced_90",
        "player_season_dribble_faced_90",
        "dribbles_faced_90",
    ])
    dribbled_past_90_col = "player_season_dribbled_past_90"
    provider_ratio_col = "player_season_dribble_faced_ratio"

    if faced_90_col and dribbled_past_90_col in players.columns:
        denom = players[faced_90_col].fillna(0) + players[dribbled_past_90_col].fillna(0)
        players["player_dribbles_stopped_ratio"] = safe_div(players[faced_90_col].fillna(0), denom)
    elif provider_ratio_col in players.columns:
        players["player_dribbles_stopped_ratio"] = players[provider_ratio_col]
    else:
        players["player_dribbles_stopped_ratio"] = np.nan

    # 2) Successful dribbles per 90
    if {"player_season_dribbles_90","player_season_failed_dribbles_90"} <= set(players.columns):
        players["player_successful_dribbles_90"] = (
            players["player_season_dribbles_90"].fillna(0)
            - players["player_season_failed_dribbles_90"].fillna(0)
        ).clip(lower=0)
    else:
        ratio_col = first_col(players, ["player_season_dribble_ratio", "dribble_success_ratio"])
        if "player_season_dribbles_90" in players.columns and ratio_col:
            players["player_successful_dribbles_90"] = (
                players["player_season_dribbles_90"].fillna(0) * players[ratio_col].fillna(0)
            )
        else:
            players["player_successful_dribbles_90"] = np.nan

    # 3) Successful crosses per 90
    crosses_90_col = first_col(players, ["player_season_crosses_90", "crosses_90"])
    crossing_ratio_col = first_col(players, ["player_season_crossing_ratio", "crossing_success_ratio"])
    if crosses_90_col and crossing_ratio_col:
        players["player_successful_crosses_90"] = players[crosses_90_col].fillna(0) * players[crossing_ratio_col].fillna(0)
    else:
        players["player_successful_crosses_90"] = np.nan

    # 4) OP xA alias
    if "player_season_op_xa_90" in players.columns:
        players["player_op_xa_90"] = players["player_season_op_xa_90"]
    elif "player_season_xa_90" in players.columns:
        players["player_op_xa_90"] = players["player_season_xa_90"]
    else:
        players["player_op_xa_90"] = np.nan

    # 5) Scoring contribution per 90 = NP goals + OP assists
    if {"player_season_npg_90","player_season_op_assists_90"} <= set(players.columns):
        players["player_scoring_contribution_90"] = (
            players["player_season_npg_90"].fillna(0)
            + players["player_season_op_assists_90"].fillna(0)
        )
    else:
        players["player_scoring_contribution_90"] = np.nan

    return players

# ---- Optional aliases to avoid old ImportErrors ----
outside_cb_metrics = OUTSIDE_CB_METRICS
winger_metrics = WINGER_METRICS
no6_no8_metrics = NO6_NO8_METRICS
runner_metrics = RUNNER_METRICS

__all__ = [
    "statbomb_metrics_needed",
    "metrics_mapping",
    "position_mapping",
    "WB_METRICS",
    "OUTSIDE_CB_METRICS",
    "NO6_NO8_METRICS",
    "WINGER_METRICS",
    "RUNNER_METRICS",
    "PROFILE_METRICS",
    "get_metrics_for_profile",
    "add_derived_player_metrics",
    # legacy lowercase aliases:
    "outside_cb_metrics",
    "winger_metrics",
    "no6_no8_metrics",
    "runner_metrics",
]

# --- Detailed metric groups (lenses) ---
# These are not tied to mapping, just groupings for cross-profile analysis

DETAILED_PROFILE_METRICS = {
    # Full Back
    "Wing Back": [
        "player_dribbles_stopped_ratio",         # Dribbles Stopped %
        "player_season_carries_90",              # Carries
        "player_op_xa_90",                       # OP xG Assisted
        "player_successful_dribbles_90",         # Successful Dribbles
        "player_successful_crosses_90",          # Successful Crosses
        "player_season_ball_recoveries_90",      # Ball Recoveries
        "player_season_op_passes_into_box_90",   # OP Passes Into Box
        "player_season_pressured_passing_ratio", # PR Pass %
        "player_season_padj_interceptions_90",   # PADJ Interceptions
        "player_season_aerial_ratio",            # Aerial Win %
    ],

    # Centre Back
    "Outside Centre Back": [
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
        "player_season_dribbles_90",
        "player_season_op_passes_into_box_90",
        "player_season_op_f3_passes_90",
    ],
    "Middle Centre Back": [
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
    ],

    # Centre Midfield
    "Ball Playing 6": [
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
    ],
    "Destroyer 6": [
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
    ],
    "Box to Box 8": [
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
    ],
    "Technical 8": [
        "player_season_np_xg_90",
        "player_season_np_shots_90",
        "player_season_op_xa_90",
        "player_season_op_key_passes_90",
        "player_season_obv_pass_90",
        "player_successful_dribbles_90",
        "player_season_xgbuildup_90",
        "player_season_carries_90",
        "player_season_op_passes_into_and_touches_inside_box_90",
        "player_scoring_contribution_90",
    ],

    # Attacking Mid
    "No10": [
        "player_season_np_shots_90",
        "player_season_np_xg_90",
        "player_scoring_contribution_90",
        "player_season_pressured_passing_ratio",
        "player_season_op_key_passes_90",
        "player_successful_dribbles_90",
        "player_season_op_xa_90",
        "player_season_carries_90",
        "player_season_shot_on_target_ratio",
    ],

    # Winger
    "Inverted Winger": [
        "player_season_np_xg_90",
        "player_season_np_shots_90",
        "player_season_op_key_passes_90",
        "player_season_dribbles_90",
        "player_successful_dribbles_90",
        "player_season_obv_90",
        "player_season_op_xa_90",
        "player_successful_crosses_90",
        "player_season_obv_dribble_carry_90",
    ],
    "Traditional Winger": [
        "player_season_np_xg_90",
        "player_season_np_shots_90",
        "player_season_op_key_passes_90",
        "player_season_dribbles_90",
        "player_successful_dribbles_90",
        "player_season_carries_90",
        "player_successful_crosses_90",
        "player_season_op_xa_90",
        "player_season_obv_dribble_carry_90",
    ],

    # Centre Forward
    "Target Man": [
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
    ],
    "Runner": [
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
    ],
}
