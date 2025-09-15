import pandas as pd
import numpy as np
from statsbombpy import sb
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bristol Rovers Player Dashboard", layout="wide")

# --- Position profiles
position_profiles = {
    "Wing Back": [
        "Dribbles Stopped %","Carries","OP XG ASSISTED","Successful Dribbles",
        "Successful Crosses","Ball Recoveries","OP Passes Into Box",
        "PR. Pass %","PADJ Interceptions","Aerial Win %"
    ],
    "Middle Centre Back": [
        "Aerial Win %","Dribbles Stopped %","PADJ Tackles","PADJ Interceptions",
        "PADJ Clearances","Pass Forward %","Defensive Regains","DA OBV",
        "Ball Recoveries","PR. Pass %"
    ],
    "Centre Back Outside": [
        "Aerial Win %","Dribbles Stopped %","PADJ Tackles","PADJ Interceptions",
        "PR. Pass %","Carries","Successful Crosses","Dribbles",
        "OP Passes Into Box","OP F3 Passes"
    ]
}

# --- Raw position mapping
raw_position_mapping = {
    'Left back': 'Wing Back',
    'Left wing back': 'Wing Back',
    'Right back': 'Wing Back',
    'Right wing back': 'Wing Back',
    'Right centre back': 'Middle Centre Back',
    'Left centre back': 'Middle Centre Back',
    'Centre back': 'Middle Centre Back'
}

# --- Metric rename mapping (short version)
metrics_mapping = {
    "player_season_dribble_faced_ratio": "Dribbles Stopped %",
    "player_season_carries_90": "Carries",
    "player_season_op_xa_90": "OP XG ASSISTED",
    "player_season_dribble_ratio": "Successful Dribbles",
    "player_season_crossing_ratio": "Successful Crosses",
    "player_season_ball_recoveries_90": "Ball Recoveries",
    "player_season_op_passes_into_box_90": "OP Passes Into Box",
    "player_season_pressured_passing_ratio": "PR. Pass %",
    "player_season_padj_interceptions_90": "PADJ Interceptions",
    "player_season_aerial_ratio": "Aerial Win %",
    "player_season_padj_tackles_90": "PADJ Tackles",
    "player_season_padj_clearances_90": "PADJ Clearances",
    "player_season_defensive_action_regains_90": "Defensive Regains",
    "player_season_obv_defensive_action_90": "DA OBV",
    "player_season_forward_pass_ratio": "Pass Forward %",
    "player_season_op_f3_passes_90": "OP F3 Passes",
    "player_season_dribbles_90": "Dribbles"
}
inverse_metrics_mapping = {v: k for k, v in metrics_mapping.items()}

# --- Load credentials
user = st.secrets["statsbomb"]["user"]
passwd = st.secrets["statsbomb"]["passwd"]
creds = {"user": user, "passwd": passwd}

@st.cache_data(show_spinner=True)
def load_data():
    all_comps = sb.competitions(creds=creds)
    dfs = []
    for _, row in all_comps.iterrows():
        comp_id, season_id = row["competition_id"], row["season_id"]
        df = sb.player_season_stats(comp_id, season_id, creds=creds)
        br = df[df['team_name'] == 'Bristol Rovers']
        dfs.append(br)
    df = pd.concat(dfs)
    df['profile'] = df['primary_position'].map(raw_position_mapping)
    return df

st.title("Bristol Rovers Player Dashboard")

with st.spinner("Loading StatsBomb data..."):
    df = load_data()

df = df[df['player_season_minutes'] >= 300]  # filter minutes

player = st.selectbox("Select Player", df['player_name'].unique())

player_df = df[df['player_name'] == player].copy()
profile = player_df['profile'].iloc[0]

st.write(f"**Profile:** {profile}")

metrics = position_profiles.get(profile, [])

subset = df[df['profile'] == profile]

# --- Compute percentiles
percentiles = {}
for m in metrics:
    raw_col = inverse_metrics_mapping.get(m)
    if raw_col not in subset.columns:
        continue
    values = subset[raw_col]
    player_val = player_df[raw_col].iloc[0]
    percentiles[m] = stats.percentileofscore(values, player_val)

percentiles_df = pd.DataFrame(percentiles, index=[0]).T
percentiles_df.columns = ['Percentile']
percentiles_df['Metric'] = percentiles_df.index

# --- Top/Bottom 3
top3 = percentiles_df.sort_values('Percentile', ascending=False).head(3)
bottom3 = percentiles_df.sort_values('Percentile', ascending=True).head(3)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Strengths")
    for i, row in top3.iterrows():
        st.write(f"- {row['Metric']} ({row['Percentile']:.1f}%)")
with col2:
    st.subheader("Areas for Improvement")
    for i, row in bottom3.iterrows():
        st.write(f"- {row['Metric']} ({row['Percentile']:.1f}%)")

# --- Chart
fig, ax = plt.subplots(figsize=(6,6))
colors = percentiles_df['Percentile'].apply(lambda x: 'red' if x<50 else ('orange' if x<70 else 'green'))
ax.barh(percentiles_df['Metric'], percentiles_df['Percentile'], color=colors)
ax.set_xlim(0,100)
ax.set_xlabel('Percentile')
st.pyplot(fig)

# --- Placeholder for player image and badge
st.image("https://via.placeholder.com/150x150.png?text=Player+Image", caption="Player Image Placeholder")
st.image("https://via.placeholder.com/100x100.png?text=Club+Badge", caption="Club Badge Placeholder")

# --- Placeholder table for opposition and minutes
st.subheader("Opposition & Minutes Played")
sample_oppo = pd.DataFrame({
    "Opponent": ["Team A","Team B","Team C"],
    "Minutes": [90,75,60],
    "Logo": ["https://via.placeholder.com/30","https://via.placeholder.com/30","https://via.placeholder.com/30"]
})
st.dataframe(sample_oppo)

