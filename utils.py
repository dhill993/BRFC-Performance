# utils.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_statsbomb_creds():
    """
    Return (username, password) for StatsBomb.
    Priority:
    1. Environment variables
    2. Streamlit secrets
    """
    user = os.getenv("SB_USERNAME")
    pwd  = os.getenv("SB_PASSWORD")

    if not user or not pwd:
        sb = st.secrets.get("statsbomb", {})
        user = user or sb.get("user")
        pwd  = pwd  or sb.get("password")

    return user, pwd
