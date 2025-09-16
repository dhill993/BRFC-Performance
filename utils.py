# utils.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from google.oauth2 import service_account as gsa
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
load_dotenv()  # saf

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"

# ---------- Local Drive helpers ----------

def _candidate_local_roots():
    root = os.getenv("GOOGLE_DRIVE_DESKTOP_ROOT")
    if root:
        yield Path(root)

    # Windows common
    yield Path("G:/My Drive")
    yield Path("g:/My Drive")
    yield Path("G:/Google Drive")
    yield Path("g:/Google Drive")

    # Git Bash style
    yield Path("/g/My Drive")
    yield Path("/g/Google Drive")

    # WSL style
    yield Path("/mnt/g/My Drive")
    yield Path("/mnt/g/Google Drive")

    # Home-based fallback
    yield Path.home() / "Google Drive" / "My Drive"


def find_local_drive_root() -> Path | None:
    for p in _candidate_local_roots():
        if p.exists():
            return p
    return None


def read_local_latest(drive_root: Path):
    candidates = [
        drive_root / "df_competitions_last_season.csv",
        drive_root / "df_competitions_last_season.tsv",
        drive_root / "competitions_latest.csv",
        drive_root / "competitions_latest.tsv",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".csv":
                return pd.read_csv(p), p
            if p.suffix.lower() == ".tsv":
                return pd.read_csv(p, sep="\t"), p
    return None, None

# ---------- OAuth helpers (used for local dev) ----------

def _is_local_run() -> bool:
    try:
        return st.runtime.exists() and st.runtime.scriptrunner.script_run_context.is_local
    except Exception:
        return False


def _get_redirect_uri() -> str:
    # [web].redirect_uris must be [local, cloud]
    uris = [u.rstrip("/") for u in list(st.secrets.get("web", {}).get("redirect_uris", []))]
    if len(uris) >= 2:
        return uris[0] if _is_local_run() else uris[1]
    return "http://localhost:8501"  # safe fallback for local


def _build_web_flow(redirect_uri: str) -> Flow:
    client_config = {"web": dict(st.secrets["web"])}  # must exist if you use OAuth
    return Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)


def _installed_app_login() -> Credentials:
    client_config = {"installed": {
        "client_id": st.secrets["web"]["client_id"],
        "project_id": st.secrets["web"]["project_id"],
        "auth_uri": st.secrets["web"]["auth_uri"],
        "token_uri": st.secrets["web"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["web"]["auth_provider_x509_cert_url"],
        "client_secret": st.secrets["web"]["client_secret"],
        "redirect_uris": ["http://localhost"]
    }}
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    return flow.run_local_server(port=0)


def get_credentials() -> Credentials | None:
    # restore from session
    if "oauth_token" in st.session_state:
        data = st.session_state["oauth_token"]
        creds = Credentials(**data)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            st.session_state["oauth_token"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes,
            }
        return creds

    # local dev helper
    if _is_local_run() and "web" in st.secrets:
        creds = _installed_app_login()
        st.session_state["oauth_token"] = {
            "token": creds.token,
            "refresh_token": creds.refresh_token,
            "token_uri": creds.token_uri,
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "scopes": creds.scopes,
        }
        return creds

    return None


def start_oauth_web_flow() -> str:
    redirect_uri = _get_redirect_uri()
    flow = _build_web_flow(redirect_uri)
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    st.session_state["oauth_state"] = state
    return auth_url


def finish_oauth_web_flow(auth_code: str):
    redirect_uri = _get_redirect_uri()
    flow = _build_web_flow(redirect_uri)
    flow.fetch_token(code=auth_code)
    creds = flow.credentials
    st.session_state["oauth_token"] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

# ---------- Credentials selection ----------

def ensure_drive_creds():
    # prefer service account on cloud
    if "service_account" in st.secrets:
        info = dict(st.secrets["service_account"])
        # convert escaped newlines if user pasted with \n
        pk = info.get("private_key", "")
        if "\\n" in pk and "\n" not in pk:
            info["private_key"] = pk.replace("\\n", "\n")

        # quick sanity check to help catch bad pastes
        if not info["private_key"].startswith("-----BEGIN PRIVATE KEY-----"):
            raise ValueError("service_account.private_key does not start with BEGIN PRIVATE KEY")
        if not info["private_key"].strip().endswith("-----END PRIVATE KEY-----"):
            raise ValueError("service_account.private_key does not end with END PRIVATE KEY")

        return gsa.Credentials.from_service_account_info(info, scopes=SCOPES)

    # fallback for local
    return get_credentials()

# Cache the Drive service for reuse
@st.cache_resource(show_spinner=False)
def get_drive_service():
    creds = ensure_drive_creds()
    if not creds:
        return None
    return build("drive", "v3", credentials=creds)

SERVICE = get_drive_service()

# ---------- Drive search and download helpers ----------

def _drive_corpora_kwargs():
    drv = st.secrets.get("drive", {})
    kwargs = dict(supportsAllDrives=True, includeItemsFromAllDrives=True)
    if drv.get("drive_id"):
        kwargs["corpora"] = "drive"
        kwargs["driveId"] = drv["drive_id"]
    else:
        kwargs["corpora"] = "allDrives"
    return kwargs


def drive_list(service, q: str):
    params = {
        "q": q,
        "fields": "files(id,name,modifiedTime,mimeType,parents)",
        "pageSize": 100,
        **_drive_corpora_kwargs(),
    }
    return service.files().list(**params).execute().get("files", [])


def drive_find_child_folder_id(service, parent_id: str, name: str) -> str | None:
    # exact name first
    q = (
        f"trashed = false and mimeType = '{FOLDER_MIME}' "
        f"and name = '{name}' and '{parent_id}' in parents"
    )
    hits = drive_list(service, q)
    if hits:
        return hits[0]["id"]
    # loose match fallback
    q = (
        f"trashed = false and mimeType = '{FOLDER_MIME}' "
        f"and name contains '{name}' and '{parent_id}' in parents"
    )
    hits = drive_list(service, q)
    return hits[0]["id"] if hits else None


def drive_resolve_path(service, top_folder_id: str, *names: str) -> str | None:
    cur = top_folder_id
    for name in names:
        nxt = drive_find_child_folder_id(service, cur, name)
        if not nxt:
            return None
        cur = nxt
    return cur


def list_files_from_drive(service, name_contains="competitions_latest", parent_id: str | None = None):
    drv = st.secrets.get("drive", {})
    folder_id = parent_id or drv.get("folder_id")

    q = f"name contains '{name_contains}' and trashed = false"
    if folder_id:
        q += f" and '{folder_id}' in parents"

    params = {
        "q": q,
        "fields": "files(id,name,size,modifiedTime,parents,mimeType)",
        "pageSize": 100,
        **_drive_corpora_kwargs(),
    }
    return service.files().list(**params).execute().get("files", [])


def download_file(service, file_id, output_path: Path):
    request = service.files().get_media(fileId=file_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                st.write(f"Download {output_path.name} {int(status.progress() * 100)}%")
    st.success(f"{output_path.name} downloaded")

# ---------- Optional: user OAuth entry point for pages ----------
# You do not need to call this if you are using a service account.

def ensure_signed_in():
    try:
        if "code" in st.query_params:
            finish_oauth_web_flow(st.query_params["code"])
            try:
                st.query_params.clear()
            except Exception:
                pass
    except Exception:
        # older Streamlit versions do not have st.query_params
        pass

    creds = get_credentials()
    if not creds:
        url = start_oauth_web_flow()
        st.link_button("Sign in with Google", url)
        st.stop()
    return creds

# --- NEW: folder-scoped Drive helpers ---------------------------------------
def drive_find_child_folder_id(service, parent_id: str, child_name: str) -> str | None:
    """Return folder id of child_name directly under parent_id."""
    drive_cfg = st.secrets.get("drive", {})
    drive_id  = drive_cfg.get("drive_id")

    q = (
        f"name = '{child_name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false and '{parent_id}' in parents"
    )
    kwargs = dict(
        q=q,
        fields="files(id,name)",
        pageSize=100,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    kwargs["corpora"] = "drive" if drive_id else "allDrives"
    if drive_id:
        kwargs["driveId"] = drive_id

    try:
        res = SERVICE.files().list(**kwargs).execute()
        items = res.get("files", [])
        return items[0]["id"] if items else None
    except Exception:
        return None


def drive_list_subfolders(service, parent_id: str):
    """List immediate subfolders under a folder id."""
    drive_cfg = st.secrets.get("drive", {})
    drive_id  = drive_cfg.get("drive_id")

    q = f"mimeType = 'application/vnd.google-apps.folder' and trashed = false and '{parent_id}' in parents"
    kwargs = dict(
        q=q,
        fields="files(id,name,modifiedTime)",
        pageSize=200,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    kwargs["corpora"] = "drive" if drive_id else "allDrives"
    if drive_id:
        kwargs["driveId"] = drive_id

    try:
        res = service.files().list(**kwargs).execute()
        return res.get("files", [])
    except Exception:
        return []


def drive_list_latest_in_folder(service, folder_id: str, name_contains: str):
    """Return newest files in a folder that contain `name_contains`."""
    drive_cfg = st.secrets.get("drive", {})
    drive_id  = drive_cfg.get("drive_id")

    q = (
        f"name contains '{name_contains}' and trashed = false "
        f"and '{folder_id}' in parents"
    )
    kwargs = dict(
        q=q,
        fields="files(id,name,modifiedTime)",
        pageSize=50,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    )
    kwargs["corpora"] = "drive" if drive_id else "allDrives"
    if drive_id:
        kwargs["driveId"] = drive_id

    try:
        res = service.files().list(**kwargs).execute()
        files = res.get("files", [])
        files.sort(key=lambda x: x.get("modifiedTime", ""), reverse=True)
        return files
    except Exception:
        return []

# utils.py
import os
import streamlit as st

def get_statsbomb_creds():
    """
    Return (user, password) from environment first, then Streamlit secrets.
    """
    user = os.getenv("SB_USERNAME")
    pwd  = os.getenv("SB_PASSWORD")

    if not user or not pwd:
        sb = st.secrets.get("statsbomb", {})
        user = user or sb.get("user")
        pwd  = pwd  or sb.get("password")

    return user, pwd
