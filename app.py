import streamlit as st
import requests
import subprocess
import threading
import time
from io import BytesIO
import zipfile
import pandas as pd

st.set_page_config(page_title="AI Dataset Verifier", layout="centered")
st.title("AI Dataset Verifier")
st.caption("Upload CSV/ZIP + description → Get quality scores")

# Start FastAPI in background
def run_fastapi():
    time.sleep(2)  # Wait for deps
    subprocess.Popen(["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
threading.Thread(target=run_fastapi, daemon=True).start()

# UI
def verify_dataset(uploaded_file, description):
    if not uploaded_file:
        return "No file uploaded", None
    files = {"file": uploaded_file}
    data = {"description": description or ""}
    try:
        r = requests.post("http://127.0.0.1:8000/verify", files=files, data=data, timeout=30)
        if r.status_code == 200:
            res = r.json()
            st.success(f"**Status:** {res['status']}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Quality", res['scores']['quality'])
            col2.metric("Completeness", res['scores']['completeness'])
            col3.metric("Consistency", res['scores']['consistency'])
            col4.metric("Relevance", res['scores']['relevance'])
            return res['issues']
        else:
            st.error(f"Error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Failed: {e}")

uploaded_file = st.file_uploader("Upload Dataset (CSV/ZIP)", type=["csv", "zip", "txt"])
description = st.text_input("Description", placeholder="e.g. advertising and sales dataset")

if st.button("Verify") and uploaded_file:
    verify_dataset(uploaded_file, description)
