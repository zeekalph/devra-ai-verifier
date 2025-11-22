import streamlit as st
import requests
from io import BytesIO

st.set_page_config(page_title="AI Dataset Verifier", layout="centered")
st.title("AI Dataset Verifier")
st.caption("Upload CSV/ZIP + description → Get quality scores")

# Start FastAPI in background
import subprocess
import threading
import time
import logging
def run_fastapi():
    time.sleep(2)  
    try:
        proc = subprocess.Popen(
            ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logger.info("FastAPI started on http://127.0.0.1:8000")
        proc.wait()  # Keep thread alive
    except Exception as e:
        logger.error(f"Failed to start FastAPI: {e}")

threading.Thread(target=run_fastapi, daemon=True).start()
# UI
uploaded_file = st.file_uploader("Upload Dataset (CSV/ZIP)", type=["csv", "zip", "txt"])
description = st.text_input("Description", placeholder="e.g. advertising and sales dataset")

if st.button("Verify") and uploaded_file:
    with st.spinner("Verifying..."):
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
                if res['issues']:
                    st.warning("Issues: " + "; ".join(res['issues']))
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Failed: {e}")

st.info("Backend: FastAPI with TinyBERT + ResNet18 (150 MiB total)")
