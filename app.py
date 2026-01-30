"""
Streamlit Frontend - SIMPLE VERSION
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64

API_BASE_URL = "http://localhost:7860"


st.set_page_config(page_title="Smart Parking", page_icon="🚗", layout="wide")

def base64_to_image(base64_string: str):
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))

def check_api():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = {
        'detected_available': 0,
        'total_spots': 0,
        'reserved_spots': 0,
        'final_available': 0,
        'last_image': None
    }

st.title("🚗 Smart Parking System")
st.markdown("---")

api_status = check_api()

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline")
        st.stop()
    
    st.markdown("---")
    sample_rate = st.slider("Sample Rate", 10, 90, 30, 10)
    
    st.markdown("---")
    st.subheader("🎫 Reservations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("➕ Reserve", use_container_width=True, type="primary"):
            try:
                response = requests.post(f"{API_BASE_URL}/reserve")
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.data['detected_available'] = result['detected_available']
                    st.session_state.data['reserved_spots'] = result['reserved_spots']
                    st.session_state.data['final_available'] = result['final_available']
                    
                    if result['success']:
                        st.success(result['message'])
                    else:
                        st.error(result['message'])
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        if st.button("➖ Cancel", use_container_width=True):
            try:
                response = requests.post(f"{API_BASE_URL}/cancel")
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.data['detected_available'] = result['detected_available']
                    st.session_state.data['reserved_spots'] = result['reserved_spots']
                    st.session_state.data['final_available'] = result['final_available']
                    
                    if result['success']:
                        st.success(result['message'])
                    else:
                        st.warning(result['message'])
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.metric("Reserved", st.session_state.data['reserved_spots'])

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total", st.session_state.data['total_spots'])
with col2:
    st.metric("Detected", st.session_state.data['detected_available'])
with col3:
    st.metric("Reserved", st.session_state.data['reserved_spots'])
with col4:
    st.metric("🟢 Final", st.session_state.data['final_available'])

st.markdown("---")

# Upload
uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded:
    if st.button("🚀 Process", type="primary"):
        with st.spinner("Processing..."):
            try:
                uploaded.seek(0)
                files = {'file': uploaded}
                params = {'sample_rate': sample_rate}
                
                response = requests.post(
                    f"{API_BASE_URL}/detect/video",
                    files=files,
                    params=params,
                    timeout=300
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.session_state.data['detected_available'] = result['detected_available']
                    st.session_state.data['total_spots'] = result['total_spots']
                    st.session_state.data['reserved_spots'] = result['reserved_spots']
                    st.session_state.data['final_available'] = result['final_available']
                    st.session_state.data['last_image'] = result['annotated_image']
                    
                    st.success(f"✅ Processed {result['frames_processed']} frames!")
                    st.success(f"📊 Detected: {result['detected_available']}/{result['total_spots']}")
                    st.info(f"🎫 Reserved: {result['reserved_spots']} | Final: {result['final_available']}")
                    
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# Display
if st.session_state.data['last_image']:
    st.markdown("---")
    st.image(base64_to_image(st.session_state.data['last_image']), use_container_width=True)

st.caption("Smart Parking v4.0")



import threading
import uvicorn
from api import app as fastapi_app

def run_api():
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=7860,
        log_level="error"
    )

threading.Thread(target=run_api, daemon=True).start()