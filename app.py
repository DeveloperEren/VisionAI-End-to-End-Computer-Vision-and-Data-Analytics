import streamlit as st
import requests
from PIL import Image
import io
import pyodbc
import pandas as pd
import time

# Page Configuration
st.set_page_config(page_title="VisionAI | End-to-End Computer Vision and Data Analytics",  layout="wide")

if 'welcome_shown' not in st.session_state:
    st.toast("System initializing... Neural network weights loaded... Success! ", icon="🟢")
    st.session_state.welcome_shown = True

# Cyberpunk / High-Tech Themed CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at 10% 20%, rgba(0, 255, 204, 0.05) 0%, transparent 20%),
                          radial-gradient(circle at 90% 80%, rgba(0, 204, 255, 0.05) 0%, transparent 20%);
    }
    h1, h2, h3 { color: #00FFCC !important; font-family: 'Consolas', monospace; text-transform: uppercase; letter-spacing: 1px;}
    .stButton>button { background-color: transparent; color: #00FFCC; border: 1px solid #00FFCC; font-weight: bold; border-radius: 5px; transition: 0.3s; }
    .stButton>button:hover { background-color: #00FFCC; color: #0b0f19; box-shadow: 0 0 10px #00FFCC; transform: scale(1.02); }
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background-color: rgba(11, 15, 25, 0.95); color: #00FFCC; text-align: center; padding: 10px; font-size: 13px; border-top: 1px solid #00FFCC; z-index: 100; backdrop-filter: blur(5px);}
    .glass-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(0, 255, 204, 0.3); padding: 25px; border-radius: 10px; border-left: 4px solid #00FFCC; backdrop-filter: blur(10px); }
    .skill-badge { background-color: rgba(0, 255, 204, 0.1); color: #00FFCC; padding: 5px 12px; border-radius: 15px; font-size: 12px; margin-right: 8px; margin-bottom: 8px; display: inline-block; border: 1px solid #00FFCC;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1> VisionAI | End-to-End Computer Vision and Data Analytics</h1>", unsafe_allow_html=True)
st.write("Advanced object detection, real-time database synchronization, and end-to-end model management interface.")
st.markdown("---")

# Left Sidebar (Advanced Control Panel)
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.2rem;'> Operation Mode</h2>", unsafe_allow_html=True)
    
    work_mode = st.radio("Select Data Source", ["Static Visual Analysis", "Video Processing (Beta)", "Live RTSP Stream"], index=0)
    st.markdown("---")
    
    if work_mode == "Static Visual Analysis":
        uploaded_file = st.file_uploader("Upload Image to System", type=["jpg", "jpeg", "png"])
    elif "Video" in work_mode:
        st.info("Video analytics module will be activated in the next update.")
        st.file_uploader("Video File (MP4, AVI)", type=["mp4", "avi"], disabled=True)
        uploaded_file = None
    else:
        st.warning("Configuring RTSP Camera connection interface...")
        st.text_input("Camera IP / RTSP URL", placeholder="rtsp://admin:12345@...", disabled=True)
        uploaded_file = None

    st.markdown("---")
    
    st.markdown("<h3 style='font-size: 1.1rem;'> Live Model Settings</h3>", unsafe_allow_html=True)
    
    user_conf = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.50, step=0.05, 
                          help="Lowering the score detects more objects but increases the margin of error.")
    
    user_iou = st.slider("IOU Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05,
                         help="Filtering tolerance for overlapping bounding boxes.")
    
    AVAILABLE_CLASSES = ["car", "person", "bus", "truck", "motorcycle", "bicycle"]
    selected_classes = st.multiselect("Filter Target Classes", AVAILABLE_CLASSES, default=AVAILABLE_CLASSES,
                                      help="Select only the object types you want to detect.")

    st.markdown("---")
    st.markdown("""
        <div style='font-size: 0.85rem; color: #aaa; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;'>
        <b>System Status:</b><br>
        <span style='color: #00FFCC;'>●</span> AI Engine: Active<br>
        <span style='color: #00FFCC;'>●</span> API (FastAPI): Running<br>
        <span style='color: #00FFCC;'>●</span> DB (SQL Server): Synchronized
        </div>
    """, unsafe_allow_html=True)

# Main Tabs
tab1, tab2, tab3 = st.tabs([" Inference Center", " SQL Data Warehouse", "System Architect"])

# --- TAB 1: INFERENCE CENTER ---
with tab1:
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        
        with col1:
            st.image(image, caption="Source Image (Unprocessed)", use_container_width=True)

        if st.button(" Trigger Deep Learning Network", use_container_width=True):
            with st.spinner(f"Processing layers (Conf: {int(user_conf*100)}%, IOU: {user_iou})..."):
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_bytes = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                classes_str = ",".join(selected_classes)
                data_payload = {"conf": user_conf, "iou": user_iou, "classes": classes_str}
                
                try:
                    start_time = time.time()
                    response = requests.post("http://127.0.0.1:8000/detect/", files=files, data=data_payload)
                    process_time = round((time.time() - start_time) * 1000)
                    
                    if response.status_code == 200:
                        data = response.json()
                        download_url = data.get("download_url")
                        img_res = requests.get(f"http://127.0.0.1:8000{download_url}")
                        
                        with col2:
                            m1, m2, m3 = st.columns(3)
                            m1.metric(label="Objects Detected", value=data['total_detections'])
                            m2.metric(label="Processing Time", value=f"{process_time} ms")
                            m3.metric(label="Database", value="Written")
                            
                            if img_res.status_code == 200:
                                processed_image = Image.open(io.BytesIO(img_res.content))
                                st.image(processed_image, caption="YOLOv8 Network Output", use_container_width=True)
                                
                                st.download_button(
                                    label=" Download Processed Data",
                                    data=img_res.content,
                                    file_name=f"visionai_{uploaded_file.name}",
                                    mime=uploaded_file.type,
                                    use_container_width=True
                                )
                    else:
                        st.error("API Connection Error: Server did not respond.")
                except Exception as e:
                    st.error(f"System Error: {e} - make sure main.py is running.")
    else:
        st.info("System ready. Upload an image from the left panel to start the analysis process.")

# --- TAB 2: DATABASE ANALYTICS ---
with tab2:
    col_btn, col_space = st.columns([1, 4])
    with col_btn:
        refresh = st.button(" Live Sync Database")
        
    if refresh:
        DB_CONFIG = r"DRIVER={ODBC Driver 17 for SQL Server};SERVER=ERENZEPHYRUS\MSSQLSERVER1;DATABASE=SecurityCamera;Trusted_Connection=yes;"
        try:
            conn = pyodbc.connect(DB_CONFIG)
            df = pd.read_sql_query("SELECT TOP 100 * FROM VehicleDetections ORDER BY Id DESC", conn)
            conn.close()

            if not df.empty:
                st.subheader("SQL Server Terminal Feed")
                st.dataframe(df, use_container_width=True)
                st.subheader("Class Distribution Statistics")
                st.bar_chart(df['VehicleType'].value_counts())
            else:
                st.warning("No records found in the database yet.")
        except Exception as e:
            st.error(f"SQL Server Connection Refused: {e}")

# --- TAB 3: SYSTEM ARCHITECT ---
with tab3:
    st.markdown("""<div class="glass-card">
    <h2 style='margin-top:0;'>Mustafa Eren Güler</h2>
    <p style='color:#bbb; font-size: 16px;'><b>AI Developer & Computer Vision</b></p>
    <div style="margin: 15px 0;">
        <span class="skill-badge">Python</span>
        <span class="skill-badge">YOLOv8</span>
        <span class="skill-badge">FastAPI</span>
        <span class="skill-badge">T-SQL / SQL Server</span>
        <span class="skill-badge">Docker</span>
        <span class="skill-badge">Artificial Intelligence</span>
    </div>
    <p style='line-height: 1.6;'>
        I am a graduate of Electrical-Electronics Engineering and Computer Programming from Karamanoğlu Mehmetbey University.
        This system is a prototype demonstrating how AI models can evolve from a simple Python script into an <b>end-to-end industrial architecture communicating via API and integrated with a relational database (SQL)</b>.
    </p>
    <p style='line-height: 1.6;'>
        My academic goal is to complete my master's degree in Electrical and Electronics engineering with a focus on artificial intelligence, designing autonomous systems and advanced data warehouse architectures.
    </p>
    <br>
    <p style='color: #00FFCC; font-style: italic;'><b>"dev.eren"</b></p>
</div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <b>ai.eren</b> | Core Vision AI Engine © 2026
    </div>
""", unsafe_allow_html=True)