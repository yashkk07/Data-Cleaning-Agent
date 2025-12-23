import streamlit as st
import os

from etl.pipeline import run_pipeline

st.set_page_config(page_title="Agentic Data Cleaner", layout="centered")
st.title("🧠 Agentic Data Cleaning System")

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    output_path = os.path.join(OUTPUT_DIR, f"cleaned_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded")

    if st.button("Run Agentic Cleaning"):
        with st.spinner("Agent is reasoning and cleaning..."):
            try:
                result = run_pipeline(input_path, output_path)

                st.success("Cleaning completed")

                st.subheader("🧠 Agent Execution History")
                for step in result["history"]:
                    st.json(step)

                with open(output_path, "rb") as f:
                    st.download_button(
                        "Download Cleaned CSV",
                        f,
                        file_name=os.path.basename(output_path),
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(str(e))
