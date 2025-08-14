import streamlit as st
from Tabs import  landingpage, metric
# streamlit_app.py

def add_bg_from_url():
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("https://i.ibb.co/QCJ6zT6/bg.png");
                background-attachment: fixed;
                background-size: cover
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Page title
st.set_page_config(page_title="Satellite Crop Monitoring Dashboard",  page_icon="🌾")
st.title("🌱 Crop Health Dashboard")


if True:
    tabs_names = ["Home", "NDVI", "NDWI", "EVI"]
    tabs = st.tabs(tabs_names)

    with tabs[0]:
        landingpage.app()

    for i in range(1, 4):
        with tabs[i]:
            metric.app(tabs_names[i])
    