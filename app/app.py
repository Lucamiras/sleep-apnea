import streamlit as st

st.set_page_config(
    page_title="Deep Sleep Apnea Detector"
)

pages = {
    "Apnea Detection": [
        st.Page("pages/app_record.py", title="Live analysis of sound", icon=":material/mic:"),
        st.Page("pages/app_analyze.py", title="Analyze a recording", icon=":material/monitoring:"),
    ],
    "More info": [
        st.Page("pages/about_apnea.py", title="About sleep apnea", icon=":material/menu_book:")
    ]
}

st.title('Deep Sleep Apnea Classifier')
pg = st.navigation(pages)
pg.run()
