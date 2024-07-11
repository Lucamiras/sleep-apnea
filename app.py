import streamlit as st


# rest of the code
pages = {
    "Apnea Detection": [
        st.Page("app/src/pages/app_record.py", title="Live analysis of sound", icon=":material/mic:"),
        st.Page("app/src/pages/app_analyze.py", title="Analyze a recording", icon=":material/monitoring:"),
    ],
    "More info": [
        st.Page("app/src/pages/about_apnea.py", title="About sleep apnea", icon=":material/menu_book:")
    ]
}

st.title('Deep Sleep Apnea Classifier')
pg = st.navigation(pages)
pg.run()
