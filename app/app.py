import streamlit as st

# Page setup
analyze_page = st.Page(
    page="views/analyze.py",
    title="Analyze your sleep",
    icon=":material/monitoring:",
    default=True
)
record_page = st.Page(
    page="views/record.py",
    title="Record your sleep",
    icon=":material/mic:",
)
about__project_page = st.Page(
    page="views/about_project.py",
    title="About this project",
    icon=":material/help:",
)
about_apnea_page = st.Page(
    page="views/about_apnea.py",
    title="About apnea",
    icon=":material/single_bed:",
)

pg = st.navigation(
    pages={
        "Analyze sleep": [analyze_page, record_page],
        "Learn more": [about__project_page, about_apnea_page],
        })

#st.logo("🛌")
st.title('Deep Sleep')
st.sidebar.text("Made with <3 by Lucas.")

pg.run()