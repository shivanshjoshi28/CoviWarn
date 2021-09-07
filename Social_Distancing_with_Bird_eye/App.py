# import home page and help
import start
import helpstart
import streamlit as st

PAGES = {
    "Detector App": start,
    "How to Use": helpstart
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()