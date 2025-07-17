# app.py
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Page Configuration ---
# It's good practice to set page config in the main app.py if you want it to be global,
# or in each page's script if you want page-specific configs.
# For this multi-page app, st.set_page_config() should be called only once,
# and ideally in app.py before st.navigation.
# However, st.Page might handle this; let's assume individual page configs or a global one.
# For now, we'll let individual pages handle their specific configs like layout.
st.set_page_config(
    page_title="NLP Tools",
    layout="wide"
) # Global config

# --- Define Pages ---
# Ensure the paths to your page files are correct.
# For example, if 'main_viewer.py' is your original script.
# page_viewer = st.Page(
#     "main_viewer.py",
#     title="Data Viewer",
#     icon="📊",
#     # default=(st.query_params().get("page", [""])[0] != "input") # Set as default unless 'input' page is queried
# )

page_input = st.Page(
    "sentence_entry_page.py",
    title="WhyFix",
    icon="✏️",
    # default=(st.query_params().get("page", [""])[0] == "input")
)

# --- Set up Navigation ---
# You can organize pages into sections if needed, e.g.,
# pg = st.navigation({"Home": [page_viewer], "Real-Time system": [page_input]})
# pg = st.navigation({"Home": [page_viewer]})
# pg = st.navigation([page_input, page_viewer])
pg = st.navigation([page_input])
# pg = st.navigation([page_viewer])

# --- Run the App ---
pg.run()
