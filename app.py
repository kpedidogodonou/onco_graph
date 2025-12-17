import streamlit as st

# Page Title
st.set_page_config(page_title="Onco Graph", page_icon="🕸️")

# Chat Page
chat_page = st.Page(
    "chat.py",
    title="Onco-Graph Chat",
    icon=":material/smart_toy:"
)

# Data Explorer Page
visualize_table_page = st.Page(
    "visualize_table.py",
    title="Onco Graph Data Explorer",
    icon=":material/table_chart:"
)

# Setup page navigation 
pg = st.navigation(
    {
        "Pages": [chat_page, visualize_table_page],
    }
)

# Run the app
pg.run()