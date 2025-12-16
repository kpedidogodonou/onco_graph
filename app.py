import streamlit as st

st.set_page_config(page_title="Onco Graph", page_icon="🕸️")

visualize_table_page = st.Page(
    "visualize_table.py",
    title="Onco Graph Data Explorer",
    icon=":material/table_chart:"
)

chat_page = st.Page(
    "chat.py",
    title="Onco-Graph Chat",
    icon=":material/smart_toy:"
)

pg = st.navigation(
    {
        "Pages": [chat_page, visualize_table_page],
    }
)

pg.run()