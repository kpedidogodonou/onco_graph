import streamlit as st
import pandas as pd
from htbuilder import div, styles
from htbuilder.units import rem
st.set_page_config(page_title="Onco Graph Data Explorer", layout="wide")
st.html(div(style=styles(font_size=rem(5), line_height=1))["📑"])

@st.cache_data
def load_data():
    df = pd.read_csv("./data/processed/completed_clinical_df.csv")
    # age appears to be in days -> convert to years for filtering
    df["age_years"] = (df["diagnoses.age_at_diagnosis"] / 365.25).round(1)
    return df

df = load_data()

st.title("Onco Graph Data Explorer")

st.info(
    'Use this page to explore the dataset that powered the Knowledge Graph used by the LLM. There is a filter in the left panel (hided) that you can used to filter the relevant data'
    '[GDC Data Portal](https://portal.gdc.cancer.gov/)',
    icon="ℹ️"
)
st.caption(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

# -------- Sidebar filters --------
with st.sidebar:
    st.header("Filters")

    gender = st.multiselect(
        "Gender",
        sorted(df["demographic.gender"].unique().tolist())
    )

    primary_dx = st.multiselect(
        "Primary diagnosis",
        sorted(df["diagnoses.primary_diagnosis"].unique().tolist())
    )

    primary_site = st.multiselect(
        "Primary site",
        sorted(df["cases.primary_site"].unique().tolist())
    )

    ncit = st.multiselect(
        "NCIT code",
        sorted(df["ncit_code"].dropna().unique().tolist())
    )




    age_min, age_max = float(df["age_years"].min()), float(df["age_years"].max())
    age_range = st.slider("Age (years)", age_min, age_max, (age_min, age_max))

# Apply filters
filtered = df.copy()

if gender:
    filtered = filtered[filtered["demographic.gender"].isin(gender)]
if primary_dx:
    filtered = filtered[filtered["diagnoses.primary_diagnosis"].isin(primary_dx)]
if primary_site:
    filtered = filtered[filtered["cases.primary_site"].isin(primary_site)]
if ncit:
    filtered = filtered[filtered["ncit_code"].isin(ncit)]

filtered = filtered[filtered["age_years"].between(age_range[0], age_range[1])]

# Search
q = st.text_input("Search (matches any cell)", "")
if q.strip():
    mask = filtered.astype(str).apply(lambda s: s.str.contains(q, case=False, na=False))
    filtered = filtered[mask.any(axis=1)]

# Columns to display
default_cols = [
    "cases.submitter_id",
    "demographic.gender",
    "age_years",
    "diagnoses.primary_diagnosis",
    "cases.primary_site",
    "ncit_code",
]
cols = st.multiselect("Columns to display", df.columns.tolist() + ["age_years"], default=default_cols)

st.write(f"Filtered rows: **{len(filtered):,}**")
st.dataframe(filtered[cols], use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered CSV",
    filtered[cols].to_csv(index=False).encode("utf-8"),
    file_name="filtered_clinical.csv",
    mime="text/csv",
)
