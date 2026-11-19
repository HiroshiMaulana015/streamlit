# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:03:53 2026

@author: user
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Data Explorer", layout="wide")
st.title("Titanic Dataset — Interactive Explorer")

# ---------------------------------------------------------
# 1. Load and Inspect
# ---------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    return pd.read_csv(url)

raw_df = load_data()

st.header("1. Load & Inspect Dataset")

col1, col2 = st.columns(2)
col1.metric("Rows", raw_df.shape[0])
col2.metric("Columns", raw_df.shape[1])

st.subheader("Column Names & Data Types")
dtypes_df = pd.DataFrame({
    "Column": raw_df.columns,
    "Data Type": raw_df.dtypes.astype(str).values,
    "Missing Values": raw_df.isnull().sum().values
})
st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

st.subheader("First 5 Rows (Selected with .loc)")
core_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
st.dataframe(raw_df.loc[:4, core_cols], use_container_width=True)

# ---------------------------------------------------------
# 2. Handle Missing Values
# ---------------------------------------------------------
st.header("2. Handle Missing Values")

# Leave 'deck' out of the analysis and focus on core fields
df = raw_df.drop(columns=["deck"]).copy()

missing_before = df[core_cols].isnull().sum()

# Fill missing age with median and missing embarked with mode
age_median = df["age"].median()
embarked_mode = df["embarked"].mode()[0]

df["age"] = df["age"].fillna(age_median)
df["embarked"] = df["embarked"].fillna(embarked_mode)

missing_after = df[core_cols].isnull().sum()

missing_comparison = pd.DataFrame({
    "Missing Before": missing_before,
    "Missing After": missing_after
})
st.write(
    f"Dropped `deck` due to high missingness. Filled missing `age` with median (**{age_median:.1f}**) "
    f"and missing `embarked` with most frequent category (**{embarked_mode}**)."
)
st.dataframe(missing_comparison, use_container_width=True)

# ---------------------------------------------------------
# 3. Create Transformed Features
# ---------------------------------------------------------
st.header("3. Transformed Features")

# Min-Max normalization for age and fare (keeping original columns)
df["age_minmax"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
df["fare_minmax"] = (df["fare"] - df["fare"].min()) / (df["fare"].max() - df["fare"].min())

# Create age_group from original age values
bins = [0, 12, 18, 35, 60, float("inf")]
labels = ["Child (0-12)", "Teen (13-18)", "Young Adult (19-35)", "Adult (36-60)", "Older Adult (61+)"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)

st.dataframe(
    df[["age", "age_minmax", "fare", "fare_minmax", "age_group"]].head(8),
    use_container_width=True
)

# ---------------------------------------------------------
# 4. Apply Dummy Coding
# ---------------------------------------------------------
st.header("4. Dummy Coding (sex & embarked)")

df_dummies = pd.get_dummies(
    df[["survived", "sex", "embarked"]],
    columns=["sex", "embarked"],
    dtype=int
)

d_col1, d_col2 = st.columns(2)
with d_col1:
    st.subheader("Before Dummy Coding")
    st.dataframe(df[["survived", "sex", "embarked"]].head(5), use_container_width=True)
with d_col2:
    st.subheader("After Dummy Coding")
    st.dataframe(df_dummies.head(5), use_container_width=True)

# ---------------------------------------------------------
# 5. Interactive Dashboard
# ---------------------------------------------------------
st.header("5. Interactive Dashboard")

st.sidebar.header("Dashboard Filters")

selected_classes = st.sidebar.multiselect(
    "Passenger Class (pclass)",
    options=sorted(df["pclass"].unique()),
    default=sorted(df["pclass"].unique())
)

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sorted(df["sex"].unique()),
    default=sorted(df["sex"].unique())
)

selected_age_groups = st.sidebar.multiselect(
    "Age Group",
    options=labels,
    default=labels
)

filtered_df = df[
    (df["pclass"].isin(selected_classes)) &
    (df["sex"].isin(selected_sex)) &
    (df["age_group"].isin(selected_age_groups))
]

if filtered_df.empty:
    st.warning("No passengers match the selected filters. Adjust your sidebar selections.")
else:
    # Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Passengers", len(filtered_df))
    m2.metric("Survival Rate", f"{filtered_df['survived'].mean() * 100:.1f}%")
    m3.metric("Average Age", f"{filtered_df['age'].mean():.1f}")
    m4.metric("Average Fare", f"${filtered_df['fare'].mean():.2f}")

    # Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Survival Rate by Passenger Class")
        class_surv = filtered_df.groupby("pclass")["survived"].mean() * 100
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        class_surv.plot(kind="bar", ax=ax1, color=["#2b5c8f", "#4a89c8", "#8cb8e6"])
        ax1.set_xlabel("Passenger Class")
        ax1.set_ylabel("Survival Rate (%)")
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis="x", rotation=0)
        st.pyplot(fig1)

    with chart_col2:
        st.subheader("Survival Rate by Age Group")
        age_surv = filtered_df.groupby("age_group", observed=False)["survived"].mean() * 100
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        age_surv.plot(kind="bar", ax=ax2, color="#3b8ea5")
        ax2.set_xlabel("Age Group")
        ax2.set_ylabel("Survival Rate (%)")
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis="x", rotation=25)
        st.pyplot(fig2)

# ---------------------------------------------------------
# 6. Key Finding
# ---------------------------------------------------------
st.header("6. Key Finding")
st.info(
    "**Observed Association:** In this dataset, higher passenger class is strongly associated with a higher "
    "survival rate—1st-class passengers show a substantially higher survival percentage than 3rd-class passengers, "
    "and female passengers show higher survival rates than male passengers across all classes. "
    "These patterns reflect observed associations in the historical data rather than proven causal relationships."
)