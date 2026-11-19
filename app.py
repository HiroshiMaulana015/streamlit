# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:03:53 2026

@author: user
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Config & Consistent Navy/Teal Palette
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic Passenger Insights",
    layout="wide"
)

NAVY = "#1B365D"
TEAL = "#2A9D8F"
SLATE = "#6C7A89"
CLASS_COLORS = {1: "#1B365D", 2: "#2A9D8F", 3: "#76B7B2"}
SEX_COLORS = {"female": "#2A9D8F", "male": "#1B365D"}

# ---------------------------------------------------------
# Load & Prepare Data (Raw, Cleaned, Transformed)
# ---------------------------------------------------------
@st.cache_data
def load_raw_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    return pd.read_csv(url)

raw_df = load_raw_data()
core_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]

# Cleaned dataset
cleaned_df = raw_df.drop(columns=["deck"]).copy()
age_median = cleaned_df["age"].median()
embarked_mode = cleaned_df["embarked"].mode()[0]
cleaned_df["age"] = cleaned_df["age"].fillna(age_median)
cleaned_df["embarked"] = cleaned_df["embarked"].fillna(embarked_mode)

# Transformed dataset
transformed_df = cleaned_df.copy()
transformed_df["age_norm"] = (
    (transformed_df["age"] - transformed_df["age"].min())
    / (transformed_df["age"].max() - transformed_df["age"].min())
)
transformed_df["fare_norm"] = (
    (transformed_df["fare"] - transformed_df["fare"].min())
    / (transformed_df["fare"].max() - transformed_df["fare"].min())
)

age_bins = [0, 12, 18, 35, 60, float("inf")]
age_labels = ["0-12 Child", "13-18 Teen", "19-35 Young Adult", "36-60 Adult", "61+ Older Adult"]
transformed_df["age_group"] = pd.cut(
    transformed_df["age"], bins=age_bins, labels=age_labels, include_lowest=True
)

# Dummy coding (sex and embarked only; survived untouched; dummies not normalized)
dummies = pd.get_dummies(transformed_df[["sex", "embarked"]], columns=["sex", "embarked"], dtype=int)
transformed_df = pd.concat([transformed_df, dummies], axis=1)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups")

# ---------------------------------------------------------
# Shared Sidebar Filters
# ---------------------------------------------------------
st.sidebar.header("Filter Passengers")

selected_classes = st.sidebar.multiselect(
    "Passenger class",
    options=sorted(transformed_df["pclass"].unique()),
    default=sorted(transformed_df["pclass"].unique())
)

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=sorted(transformed_df["sex"].unique()),
    default=sorted(transformed_df["sex"].unique())
)

selected_age_groups = st.sidebar.multiselect(
    "Age group",
    options=age_labels,
    default=age_labels
)

mask = (
    transformed_df["pclass"].isin(selected_classes)
    & transformed_df["sex"].isin(selected_sex)
    & transformed_df["age_group"].isin(selected_age_groups)
)
filtered_df = transformed_df.loc[mask].copy()

# ---------------------------------------------------------
# Four-Tab Layout
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Explore",
    "Preprocessing",
    "Data Explorer"
])

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================
with tab1:
    if filtered_df.empty:
        st.warning("No passengers match the current sidebar filters.")
    else:
        total_passengers = len(filtered_df)
        survivors = int(filtered_df["survived"].sum())
        survival_rate = filtered_df["survived"].mean() * 100
        avg_fare = filtered_df["fare"].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Passengers in Selection", f"{total_passengers:,}")
        k2.metric("Survivors", f"{survivors:,}")
        k3.metric("Survival Rate", f"{survival_rate:.1f}%")
        k4.metric("Average Fare", f"${avg_fare:.2f}")

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Survival Counts")
            surv_counts = filtered_df["survived"].value_counts().reindex([0, 1], fill_value=0)
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar(["Did Not Survive", "Survived"], surv_counts.values, color=[NAVY, TEAL])
            ax.set_ylabel("Number of Passengers")
            ax.set_title("Passenger Survival Counts")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        with c2:
            st.subheader("Survival Rate by Passenger Class")
            class_surv = filtered_df.groupby("pclass")["survived"].mean() * 100
            bar_colors = [CLASS_COLORS[c] for c in class_surv.index]
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar([f"Class {c}" for c in class_surv.index], class_surv.values, color=bar_colors)
            ax.set_ylabel("Survival Rate (%)")
            ax.set_ylim(0, 100)
            ax.set_title("Survival Rate by Passenger Class")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        st.caption("Note: Displayed rates and counts describe this Titanic dataset only.")

# =========================================================
# TAB 2 — EXPLORE
# =========================================================
with tab2:
    st.subheader("Compare Passenger Groups")

    if filtered_df.empty:
        st.warning("No passengers match the current sidebar filters.")
    else:
        r1c1, r1c2 = st.columns(2)

        with r1c1:
            class_surv = filtered_df.groupby("pclass")["survived"].mean() * 100
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar(
                [f"Class {c}" for c in class_surv.index],
                class_surv.values,
                color=[CLASS_COLORS[c] for c in class_surv.index]
            )
            ax.set_title("Survival Rate by Passenger Class")
            ax.set_xlabel("Passenger Class")
            ax.set_ylabel("Survival Rate (%)")
            ax.set_ylim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        with r1c2:
            sex_surv = filtered_df.groupby("sex")["survived"].mean() * 100
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar(
                [s.capitalize() for s in sex_surv.index],
                sex_surv.values,
                color=[SEX_COLORS[s] for s in sex_surv.index]
            )
            ax.set_title("Survival Rate by Sex")
            ax.set_xlabel("Sex")
            ax.set_ylabel("Survival Rate (%)")
            ax.set_ylim(0, 100)
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        r2c1, r2c2 = st.columns(2)

        with r2c1:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.hist(filtered_df["age"], bins=20, color=TEAL, edgecolor="white")
            ax.set_title("Age Distribution")
            ax.set_xlabel("Age (Years)")
            ax.set_ylabel("Number of Passengers")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        with r2c2:
            fare_by_class = filtered_df.groupby("pclass")["fare"].mean()
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            ax.bar(
                [f"Class {c}" for c in fare_by_class.index],
                fare_by_class.values,
                color=[CLASS_COLORS[c] for c in fare_by_class.index]
            )
            ax.set_title("Average Fare by Passenger Class")
            ax.set_xlabel("Passenger Class")
            ax.set_ylabel("Average Fare ($)")
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

        st.info(
            "**Finding:** Higher passenger class and female passenger status are associated with "
            "substantially higher survival rates in this dataset."
        )

# =========================================================
# TAB 3 — PREPROCESSING
# =========================================================
with tab3:
    st.subheader("Data Cleaning & Feature Engineering")

    with st.expander("Dataset Inspection (Shape, Types & .loc Preview)", expanded=False):
        st.caption("Inspects raw dimensions, data types, and the first five records using `.loc`.")
        st.write(f"**Raw Dataset Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
        info_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Data Type": raw_df.dtypes.astype(str).values,
            "Missing Values": raw_df.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        st.dataframe(raw_df.loc[:4, core_cols], use_container_width=True)

    with st.expander("Section 1: Missing Values", expanded=True):
        st.caption(
            f"Missing `age` values were imputed with the median ({age_median:.1f}), missing `embarked` "
            f"values with the most frequent category ({embarked_mode}), and `deck` was excluded."
        )
        missing_comp = pd.DataFrame({
            "Missing Before": raw_df[["age", "embarked"]].isnull().sum(),
            "Missing After": cleaned_df[["age", "embarked"]].isnull().sum()
        })
        st.dataframe(missing_comp, use_container_width=True)

    with st.expander("Section 2: Normalization", expanded=False):
        st.caption(
            "Min-Max normalization rescales `age` and `fare` to a [0, 1] range while preserving the original columns."
        )
        st.dataframe(
            transformed_df[["age", "age_norm", "fare", "fare_norm"]].head(8),
            use_container_width=True
        )

    with st.expander("Section 3: Age Discretization", expanded=False):
        st.caption(
            "Original `age` values were grouped into five ordered brackets (`age_group`) to compare life stages."
        )
        st.dataframe(
            transformed_df[["age", "age_group"]].head(8),
            use_container_width=True
        )

    with st.expander("Section 4: Dummy Coding", expanded=False):
        st.caption(
            "Categorical `sex` and `embarked` columns were converted into binary indicator columns without altering `survived`."
        )
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Before**")
            st.dataframe(cleaned_df[["survived", "sex", "embarked"]].head(5), use_container_width=True)
        with d2:
            st.markdown("**After**")
            dummy_preview_cols = ["survived"] + list(dummies.columns)
            st.dataframe(transformed_df[dummy_preview_cols].head(5), use_container_width=True)

# =========================================================
# TAB 4 — DATA EXPLORER
# =========================================================
with tab4:
    st.subheader("Interactive Data Table")

    view_choice = st.radio(
        "Select dataset view",
        options=["Raw", "Cleaned", "Transformed"],
        horizontal=True
    )

    if view_choice == "Raw":
        active_df = raw_df.loc[mask].copy()
        default_cols = core_cols
    elif view_choice == "Cleaned":
        active_df = cleaned_df.loc[mask].copy()
        default_cols = core_cols
    else:
        active_df = filtered_df.copy()
        default_cols = ["survived", "pclass", "sex", "age", "age_group", "age_norm", "fare", "fare_norm", "embarked"]

    selected_cols = st.multiselect(
        "Choose columns to display",
        options=list(active_df.columns),
        default=default_cols
    )

    st.write(f"**Matching rows after filters:** {len(active_df):,}")

    if selected_cols:
        st.dataframe(active_df[selected_cols], use_container_width=True, height=400)
    else:
        st.info("Select at least one column above to display records.")