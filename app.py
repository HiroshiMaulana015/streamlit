# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:03:53 2026

@author: user
"""
import streamlit as st
import pandas as pd

st.title("Titanic Dataset Explorer")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# 1. Check actual shape
st.subheader("Dataset Overview")
st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Focus on the required fields
cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
df_selected = df[cols]

# 3. Check missing-value counts
st.subheader("Missing Values (Selected Columns)")
st.write(df_selected.isnull().sum())

# 4. Preview the data
st.subheader("Data Preview")
st.dataframe(df_selected.head(10))
