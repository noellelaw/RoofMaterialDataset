import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# Set page configuration
st.set_page_config(page_title="Roof Materials Classification Dataset", layout="wide")

# Title and intro
st.title("Building Roof Materials Classification Dataset")
st.markdown("""
This app provides an overview and download link of a dataset used for classifying building roof materials based on satellite imagery.
You can explore the geographic distribution of the sample points and view dataset statistics below.
Here are some details about the dataset:
- **Dataset Size**: 8,688 images
- **Materials**: 14 classes
- **Image Size**: 256x256 pixels
- **Image Format**: JPEG
""")

# Load CSV data
@st.cache_data

def load_data():
    csv_path = "resources/parsed_roof_data.csv"  # <-- update path as needed
    df = pd.read_csv(csv_path)
    return df

df = load_data()

# Ensure valid coordinates
df = df.dropna(subset=["latitude", "longitude"])
df["latitude"] = df["latitude"].astype(float)
df["longitude"] = df["longitude"].astype(float)

# Prepare city-level aggregation
df_city = df.dropna(subset=["city", "city_coordinates"])
df_city[["city_lat", "city_lon"]] = df_city["city_coordinates"].str.split(", ", expand=True).astype(float)
df_grouped = df_city.groupby(["city", "city_lat", "city_lon"]).size().reset_index(name="count")

# Load known locations dictionary
@st.cache_data

def load_reference_locations():
    with open("resources/location_coordinates.json", "r") as f:
        location_dict = json.load(f)
    data = []
    for name, coord in location_dict.items():
        if "not found" not in coord.lower() and "error" not in coord.lower():
            lat, lon = map(float, coord.split(", "))
            data.append({"name": name, "latitude": lat, "longitude": lon})
    return pd.DataFrame(data)

reference_df = load_reference_locations()

# Show map with both city data (size-coded) and reference markers
st.subheader("Mapped Dataset Locations")
map_df = pd.concat([
    df_grouped.rename(columns={"city_lat": "latitude", "city_lon": "longitude"})[["latitude", "longitude", "count"]].rename(columns={"count": "size"}),
    #reference_df.assign(size=20)  # fixed size for reference points
], ignore_index=True)

st.map(map_df)

# Histogram of number of images per city
st.subheader("Image Count by City")
st.bar_chart(df_grouped.set_index("city")["count"])

# Additional Statistics
st.subheader("Dataset Statistics")

total = len(df)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Roof Material Distribution**")
    st.bar_chart(df["roof_material"].value_counts())

with col2:
    height_data = df["height"].dropna().astype(float)
    height_counts = height_data.value_counts().sort_index()
    height_percent = round(len(height_data) / total * 100, 1)
    st.markdown(f"**Height Distribution ({height_percent}% data coverage)**")
    st.bar_chart(height_counts)

col3, col4 = st.columns(2)
with col3:
    story_data = df["numstories"].dropna().astype(float)
    story_counts = story_data.value_counts().sort_index()
    story_percent = round(len(story_data) / total * 100, 1)
    st.markdown(f"**Number of Stories ({story_percent}% data coverage)**")
    st.bar_chart(story_counts)

with col4:
    roofshape_data = df["roofshape"].dropna().astype(str)
    roofshape_counts = roofshape_data.value_counts()
    roofshape_percent = round(len(roofshape_data) / total * 100, 1)
    st.markdown(f"**Roof Shape Frequency ({roofshape_percent}% data coverage)**")
    st.bar_chart(roofshape_counts)

fp_area_data = df["fpArea"].dropna().astype(float).value_counts().sort_index()
fp_area_percent = round(len(fp_area_data) / total * 100, 1)
st.markdown(f"**Footprint Area Distribution ({fp_area_percent}% data coverage)**")
st.bar_chart(fp_area_data)
# Country histogram if available
if "country" in df.columns:
    st.subheader("Images by Country")
    st.bar_chart(df["country"].value_counts())



# Show sample data
st.subheader("Dataset Preview")
st.dataframe(df.head())