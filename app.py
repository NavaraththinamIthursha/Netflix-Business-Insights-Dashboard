
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Netflix Business Insights Dashboard",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR / "netflix_dashboard_data.csv"
MODEL_PATH = BASE_DIR / "netflix_best_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.json"


# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None

    data = pd.read_csv(DATA_PATH)
    return data


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_comparison():
    if not COMPARISON_PATH.exists():
        return None

    with open(COMPARISON_PATH, "r") as file:
        return json.load(file)


df = load_data()
best_model = load_model()
model_comparison = load_comparison()


# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------

st.title("🎬 Netflix Business Insights Dashboard")

st.markdown(
    "### Data Analytics | Machine Learning | Business Intelligence"
)

st.markdown("---")

if df is None:
    st.error(
        "Dataset not found. Please place "
        "'netflix_dashboard_data.csv' in the same folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# 4. SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.title("Dashboard Filters")

filtered_df = df.copy()

# Content type filter
if "type" in df.columns:
    content_types = sorted(df["type"].dropna().unique().tolist())

    selected_types = st.sidebar.multiselect(
        "Select Content Type",
        content_types,
        default=content_types
    )

    filtered_df = filtered_df[
        filtered_df["type"].isin(selected_types)
    ]

# Country filter
if "primary_country" in df.columns:
    countries = sorted(
        df["primary_country"].dropna().unique().tolist()
    )

    selected_countries = st.sidebar.multiselect(
        "Select Countries",
        countries,
        default=[]
    )

    if selected_countries:
        filtered_df = filtered_df[
            filtered_df["primary_country"].isin(selected_countries)
        ]

# Genre filter
if "primary_genre" in df.columns:
    genres = sorted(
        df["primary_genre"].dropna().unique().tolist()
    )

    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        genres,
        default=[]
    )

    if selected_genres:
        filtered_df = filtered_df[
            filtered_df["primary_genre"].isin(selected_genres)
        ]

# Release year filter
if "release_year" in df.columns:
    min_year = int(df["release_year"].dropna().min())
    max_year = int(df["release_year"].dropna().max())

    selected_years = st.sidebar.slider(
        "Release Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )

    filtered_df = filtered_df[
        filtered_df["release_year"].between(
            selected_years[0],
            selected_years[1]
        )
    ]

st.sidebar.markdown("---")
st.sidebar.caption("Netflix Dataset | Auspify Technologies Internship")


# --------------------------------------------------
# 5. KPI SUMMARY
# --------------------------------------------------

st.header("📊 Business KPI Summary")

total_titles = len(filtered_df)

total_movies = (
    filtered_df["type"].eq("Movie").sum()
    if "type" in filtered_df.columns else 0
)

total_shows = (
    filtered_df["type"].eq("TV Show").sum()
    if "type" in filtered_df.columns else 0
)

total_countries = (
    filtered_df["primary_country"].nunique()
    if "primary_country" in filtered_df.columns else 0
)

total_genres = (
    filtered_df["primary_genre"].nunique()
    if "primary_genre" in filtered_df.columns else 0
)

average_movie_duration = 0

if {"type", "duration_value", "duration_unit"}.issubset(filtered_df.columns):
    movie_durations = filtered_df[
        (filtered_df["type"] == "Movie")
        & (filtered_df["duration_unit"] == "min")
    ]["duration_value"]

    if not movie_durations.empty:
        average_movie_duration = movie_durations.mean()

average_tv_seasons = 0

if {"type", "duration_value", "duration_unit"}.issubset(filtered_df.columns):
    tv_durations = filtered_df[
        (filtered_df["type"] == "TV Show")
        & (filtered_df["duration_unit"] == "Season")
    ]["duration_value"]

    if not tv_durations.empty:
        average_tv_seasons = tv_durations.mean()


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", f"{total_titles:,}")
col2.metric("Movies", f"{total_movies:,}")
col3.metric("TV Shows", f"{total_shows:,}")
col4.metric("Countries", f"{total_countries:,}")

col5, col6, col7 = st.columns(3)

col5.metric("Primary Genres", f"{total_genres:,}")
col6.metric("Average Movie Time", f"{average_movie_duration:.2f} min")
col7.metric("Average TV Seasons", f"{average_tv_seasons:.2f}")


# --------------------------------------------------
# 6. CONTENT TYPE DISTRIBUTION
# --------------------------------------------------

st.markdown("---")
st.header("🎥 Content Type Analysis")

if not filtered_df.empty and "type" in filtered_df.columns:

    content_counts = (
        filtered_df["type"]
        .value_counts()
        .reset_index()
    )

    content_counts.columns = ["Content Type", "Number of Titles"]

    fig_content = px.pie(
        content_counts,
        names="Content Type",
        values="Number of Titles",
        title="Movies vs TV Shows",
        hole=0.4
    )

    st.plotly_chart(fig_content, use_container_width=True)

    content_counts["Percentage"] = (
        content_counts["Number of Titles"]
        / content_counts["Number of Titles"].sum()
        * 100
    )

    st.dataframe(
        content_counts.round(2),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# 7. COUNTRY ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.header("🌍 Country Analysis")

if not filtered_df.empty and "primary_country" in filtered_df.columns:

    country_counts = (
        filtered_df["primary_country"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    country_counts.columns = ["Country", "Number of Titles"]

    fig_country = px.bar(
        country_counts.sort_values("Number of Titles"),
        x="Number of Titles",
        y="Country",
        orientation="h",
        title="Top 10 Countries by Primary Country",
        text="Number of Titles"
    )

    st.plotly_chart(fig_country, use_container_width=True)


# --------------------------------------------------
# 8. GENRE ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.header("🎭 Genre Analysis")

if not filtered_df.empty and "primary_genre" in filtered_df.columns:

    genre_counts = (
        filtered_df["primary_genre"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    genre_counts.columns = ["Genre", "Number of Titles"]

    fig_genre = px.bar(
        genre_counts.sort_values("Number of Titles"),
        x="Number of Titles",
        y="Genre",
        orientation="h",
        title="Top 10 Primary Genres",
        text="Number of Titles"
    )

    st.plotly_chart(fig_genre, use_container_width=True)


# --------------------------------------------------
# 9. YEARLY CONTENT ADDITION TREND
# --------------------------------------------------

st.markdown("---")
st.header("📈 Netflix Content Addition Trends")

if not filtered_df.empty and "year_added" in filtered_df.columns:

    yearly_counts = (
        filtered_df["year_added"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )

    yearly_counts.columns = ["Year", "Number of Titles"]

    fig_year = px.line(
        yearly_counts,
        x="Year",
        y="Number of Titles",
        markers=True,
        title="Number of Titles Added Each Year"
    )

    st.plotly_chart(fig_year, use_container_width=True)


# --------------------------------------------------
# 10. RATING ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.header("🔞 Content Rating Analysis")

if not filtered_df.empty and "rating" in filtered_df.columns:

    rating_counts = (
        filtered_df["rating"]
        .value_counts()
        .reset_index()
    )

    rating_counts.columns = ["Rating", "Number of Titles"]

    fig_rating = px.bar(
        rating_counts,
        x="Rating",
        y="Number of Titles",
        title="Distribution of Content Ratings",
        text="Number of Titles"
    )

    st.plotly_chart(fig_rating, use_container_width=True)


# --------------------------------------------------
# 11. MOVIE DURATION ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.header("⏱️ Movie Duration Analysis")

if {"type", "duration_value", "duration_unit"}.issubset(filtered_df.columns):

    movie_data = filtered_df[
        (filtered_df["type"] == "Movie")
        & (filtered_df["duration_unit"] == "min")
    ].copy()

    if not movie_data.empty:

        fig_duration = px.histogram(
            movie_data,
            x="duration_value",
            nbins=30,
            title="Distribution of Movie Durations",
            labels={"duration_value": "Duration (Minutes)"}
        )

        st.plotly_chart(fig_duration, use_container_width=True)

        st.write(
            "Average movie duration:",
            f"{movie_data['duration_value'].mean():.2f} minutes"
        )

    else:
        st.info("No movie duration data available for these filters.")


# --------------------------------------------------
# 12. MACHINE LEARNING MODEL COMPARISON
# --------------------------------------------------

st.markdown("---")
st.header("🤖 Machine Learning Model Comparison")

if model_comparison is not None:

    try:
        comparison_df = pd.DataFrame(model_comparison)

        if not comparison_df.empty:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            numeric_columns = [
                col for col in comparison_df.columns
                if col.lower() in [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1"
                ]
            ]

            if numeric_columns:
                model_name_column = next(
                    (
                        col for col in comparison_df.columns
                        if col.lower() in ["model", "model_name"]
                    ),
                    None
                )

                if model_name_column:
                    melted_df = comparison_df.melt(
                        id_vars=model_name_column,
                        value_vars=numeric_columns,
                        var_name="Metric",
                        value_name="Score"
                    )

                    fig_comparison = px.bar(
                        melted_df,
                        x=model_name_column,
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Machine Learning Model Performance"
                    )

                    st.plotly_chart(
                        fig_comparison,
                        use_container_width=True
                    )

    except Exception:
        st.info(
            "Model comparison file is available, "
            "but its format needs checking."
        )

else:
    st.info("Model comparison JSON file was not found.")


# --------------------------------------------------
# 13. MACHINE LEARNING PREDICTION
# --------------------------------------------------

st.markdown("---")
st.header("🔮 Netflix Content Rating Prediction")

st.write(
    "Enter content information to predict its possible Netflix rating."
)

if best_model is None:

    st.warning(
        "Machine learning model not found. "
        "Please place netflix_best_model.pkl in the project folder."
    )

else:

    with st.form("prediction_form"):

        prediction_col1, prediction_col2 = st.columns(2)

        with prediction_col1:

            content_type = st.selectbox(
                "Content Type",
                ["Movie", "TV Show"]
            )

            primary_country = st.selectbox(
                "Primary Country",
                sorted(df["primary_country"].dropna().unique())
                if "primary_country" in df.columns
                else ["United States", "India"]
            )

            primary_genre = st.selectbox(
                "Primary Genre",
                sorted(df["primary_genre"].dropna().unique())
                if "primary_genre" in df.columns
                else ["Dramas", "Comedies"]
            )

            release_year = st.number_input(
                "Release Year",
                min_value=1900,
                max_value=2026,
                value=2020
            )

        with prediction_col2:

            duration_unit = st.selectbox(
                "Duration Unit",
                ["min", "Season"]
            )

            if duration_unit == "min":
                duration_value = st.number_input(
                    "Duration",
                    min_value=1,
                    max_value=500,
                    value=100
                )
            else:
                duration_value = st.number_input(
                    "Number of Seasons",
                    min_value=1,
                    max_value=50,
                    value=2
                )

            genre_count = st.number_input(
                "Number of Genres",
                min_value=1,
                max_value=20,
                value=2
            )

            year_added = st.number_input(
                "Year Added to Netflix",
                min_value=2008,
                max_value=2026,
                value=2021
            )

        predict_button = st.form_submit_button(
            "Predict Content Rating"
        )

    if predict_button:

        input_data = pd.DataFrame([{
            "type": content_type,
            "primary_country": primary_country,
            "primary_genre": primary_genre,
            "release_year": release_year,
            "duration_value": duration_value,
            "duration_unit": duration_unit,
            "genre_count": genre_count,
            "year_added": year_added
        }])

        try:
            prediction = best_model.predict(input_data)[0]

            st.success(
                f"Predicted Netflix Rating: {prediction}"
            )

            st.caption(
                "This is an experimental classification result, "
                "not an official Netflix rating."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# 14. DATA PREVIEW
# --------------------------------------------------

st.markdown("---")
st.header("📁 Netflix Dataset Preview")

st.write(
    f"Filtered dataset contains {len(filtered_df):,} titles."
)

st.dataframe(
    filtered_df.head(100),
    use_container_width=True,
    hide_index=True
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Dataset",
    data=csv_data,
    file_name="netflix_filtered_data.csv",
    mime="text/csv"
)


# --------------------------------------------------
# 15. FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <h4>Netflix Business Insights Dashboard</h4>
        <p>Developed as part of the Data Science Internship</p>
        <p>Data Analytics | Machine Learning | Business Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)
