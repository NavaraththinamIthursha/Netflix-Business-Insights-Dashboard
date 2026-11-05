# Netflix Business Insights Dashboard

An interactive Data Science dashboard built using Python and Streamlit to explore Netflix content, understand business trends, and predict content ratings using Machine Learning.

## Features

- Netflix content distribution analysis
- Top countries and genres
- Yearly content addition trends
- Rating distribution analysis
- Movie duration analysis
- Interactive filters
- Machine Learning model comparison
- Content rating prediction
- Dataset preview and filtering

## Machine Learning

Two classification models were evaluated:

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Random Forest | 48.18% | 43.95% |
| Gradient Boosting | 45.68% | 45.85% |

Gradient Boosting was selected based on its balanced accuracy and macro F1-score.

Note: These are experimental classification results, not guaranteed prediction performance.

## Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib

## Project Structure

```text
Netflix_Business_Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── netflix_best_model.pkl
├── model_comparison.json
└── charts/
    ├── 01_content_distribution.png
    ├── 02_top_countries.png
    ├── 03_top_genres.png
    ├── 04_yearly_trend.png
    ├── 05_rating_distribution.png
    ├── 06_movie_duration.png
    └── 07_confusion_matrix.png
```

## How to Run

1. Install Python 3.12.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Place `netflix_dashboard_data.csv` in the project folder.
4. Run the application:

```bash
streamlit run app.py
```

## Author
Ithursha Navaraththinam
Data Science Undergraduate