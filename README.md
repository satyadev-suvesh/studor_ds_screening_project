# PathAI Engine — Studor DS Screening Project

A data-driven system that tracks student engagement, flags disengagement risk, and recommends courses, built on the [OULAD dataset](https://analyse.kmi.open.ac.uk/open_dataset) (32,593 students, 22 course-presentations).

## What This Does

**Task 1 - Engagement Scoring:** Computes a dynamic 0-100 engagement score per student per week using 11 behavioral features with outcome-calibrated weights. Discovers student archetypes via K-Means and DTW clustering.

**Task 2 - Disengagement Prediction:** Predicts withdrawal/failure before Week 6 using only data available at the prediction point. LightGBM achieves AUC=0.84 on leave-one-course-out CV with 82% recall after threshold tuning.

**Task 3 - Course Recommendations:** Recommends the 3 best next-semester courses using content-based filtering, collaborative filtering (KNN + SVD), and a hybrid blender. Handles cold-start students via demographic matching and popularity fallback.

## Project Structure

```
studor_ds_screening_project/
│
├── data/                         # Raw OULAD CSVs (not committed, see Setup)
│
├── output/
│   ├── task1_output/             # Engagement scores, archetype plots, SHAP heatmap
│   ├── task2_output/             # Model metrics, calibration, fairness audit, dashboard
│   └── task3_output/             # Course profiles, evaluation, sample recommendations
│
├── task1_behavioral/
│   ├── task1_scoring.py          # Feature engineering, scoring, archetypes, SHAP
│   └── task1_visualizations.py   # All Task 1 plots
│
├── task2_predictive/
│   └── task2_model.py            # Prediction pipeline, calibration, fairness, alerts
│
├── task3_recommendation/
│   └── task3_recommender.py      # Content-based, CF, hybrid, cold-start, evaluation
│
├── report.tex                    # 6-page LaTeX report
├── requirements.txt
└── README.md
```

## Setup

### 1. Download the dataset

Download the OULAD dataset from [Kaggle](https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad) and place all 7 CSV files in the `data/` folder:

```
data/
├── courses.csv
├── vle.csv
├── assessments.csv
├── studentInfo.csv
├── studentVle.csv
├── studentAssessment.csv
└── studentRegistration.csv
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipelines

Each task runs independently from the project root:

```bash
# Task 1: Engagement Scoring (~5 min)
python task1_behavioral/task1_scoring.py

# Task 2: Disengagement Prediction (~8 min)
python task2_predictive/task2_model.py

# Task 3: Course Recommendations (~3 min)
python task3_recommendation/task3_recommender.py
```

All outputs (CSVs, PNGs, JSON specs) are saved to `output/taskN_output/`.

You can also pass custom paths:

```bash
python task2_predictive/task2_model.py --data-dir ./data --output-dir ./output/task2_output
```

### 4. Compile the report

```bash
pdflatex report.tex
```

### 5. Launch the Streamlit dashboard (optional)

```bash
streamlit run output/task2_output/task2_dashboard.py
```

## Requirements

- Python 3.9+
- See `requirements.txt` for full list

Core dependencies: pandas, numpy, scikit-learn, lightgbm, shap, matplotlib, seaborn, scipy, dtaidistance.

## Key Results

| Metric | Value |
|---|---|
| Engagement features engineered | 11 (across 5 dimensions) |
| Student archetypes discovered | 3 (Steady Engager, Fading Away, Minimal) |
| Best prediction model | LightGBM |
| AUC (leave-one-course-out) | 0.837 |
| Recall (tuned threshold=0.36) | 0.817 |
| Top risk driver | Assessment completion (SHAP=0.84) |
| Recommendation P@3 (hybrid) | 0.889 |
| Cold-start coverage | 100% of catalog |
