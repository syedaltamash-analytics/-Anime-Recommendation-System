## 👤 Author

<div align="center">

### Syed Mohd Altamash

*Data Science & Machine Learning Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-syedaltamash--analytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/syedaltamash-analytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed%20Mohd%20Altamash-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syedaltamash-analytics)

</div>

---
# 🎌 Anime Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-16c79a?style=for-the-badge)

**A full-stack content-based anime recommender with ML classification, clustering, and regression — built on cosine similarity + TF-IDF with a dark cyberpunk aesthetic.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [Models](#-ml-models) · [Structure](#-project-structure) · [Results](#-results) · [Roadmap](#-roadmap)

</div>

---

## 📸 Demo

> All charts are generated automatically when you run the notebook. Theme: **dark anime / cyberpunk** (`#0f0f1a` background, `#e94560` accent).

| Rating Distribution | Top Genres | Model Comparison |
|---|---|---|
| ![Rating](plot_01_rating_distribution.png) | ![Genres](plot_03_top_genres.png) | ![Models](plot_12_model_comparison.png) |

---

## ✨ Features

### 🎯 Recommendation Engine
- **Cosine Similarity Recommender** — content-based filtering using genre + type + normalised stats
- **TF-IDF Genre Recommender** — weights rare genres more heavily for niche-accurate matching
- Configurable **similarity threshold** (default: 0.5) controlling breadth vs. precision
- Threshold sensitivity experiment with visualisation

### 📊 Exploratory Data Analysis
- Rating distribution with mean/median overlay
- Broadcast type breakdown (TV / OVA / Movie / ONA / Special)
- Top 15 genres by frequency
- Top 10 highest-rated anime
- Community members distribution (log scale)
- Rating vs. popularity scatter plot
- Average rating by type
- Feature correlation heatmap
- Episodes distribution + boxplot by type *(new)*
- Genre vs. average rating analysis *(new)*

### 🤖 Machine Learning — Classification
| Model | Notes |
|---|---|
| K-Nearest Neighbours | Instance-based baseline |
| Logistic Regression | Fast, interpretable linear model |
| Decision Tree | Interpretable; `max_depth=10` |
| Random Forest | Ensemble; best feature importances |
| SVM (RBF kernel) | Maximum-margin classifier |
| Gradient Boosting | Iterative boosting; typically top performer |
| XGBoost *(optional)* | Extreme gradient boosting |

Evaluation: **Accuracy · Precision · Recall · F1 · ROC-AUC · 5-fold CV · Confusion Matrices**

### 🔬 Advanced Analysis *(new)*
- **K-Means Clustering** (Elbow method + PCA 2D projection + cluster profiling)
- **Rating Regression** (Ridge · Lasso · Random Forest Regressor) with R²/RMSE/MAE
- **Popularity Tier Analysis** — Niche → Moderate → Popular → Blockbuster
- **GridSearchCV Hyperparameter Tuning** for Random Forest

---

## 🗂 Dataset

| Field | Description |
|---|---|
| `anime_id` | Unique identifier |
| `name` | Anime title |
| `genre` | Comma-separated genres (e.g. `"Action, Comedy, Drama"`) |
| `type` | Broadcast type: TV · Movie · OVA · ONA · Special · Music |
| `episodes` | Number of episodes (`"Unknown"` for some entries) |
| `rating` | Community average rating (0–10) |
| `members` | Number of community members who added it |

**Source:** [MyAnimeList Dataset on Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

> Place `anime.csv` in the same directory as the notebook before running.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/anime-recommendation-system.git
cd anime-recommendation-system
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Install XGBoost
```bash
pip install xgboost
```

### 5. Launch Jupyter
```bash
jupyter notebook Anime_Recommendation_System_Enhanced.ipynb
```

---

## 📦 Requirements

```
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
seaborn>=0.12
scikit-learn>=1.2
jupyter>=1.0
xgboost>=1.7   # optional
```

Or install everything at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter xgboost
```

---

## 🚀 Usage

### Run the full notebook
Open the notebook and run **Kernel → Restart & Run All**. All sections execute sequentially and save chart PNGs to the working directory.

### Get recommendations in Python
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# After running the notebook cells, call:
recs = recommend_anime("Naruto", top_n=10, threshold=0.5)
print(recs[["name", "type", "rating", "similarity"]])
```

**Example output:**
```
                        name   type  rating  similarity
0            Naruto: Shippuden     TV    8.17    0.921
1            Bleach                TV    7.83    0.894
2            One Piece             TV    8.54    0.881
3            Dragon Ball Z         TV    8.32    0.867
...
```

### TF-IDF recommender
```python
recs_tfidf = recommend_tfidf("Death Note", top_n=10)
print(recs_tfidf[["name", "type", "rating", "tfidf_similarity"]])
```

### Change similarity threshold
```python
# Broad recommendations (lower threshold)
recs_broad = recommend_anime("Sword Art Online", top_n=20, threshold=0.3)

# Strict recommendations (higher threshold)  
recs_strict = recommend_anime("Sword Art Online", top_n=5, threshold=0.7)
```

---

## 🤖 ML Models

### Binary Classification Task
**Target:** `popular` — 1 if `members > median`, else 0 (balanced 50/50 split by design)

**Features:** Genre one-hot (44+ columns) + type dummies + normalised rating/members/episodes

### Results Summary *(approximate — varies by dataset version)*

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Gradient Boosting | ~0.88 | ~0.88 | ~0.95 |
| Random Forest | ~0.87 | ~0.87 | ~0.94 |
| XGBoost | ~0.88 | ~0.88 | ~0.95 |
| Logistic Regression | ~0.83 | ~0.83 | ~0.90 |
| SVM (RBF) | ~0.84 | ~0.84 | ~0.91 |
| Decision Tree | ~0.79 | ~0.79 | ~0.79 |
| KNN (k=7) | ~0.81 | ~0.81 | ~0.88 |

### Rating Regression Task
**Target:** `rating` (continuous float, 0–10)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| RF Regressor | ~0.75 | ~0.55 | ~0.47 |
| Ridge Regression | ~0.90 | ~0.68 | ~0.31 |
| Lasso Regression | ~0.92 | ~0.70 | ~0.28 |

---

## 📁 Project Structure

```
anime-recommendation-system/
│
├── Anime_Recommendation_System_Enhanced.ipynb  # Main notebook (79 cells)
├── anime.csv                                   # Dataset (download from Kaggle)
├── README.md                                   # This file
├── requirements.txt                            # Python dependencies
│
├── plots/                                      # Auto-generated chart PNGs
│   ├── plot_01_rating_distribution.png
│   ├── plot_02_type_distribution.png
│   ├── plot_03_top_genres.png
│   ├── plot_04_top10_rated.png
│   ├── plot_05_members_distribution.png
│   ├── plot_06_rating_vs_members.png
│   ├── plot_07_avg_rating_by_type.png
│   ├── plot_08_correlation_heatmap.png
│   ├── plot_09_episodes_analysis.png
│   ├── plot_10_genre_rating.png
│   ├── plot_11_threshold_effect.png
│   ├── plot_12_model_comparison.png
│   ├── plot_13_feature_importance.png
│   ├── plot_14_confusion_matrices.png
│   ├── plot_15_roc_auc.png
│   ├── plot_16_cross_validation.png
│   ├── plot_17_elbow.png
│   ├── plot_18_kmeans_pca.png
│   ├── plot_19_regression_r2.png
│   └── plot_20_popularity_tiers.png
```

---

## 🧪 Notebook Sections

| # | Section | Key Steps |
|---|---------|-----------|
| 0 | **Imports & Setup** | Libraries, global plot theme, constants |
| 1 | **Data Loading** | `pd.read_csv`, shape/dtype inspection, descriptive stats |
| 2 | **Data Cleaning** | NaN handling, type conversion, duplicate removal, outlier detection |
| 3 | **EDA** | 10 visualisations covering rating, genre, type, members, episodes |
| 4 | **Feature Engineering** | MultiLabelBinarizer, one-hot encoding, MinMaxScaler |
| 5 | **Recommendation System** | Cosine similarity + TF-IDF recommender + threshold experiment |
| 6 | **ML Classification** | 7 models, confusion matrices, ROC-AUC, CV, GridSearchCV |
| 7 | **Advanced Analysis** | K-Means clustering, rating regression, popularity tiers |
| 8 | **Conclusions** | Full project summary, findings, future work |

Each section ends with a **📝 Conclusion** markdown cell summarising findings and design decisions.

---

## 📊 Key Findings

- **TV series** dominate the catalogue (~55–60% of all entries)
- **Comedy, Action, Adventure** are the top 3 genres by count; **Thriller/Mystery** score higher on average despite lower volume
- `log(members)` shows a weak-to-moderate positive correlation with rating (Pearson r ≈ 0.3–0.4) — popular ≠ universally high quality
- **Gradient Boosting / Random Forest** consistently outperform simpler classifiers
- **K-Means (K=5)** reveals interpretable clusters: Action/Shounen · Romance/Slice-of-Life · Sci-Fi/Mecha · Kids/Family · Niche/Experimental
- Feature importance analysis confirms `norm_members` is the single strongest predictor of popularity class

---

## 🗺 Roadmap

- [ ] **Collaborative Filtering** — user–item matrix via SVD / ALS
- [ ] **NLP on Synopses** — BERT embeddings for semantic similarity
- [ ] **Hybrid Recommender** — content + collaborative + popularity weighting
- [ ] **Neural Collaborative Filtering (NCF)** — deep learning approach
- [ ] **FastAPI / Flask deployment** — REST endpoint for real-time recommendations
- [ ] **Streamlit web app** — interactive UI for browsing recommendations
- [ ] **User rating personalisation** — per-user preference profiles

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- Dataset: [MyAnimeList Database — Kaggle (CooperUnion)](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
- [Scikit-learn](https://scikit-learn.org/) for ML utilities
- [MyAnimeList](https://myanimelist.net/) for the underlying community data

---

<div align="center">

Made with ❤️ and 🍜 &nbsp;|&nbsp; If this helped you, give it a ⭐

</div>
---

## 👤 Author

<div align="center">

### Syed Mohd Altamash

*Data Science & Machine Learning Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-syedaltamash--analytics-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/syedaltamash-analytics)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Syed%20Mohd%20Altamash-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/syedaltamash-analytics)

</div>

---
