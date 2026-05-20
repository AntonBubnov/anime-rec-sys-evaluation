# Anime Recommendation System Evaluation

This project is dedicated to exploring, implementing, and benchmarking different recommendation system architectures — from Deep Learning models to highly optimized In-Memory algorithms for real-time performance. A special focus is placed on solving the **Cold Start problem** using Graph Theory (the **Louvain algorithm**) and building a hybrid system that dynamically adapts to user preferences in milliseconds.

## 🎯 Project Goals
- **End-to-End Pipeline:** Complete process including data scraping, cleaning, NLP text processing (**Spacy**, **NLTK**), and feature engineering.
- **Deep Learning Integration:** Implementation of neural network approaches based on **Embedding layers**, including **DNN** and **Wide & Deep** architectures using **TensorFlow/Keras**.
- **Hybrid Recommendation System:** A combination of Collaborative Filtering (**SVD** via `Surprise`) and Content-Based filtering (**TF-IDF** + **Cosine Similarity**) to balance precision and catalog coverage.
- **Real-Time In-Memory Algorithm:** Vector projection of user profiles. Using pre-computed **Hash Maps** (`O(1)`), the system avoids heavy matrix recalculations, allowing instant recommendation updates.
- **Smart Onboarding (Active Learning):** Solving the cold start problem by detecting macro-communities within the anime similarity graph using the **Louvain method**. This enables the creation of an ideal Round-Robin queue for surveying new users.

## 📂 Project Structure
- `data/`: Raw and processed datasets (excluded from Git).
- `notebooks/`: 
  - `01_preprocessing.ipynb`: Data collection, cleaning, and synopsis processing (NLP). Outputs `anime_clean.parquet`.
  - `02_user_preprocessing.ipynb`: Processing user rating history, filtering inactive users, and chronological train/val/test splitting.
  - `03_model_Model_based_CF_DL.ipynb`: Building the baseline neural network (**DNN**) using Keras.
  - `04_model_Model_based_CF_DL.ipynb`: Advanced **Wide & Deep** neural network architecture.
  - `05_model_Hybrid_RS_SVD_TF-IDF.ipynb`: Training classical Matrix Factorization (**SVD**) and calculating the **TF-IDF** matrix.
  - `06_model_Real-Time_Item-Based.ipynb`: Optimizing matrices into lightweight dictionaries, building a network graph (**NetworkX**), Louvain clustering, and Active Learning queue generation.
  - `07_evaluate.ipynb`: Final visualization and comparison of all 8 approaches across the length of the user's history.
- `src/`: Utility scripts and shared functions.

## 🚀 Implemented Models
- **Baseline 1 & 2:** Recommendations purely by global popularity.
- **Content-Based:** Textual similarity of synopses and genres (**TF-IDF** + Cosine).
- **Collaborative Filtering:** Matrix factorization of user behavior (**SVD**).
- **Hybrid (SVD + Content):** A weighted combination of behavioral and content factors.
- **Deep Learning (v1 & v2):** Neural networks learning latent embeddings (**Keras**).
- **Real-Time In-Memory:** Fast-response vector item-based algorithm.

## 📈 Evaluation Metrics
The evaluation measures performance across different user history lengths (from 1 to 1000+ ratings) using Top-10 recommendations:
- **Precision@10:** Accuracy of recommendations in the Top-10.
- **Recall@10:** Completeness of finding relevant titles.
- **MRR (Mean Reciprocal Rank):** How high the first relevant title appears.
- **AUC (Area Under the ROC Curve):** Overall ranking quality.
- **Coverage:** The model's ability to recommend niche titles (Long Tail).

## 🛠 Tech Stack
- **Language:** Python 3.11+
- **Machine Learning & NLP:** TensorFlow, Keras, Scikit-learn, Surprise, Spacy, NLTK
- **Data Processing:** Pandas, NumPy, SciPy (Sparse Matrices), Parquet
- **Graph Theory:** NetworkX
- **Visualization:** Matplotlib, Seaborn
- **Tools:** VS Code, Jupyter Notebooks, Git

## 📊 Dataset
The project utilizes anime metadata and user ratings, incorporating data such as the [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) from Kaggle, supplemented by custom data collection and NLP preprocessing pipelines.