# Aspect-Based-Sentiment-Analysis-of-Airline-Reviews
This project focuses on labeling airline reviews with aspect-level sentiments (e.g., Seat, Food, Staff) using GPT-4 for annotation, and then training SVM, Random Forest, and BERT models to classify the sentiments.

## 🧠 Problem Statement

Passengers share reviews online that include detailed feedback on various service aspects. Traditional sentiment models treat reviews holistically, ignoring specific opinions on different service components. This project builds a system that:

- Identifies **multiple aspects** in a review (e.g., Food, Staff, Punctuality).
- Classifies **individual sentiment** for each aspect.
- Leverages a **multi-label classification** setup.
- Compares the performance of **SVM**, **Random Forest**, and **BERT**-based models.

## ✅ Deliverables

- `Annotation.ipynb`: Code for semi-automated aspect-level labeling using GPT-4 API.
- `annotated_reviews_final.csv`: Final labeled dataset used for model training.
- `Text_Classifiers_v1.ipynb`: Main notebook for training & evaluation of classifiers.
- `methodology pipeline.jpg`: Visual representation of the project workflow.


## 📊 Models Trained

| Model               | Description                                       |
|---------------------|---------------------------------------------------|
| **SVM (OneVsRest)** | Classical linear model adapted for multi-label tasks |
| **Random Forest**   | Ensemble method with decision trees               |
| **BERT**            | Fine-tuned transformer-based deep learning model  |

Each model outputs a **multi-label sentiment prediction** for the aspects mentioned in each review.

## 📈 Evaluation Metrics

- **Accuracy**
- **F1 Score (macro/micro)**
- **Precision & Recall**
- **Visualization of performance across aspects**

All models are evaluated aspect-wise, and results are both tabulated and visualized.

## 🗂 Methodology Pipeline

![Methodology Pipeline](./methodology%20pipeline.jpg)

Steps:
1. Data preprocessing
2. Aspect-sentiment labeling using GPT-4
3. Model training (SVM, RF, BERT)
4. Evaluation and visualization

## 🔍 Summary of Findings

- BERT showed **superior performance** in detecting subtle sentiments across multiple aspects.
- SVM and Random Forest performed decently, especially on strongly polarized sentiments.
- GPT-4 proved effective for **semi-automated dataset labeling**, significantly reducing manual effort.

## ⚙️ Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- transformers (HuggingFace)
- datasets
- matplotlib / seaborn
- Jupyter Notebook
- OpenAI API key (for GPT-based labeling)
