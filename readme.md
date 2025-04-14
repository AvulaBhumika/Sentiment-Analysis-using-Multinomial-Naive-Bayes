# 🎯 Sentiment Analysis using Multinomial Naive Bayes

![Sentiment Analytics Using NLP](https://github.com/user-attachments/assets/c8331ed9-8959-4d20-b45d-70db5ba7eb3f)



This project is an end-to-end implementation of a **Sentiment Analysis** model using the classic **Multinomial Naive Bayes** algorithm. It leverages the [IMDB Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) for binary sentiment classification (`positive` / `negative`).

---

## 📦 Dataset

- **Source:** Kaggle - IMDB Movie Review Dataset
- **Size:** 50,000 labeled reviews
- **Columns:**  
  - `review`: Text review of a movie  
  - `sentiment`: Target label (`positive` or `negative`)

---

## 📌 Features

- Text preprocessing: HTML removal, lowercasing, stopword filtering
- TF-IDF vectorization with bigrams
- Multinomial Naive Bayes classifier
- Model evaluation with accuracy, confusion matrix, and classification report
- Real-time prediction function with confidence score

---

## 🔧 Requirements

Install required libraries using:

```bash
pip install pandas numpy scikit-learn nltk
```

For downloading stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## 📊 Model Performance 

| Metric        | Score     |
|---------------|-----------|
| Accuracy      | 89%+      |
| Precision     | ~0.85     |
| Recall        | ~0.86     |
| F1-Score      | ~0.89     |

> Note: Actual metrics may vary based on train/test split and random state.

---

## 🧠 Example Inference

```python
predict_sentiment("This movie was absolutely wonderful and emotionally touching.")
# Output: Sentiment: positive, Confidence: 0.96
```

---

## 📂 Project Structure

```text
├── IMDB Dataset.csv
├── sentiment_analysis.py      # Core implementation
├── README.md                  # Project overview
```

---

## ✅ Future Enhancements

- Replace Naive Bayes with Logistic Regression, XGBoost, or Transformers
- Streamlit-based UI for interactive review scoring
- Model persistence with `joblib` for production deployment
- Advanced preprocessing: lemmatization, named entity removal, negation handling

---

## 📜 License

This project is licensed under the [MIT License]().




