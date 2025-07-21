# Fake_News_Detection



# 📰 Fake News Detection with Logistic Regression

This project is a simple machine learning model that classifies news articles as either **Fake** or **Real**, using text processing techniques and a Logistic Regression classifier.

## 📁 Dataset

We used two datasets:
- `Fake.csv` – containing fake news articles
- `True.csv` – containing real news articles

Each dataset contains the following columns:
- `title`: Title of the news
- `text`: Full content of the news
- `subject`: Topic
- `date`: Date of publication

## 🧹 Preprocessing

1. Combined both datasets and labeled:
   - Fake = 0
   - True = 1
2. Removed unnecessary columns (`title`, `subject`, `date`)
3. Cleaned the `text` using regex to:
   - Lowercase text
   - Remove punctuation, URLs, HTML tags, digits, etc.

## 🧠 Model

- Text features were extracted using **TF-IDF Vectorizer**
- Trained a **Logistic Regression** classifier
- Evaluated using:
  - Accuracy
  - Classification report (Precision, Recall, F1-score)

## 🧪 Evaluation Example

The model shows good performance:

```text
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      5684
           1       0.99      0.99      0.99      5407

    accuracy                           0.99     11091
   macro avg       0.99      0.99      0.99     11091
weighted avg       0.99      0.99      0.99     11091
