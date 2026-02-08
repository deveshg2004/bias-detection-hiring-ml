import os
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download("stopwords")
nltk.download("wordnet")
os.makedirs("model", exist_ok=True)

df = pd.read_excel("job_applicant_dataset.xlsx")
stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
df["combined_text"] = (
    df["Job Description"].fillna("") + " " +
    df["Resume"].fillna("")
).apply(preprocess)
y = df["Bias_in_JobDesc"]
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df["combined_text"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
nb = MultinomialNB()
models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Naive Bayes": nb
}
print("\nModel Accuracy Comparison:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc * 100:.2f}%")
pickle.dump(lr, open("model/bias_model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
print("\nFinal model saved: Logistic Regression")


# venv\Scripts\activate
# pip install -r requirements.txt
# python app.py