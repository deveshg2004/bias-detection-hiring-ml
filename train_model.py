import os
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")
nltk.download("wordnet")
os.makedirs("model", exist_ok=True)
df = pd.read_excel("job_applicant_with_biasnew.xlsx")
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
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
pickle.dump(model, open("model/bias_model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
print("Bias Detection Model trained successfully")



# venv\Scripts\activate
# pip install -r requirements.txt
# python app.py