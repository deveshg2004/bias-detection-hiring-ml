from flask import Flask, render_template, request
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from PyPDF2 import PdfReader
nltk.download("stopwords")
nltk.download("wordnet")
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
model = pickle.load(open("model/bias_model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
biased_words = {
    "ambitious": ("goal-oriented", "low"),
    "competitive": ("motivated", "low"),
    "driven": ("determined", "low"),
    "assertive": ("confident communicator", "low"),
    "independent": ("self-reliant", "low"),

    "compassionate": ("supportive", "high"),
    "nurturing": ("encouraging", "high"),
    "sensitive": ("understanding", "high"),
    "empathetic": ("understanding", "high"),

    "supportive": ("collaborative", "low"),
    "caring": ("considerate", "low"),
    "helpful": ("resourceful", "low"),
    "kind": ("respectful", "low"),
    "polite": ("professional", "low"),
    "warm": ("approachable", "low"),
    "loyal": ("committed", "low"),

    "aggressive": ("proactive", "high"),
    "dominant": ("confident", "high"),
    "fearless": ("confident", "high"),
    "manpower": ("workforce", "high"),
    "manpower planning": ("workforce planning", "high"),
    "chairman": ("chairperson", "high"),
    "salesman": ("sales executive", "high"),
    "businessman": ("business professional", "high"),
    "foreman": ("supervisor", "high"),
    "policeman": ("police officer", "high"),
    "fireman": ("firefighter", "high"),

    "he": ("they", "high"),
    "him": ("them", "high"),
    "his": ("theirs", "high"),
    "she": ("they", "high"),
    "her": ("them", "high"),
    "hers": ("theirs", "high")
}
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    found_words = {}
    extracted_text = ""
    if request.method == "POST":
        file = request.files["pdf_file"]
        if file and file.filename.endswith(".pdf"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            extracted_text = extract_text_from_pdf(file_path)
            clean_text = preprocess(extracted_text)
            vector = tfidf.transform([clean_text])
            prediction = model.predict(vector)[0]
            for word in extracted_text.lower().split():
                if word in biased_words:
                    neutral, severity = biased_words[word]
                    found_words[word] = (neutral, severity)
            high_severity_found = any(
                severity == "high" for _, severity in found_words.values()
            )
            if high_severity_found:
                result = "Biased"
            else:
                result = "Biased" if prediction == 1 else "Not Biased"
    return render_template(
        "index.html",
        result=result,
        found_words=found_words,
        extracted_text=extracted_text
    )
if __name__ == "__main__":
    app.run(debug=True)
