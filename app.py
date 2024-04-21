from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from io import StringIO
import nltk
import re
from googletrans import Translator
from langdetect import detect
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

nltk.download('punkt')
nltk.download('stopwords')

translator = Translator()
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

def translate_if_needed(text):
    try:
        language = detect(text)
        if language == 'en':
            return translator.translate(text, src='en', dest='es').text
        return text
    except Exception as e:
        return text  

def normalize_numbers(text):
    return re.sub(r'\b\d+(?:\.\d+)?\b', 'NUM', text)

def preprocessing(texts):
    processed_texts = [translate_if_needed(text) for text in texts]
    processed_texts = [normalize_numbers(text) for text in processed_texts]
    processed_texts = [' '.join(nltk.word_tokenize(text)) for text in processed_texts]
    processed_texts = [' '.join([word.lower() for word in text.split() if word not in stop_words]) for text in processed_texts]
    processed_texts = [' '.join([stemmer.stem(word) for word in text.split()]) for text in processed_texts]
    return processed_texts


modelo = joblib.load('model.joblib')  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    return render_template('retrain.html')  

@app.route('/cliente')
def cliente():
    return render_template('cliente.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        content = file.stream.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        processed_texts = preprocessing(df['Review'].tolist())  
        predictions = modelo.predict(processed_texts)
        results = list(zip(df['Review'].tolist(), predictions)) 
        return render_template('result.html', results=results)
    return 'Archivo inválido o formato de archivo incorrecto', 40

@app.route('/retrain', methods=['POST'])
def retrain():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        content = file.stream.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        processed_texts = preprocessing(df['Review'].tolist())
        modelo = joblib.load('model.joblib')
        modelo.fit(processed_texts, df['Class'])
        joblib.dump(modelo, 'model.joblib')  
        return 'Model retrained successfully', 200
    return 'Invalid file or file format', 400

if __name__ == '__main__':
    app.run(debug=True)
