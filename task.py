import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

nltk.download('stopwords')



df = pd.read_csv("complaints.csv/complaints.csv")

df = df[['Product', 'Consumer complaint narrative']].dropna()
df.rename(columns={'Consumer complaint narrative': 'text'}, inplace=True)

product_map = {
    'Credit reporting, credit repair services, or other': 0,
    'Debt collection':"Debt collection 1" ,
    'Consumer Loan': "Consumer Loan 2" ,
    'Mortgage': "Mortgage 3" 
}
df['category'] = df['Product'].map(product_map)
df = df.dropna()

sns.countplot(x='category', data=df)
plt.title('Complaint Count by Category')
plt.show()

for cat, label in product_map.items():
    texts = ' '.join(df[df['Product']==cat]['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(cat)
    plt.show()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Multinomial NB': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': None 
}

from xgboost import XGBClassifier
models['XGBoost'] = XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', use_label_encoder=False)

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    results[name] = {'model': model, 'report': classification_report(y_test, y_pred, output_dict=True)}

sample_texts = [
    "I received a call from debt collector for wrong amount",
    "My mortgage payment was misapplied by the bank",
    "I found an error in my credit report",
    "Loan application was rejected unfairly"
]

sample_clean = [clean_text(t) for t in sample_texts]
sample_tfidf = tfidf.transform(sample_clean)

for name, res in results.items():
    preds = res['model'].predict(sample_tfidf)
    print(f"\nPredictions by {name}: {preds}")

df = df[['Product', 'Consumer complaint narrative']].dropna()
df.rename(columns={'Consumer complaint narrative': 'text'}, inplace=True)

product_map = {
    'Credit reporting, credit repair services, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}
df['category'] = df['Product'].map(product_map)
df = df.dropna()

sns.countplot(x='category', data=df)
plt.title('Complaint Count by Category')
plt.show()

for cat, label in product_map.items():
    texts = ' '.join(df[df['Product']==cat]['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texts)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(cat)
    plt.show()

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
)

tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Multinomial NB': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': None  
}

from xgboost import XGBClassifier
models['XGBoost'] = XGBClassifier(objective='multi:softmax', num_class=4, eval_metric='mlogloss', use_label_encoder=False)

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()
    results[name] = {'model': model, 'report': classification_report(y_test, y_pred, output_dict=True)}

sample_texts = [
    "I received a call from debt collector for wrong amount",
    "My mortgage payment was misapplied by the bank",
    "I found an error in my credit report",
    "Loan application was rejected unfairly"
]

sample_clean = [clean_text(t) for t in sample_texts]
sample_tfidf = tfidf.transform(sample_clean)

for name, res in results.items():
    preds = res['model'].predict(sample_tfidf)
    print(f"\nPredictions by {name}: {preds}")
