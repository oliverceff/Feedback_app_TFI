import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Charger les données depuis ton fichier
df = pd.read_excel("data.xlsx")

# Garder uniquement les feedbacks classés
df = df[df["Niveau Cleary"] != "Non classé"]
X_texts = df["Feedback"]
y = df["Niveau Cleary"]

# Convertir les textes en vecteurs numériques (TF-IDF)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(X_texts)

# Séparer en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import joblib

# Enregistrer le modèle et le vectoriseur localement
joblib.dump(model, "modele_cleary_local.pkl")
joblib.dump(vectorizer, "vectorizer_cleary_local.pkl")
