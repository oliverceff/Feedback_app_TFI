import streamlit as st
import joblib
import pandas as pd
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Configuration Streamlit
st.set_page_config(page_title="Classificateur Cleary", layout="centered")
st.title("🧠 Classificateur de feedback selon Cleary")

# Connexion à Google Sheets via credentials
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds_dict = json.loads(st.secrets["GOOGLE_SHEETS_CREDENTIALS"])
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)
worksheet = client.open("feedbacks_cleary").sheet1

# Charger les données
data = worksheet.get_all_records()
base_data = pd.DataFrame(data).fillna("")

# Charger modèle et vectoriseur
model = joblib.load("modele_cleary_local.pkl")
vectorizer = joblib.load("vectorizer_cleary_local.pkl")

# Compteur de commentaires à valider
non_classes = base_data[(base_data["Validation humaine"] == "") & (base_data["Niveau Cleary"].str.strip().isin(["", "Non classé"]))]
st.markdown(f"### 🧮 Commentaires non classés restants à valider : {len(non_classes)}")

# Sélection et affichage d’un commentaire non classé
if not non_classes.empty:
    idx = non_classes.index[st.session_state.get("feedback_index", 0) % len(non_classes)]
    feedback_input = base_data.loc[idx, "Feedback"]
    true_label = base_data.loc[idx, "Niveau Cleary"]

    st.markdown("### 💬 Commentaire extrait du Google Sheet :")
    st.info(feedback_input)

    vect = vectorizer.transform([feedback_input])
    prediction_raw = model.predict(vect)[0]

    # Règle hiérarchique manuelle
    text = feedback_input.lower()
    if any(w in text for w in ["s'organiser", "planifier", "répéter", "réviser", "mémoriser", "flashcards", "autoévaluation"]):
        prediction = "Niveau 4 – Autorégulation"
    elif any(w in text for w in ["méthode", "technique", "outil", "processus", "fonctionnalité"]):
        prediction = "Niveau 3 – Processus"
    elif any(w in text for w in ["objectif", "tâche", "activité", "exercice"]):
        prediction = "Niveau 2 – Tâche"
    else:
        prediction = prediction_raw

    # Affichage des niveaux
    st.markdown(f"**📘 Niveau Cleary (fichier)** : {true_label}")
    st.markdown(f"**🤖 Prédiction recalculée** : {prediction}")

    # Validation
    st.markdown("### ✏️ Validation ou correction :")
    corrected = st.selectbox("Quel est le bon niveau selon vous ?", 
        ["", "Niveau 1 – Soi", "Niveau 2 – Tâche", "Niveau 3 – Processus", "Niveau 4 – Autorégulation", "Observation descriptive"])

    col1, col2 = st.columns(2)
    with col1:
        if corrected and st.button("Enregistrer la validation/correction"):
            base_data.at[idx, "Validation humaine"] = corrected
            worksheet.clear()
            worksheet.update([base_data.columns.tolist()] + base_data.values.tolist())
            st.success("✅ Validation enregistrée dans Google Sheets !")
            st.session_state["feedback_index"] = st.session_state.get("feedback_index", 0) + 1
            st.rerun()

    with col2:
        if st.button("Passer"):
            st.session_state["feedback_index"] = st.session_state.get("feedback_index", 0) + 1
            st.rerun()
else:
    st.info("🎉 Tous les commentaires non classés ont été validés !")

# Réentraîner
st.markdown("---")
st.header("🔄 Réentraîner le modèle avec les validations")

if st.button("Réentraîner maintenant"):
    validés = base_data[(base_data["Validation humaine"] != "") | 
                        (base_data["Niveau Cleary"].str.strip().isin(["Niveau 1 – Soi", "Niveau 2 – Tâche", "Niveau 3 – Processus", "Niveau 4 – Autorégulation", "Observation descriptive"]))]
    if validés.empty:
        st.warning("Aucune donnée annotée trouvée.")
    else:
        X = validés["Feedback"]
        y = validés["Validation humaine"].where(validés["Validation humaine"] != "", validés["Niveau Cleary"])

        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
        X_vect = vectorizer.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_vect, y)

        joblib.dump(model, "modele_cleary_local.pkl")
        joblib.dump(vectorizer, "vectorizer_cleary_local.pkl")

        # Réappliquer à tout le fichier
        base_data["Prédiction recalculée"] = model.predict(vectorizer.transform(base_data["Feedback"]))
        worksheet.clear()
        worksheet.update([base_data.columns.tolist()] + base_data.values.tolist())

        st.success("✅ Modèle réentraîné et prédictions mises à jour !")

        # Évaluation
        st.markdown("### 📊 Performances globales du modèle")
        mask = base_data["Niveau Cleary"].str.strip() != ""
        if mask.any():
            y_true = base_data.loc[mask, "Niveau Cleary"]
            y_pred = base_data.loc[mask, "Prédiction recalculée"]
            report = classification_report(y_true, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

        # Compteur par niveau
        st.markdown("### 🔢 Nombre d'exemples validés (objectif : 100)")
        counts = y.value_counts().rename_axis("Niveau").reset_index(name="Exemples")
        counts["Objectif atteint"] = counts["Exemples"] >= 100
        st.dataframe(counts.style.applymap(lambda x: 'background-color: lightgreen' if x else '', subset=["Objectif atteint"]))

        # Graphique des prédictions
        st.markdown("### 📈 Répartition des prédictions recalculées")
        pred_counts = base_data["Prédiction recalculée"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
        st.dataframe(pred_counts)
        st.bar_chart(pred_counts.set_index("Classe"))
