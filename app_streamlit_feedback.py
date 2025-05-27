import streamlit as st
import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Charger les données de base (data.xlsx)
base_file = "data.xlsx"
if os.path.exists(base_file):
    base_data = pd.read_excel(base_file)
    base_data = base_data.fillna("")
else:
    base_data = pd.DataFrame(columns=["Feedback", "Niveau Cleary", "Validation humaine", "Prédiction recalculée"])

# Charger le modèle et le vectoriseur
model = joblib.load("modele_cleary_local.pkl")
vectorizer = joblib.load("vectorizer_cleary_local.pkl")

st.set_page_config(page_title="Classificateur de feedback Cleary", layout="centered")
st.title("🧠 Classificateur de feedback selon Cleary")

# Affichage du compteur de non classés restants
non_classes = base_data[(base_data["Validation humaine"] == "") & (base_data["Niveau Cleary"].str.strip().isin(["", "Non classé"]))]
st.markdown(f"### 🧮 Commentaires non classés restants à valider : {len(non_classes)}")

# Sélection d'un commentaire uniquement non classé (dans Niveau Cleary)
if not base_data.empty:
    filtered_data = non_classes

    if not filtered_data.empty:
        session_idx = st.session_state.get("feedback_index", 0)
        if session_idx >= len(filtered_data):
            session_idx = 0
        idx = filtered_data.index[session_idx]
        feedback_input = base_data.loc[idx, "Feedback"]
        true_label = base_data.loc[idx, "Niveau Cleary"]

        st.markdown(f"### 💬 Commentaire extrait du fichier Excel :")
        st.info(feedback_input)

        vect = vectorizer.transform([feedback_input])
        prediction_raw = model.predict(vect)[0]
        text = feedback_input.lower()
        if any(word in text for word in ["s'organiser", "prévoir", "anticiper", "planifier", "répéter", "stratégie", "réviser", "mémoriser", "flashcards", "revoir"]):
            prediction = "Niveau 4 – Autorégulation"
        elif any(word in text for word in ["méthode", "processus", "technique", "outil", "fonctionnalité", "organisation"]):
            prediction = "Niveau 3 – Processus"
        elif any(word in text for word in ["objectif", "tâche", "activité", "exercice"]):
            prediction = "Niveau 2 – Tâche"
        else:
            prediction = prediction_raw
        st.markdown(f"**📘 Niveau Cleary (fichier Excel)** : {true_label}")
        st.markdown(f"**🤖 Prédiction recalculée** : {prediction}")

        st.markdown("### ✏️ Validation ou correction :")
        corrected_label = st.selectbox(
            "Quel est le bon niveau selon vous ?",
            ["", "Niveau 1 – Soi", "Niveau 2 – Tâche", "Niveau 3 – Processus", "Niveau 4 – Autorégulation", "Observation descriptive"],
            index=0
        )

        col1, col2 = st.columns(2)
        with col1:
            if corrected_label and st.button("Enregistrer la validation/correction"):
                base_data.at[idx, "Validation humaine"] = corrected_label
                base_data.to_excel(base_file, index=False)
                st.success("✅ Commentaire validé ou corrigé enregistré dans le fichier Excel !")
                st.session_state["feedback_index"] = session_idx + 1
                st.rerun()
        with col2:
            if st.button("Passer"):
                st.session_state["feedback_index"] = session_idx + 1
                st.rerun()
    else:
        st.info("🎉 Tous les commentaires non classés ont été validés !")

# Zone pour réentraîner le modèle à partir des validations
st.markdown("---")
st.markdown("## 🔄 Réentraîner le modèle avec les validations")

if st.button("Réentraîner maintenant"):
    validated_data = base_data[(base_data["Validation humaine"] != "") | (base_data["Niveau Cleary"].str.strip().isin(["Niveau 1 – Soi", "Niveau 2 – Tâche", "Niveau 3 – Processus", "Niveau 4 – Autorégulation", "Observation descriptive"]))]
    if validated_data.empty:
        st.warning("Aucune validation trouvée dans le fichier data.xlsx.")
    else:
        X = validated_data["Feedback"]
        y = validated_data["Validation humaine"].where(validated_data["Validation humaine"] != "", validated_data["Niveau Cleary"])

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
        X_vect = vectorizer.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_vect, y)

        joblib.dump(model, "modele_cleary_local.pkl")
        joblib.dump(vectorizer, "vectorizer_cleary_local.pkl")

        st.success("🚀 Nouveau modèle entraîné et enregistré avec succès à partir des validations !")

        # Réappliquer le modèle à tout le fichier
        all_vect = vectorizer.transform(base_data["Feedback"])
        base_data["Prédiction recalculée"] = model.predict(all_vect)
        base_data.to_excel(base_file, index=False)

        # Évaluer sur tout le fichier si les labels sont connus
        if "Niveau Cleary" in base_data.columns and base_data["Niveau Cleary"].str.strip().any():
            mask = base_data["Niveau Cleary"].str.strip() != ""
            y_true = base_data.loc[mask, "Niveau Cleary"]
            y_pred_total = base_data.loc[mask, "Prédiction recalculée"]
            report_total = classification_report(y_true, y_pred_total, output_dict=True)
            st.markdown("### 📊 Performances du modèle sur l'ensemble du fichier")
            st.dataframe(pd.DataFrame(report_total).transpose().style.format("{:.2f}"))

        # Compteur par niveau avec objectif
        st.markdown("### 🔢 Nombre d'exemples validés par niveau (objectif : 100)")
        counts = y.value_counts().rename_axis("Niveau").reset_index(name="Nombre d'exemples")
        counts["Objectif atteint"] = counts["Nombre d'exemples"] >= 100
        st.dataframe(counts.style.applymap(lambda x: 'background-color: lightgreen' if x is True else '', subset=["Objectif atteint"]))

        # Répartition des prédictions recalculées
        st.markdown("### 📈 Répartition actuelle des prédictions recalculées")
        pred_counts = base_data["Prédiction recalculée"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
        st.dataframe(pred_counts)
        st.bar_chart(pred_counts.set_index("Classe"))
