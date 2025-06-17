import os, time, zipfile
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


st.set_page_config(page_icon="🩺", page_title="Skin Disease Detection", layout="wide")
st.markdown("""
<style>
:root { --primary:#001f3f; --accent:#0074D9; --bg:#f5f5f5; }
body { background:var(--bg); }
.sidebar .sidebar-content { background:var(--primary); color:#fff; }
.stButton>button, .stDownloadButton>button {
  background:var(--accent); color:#fff; border-radius:8px; font-weight:bold;
  transition: transform .2s;
}
.stButton>button:hover, .stDownloadButton>button:hover { transform:scale(1.05); }
h1,h2,h3 { color:var(--primary); }
.block-container { padding:2rem !important; }
@keyframes pulse {0%,100%{transform:scale(1);}50%{transform:scale(1.02);}}
img.pulse { animation:pulse 4s infinite; }
</style>
""", unsafe_allow_html=True)

img = (224, 224)
CLASSES = ['Acne', 'Actinic_Keratosis', 'Benign_tumors', 'Bullous', 'Candidiasis', 'DrugEruption', 'Eczema', 'Infestations_Bites', 'Lichen', 'Lupus', 'Moles', 'Psoriasis', 'Rosacea', 'Seborrh_Keratoses', 'SkinCancer', 'Sun_Sunlight_Damage', 'Tinea', 'Unknown_Normal', 'Vascular_Tumors', 'Vasculitis', 'Vitiligo', 'Warts']

# Charge modèle + historique

ZIP_FILENAME = "Hamad_Rassem_Mahamat_SkinDiseaseModel.zip"  
with zipfile.ZipFile(ZIP_FILENAME, "r") as archive:
    h5_files = [f for f in archive.namelist() if f.lower().endswith(".h5")]
    h5_name = h5_files[0]
    data = archive.read(h5_name)
    with open(h5_name, "wb") as out_f: out_f.write(data)
    model = tf.keras.models.load_model(h5_name)
try:
    history1 = pd.read_csv("Hamad_Rassem_Mahamat_HistoryPhase1.csv")
    history2 = pd.read_csv("Hamad_Rassem_Mahamat_HistoryPhase2.csv")
except:
    history1 = pd.read_csv("Hamad_Rassem_Mahamat_History.csv")
    history2 = history1

# Sidebar
st.title("📊 SkinDisease AI ")
st.sidebar.write('''
    # Bienvenue sur votre site de Deep Learning
    Nous prédisons toutes vos maladies de la peau
''')
st.write("\n\n")
tab1, tab2, tab3 = st.tabs(["📈 Training", "🖼️ Prédiction", "🧪 Évaluation"])
st.markdown("---")

with tab1:
    st.write("Visualisez la courbe d’entraînement :")
    st.subheader("Accuracy / Val Accuracy")
    st.line_chart(history1[["accuracy", "val_accuracy"]])
    st.write("\n")
    st.line_chart(history2[["accuracy", "val_accuracy"]])
    st.write("\n")
    st.subheader("Loss / Val Loss")
    st.line_chart(history1[["loss", "val_loss"]])
    st.write("\n")
    st.line_chart(history2[["loss", "val_loss"]])

with tab2:
    st.header("🖼️ Testez votre image")
    upload = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if upload:
        picture = Image.open(upload).convert("RGB").resize(img)
        st.image(picture.resize((1000, int((float(picture.size[1]) * float((350 / float(picture.size[0])))))), Image.FILTERED), use_container_width=False)
        x = np.expand_dims(np.array(picture)/255.0, 0)
        preds = model.predict(x)[0]
        top3 = preds.argsort()[::-1][:5]
        st.write("\n\n")
        st.markdown("### 🎯 Top 5 des prédictions pour votre recherche")
        st.write("\n")
        for i in top3: st.write(f"    - **{CLASSES[i]}** — {preds[i]*100}%")

with tab3:
    st.header("🧪 Évaluation sur le jeu Test")
    try:
        matrice = pd.read_csv("Hamad_Rassem_Mahamat_Matrice.csv", index_col=0)
        fig_cm = px.imshow(matrice, text_auto=True, aspect="auto")
        st.plotly_chart(fig_cm, use_container_width=True)
    except: st.info("Aucune matrice trouvée!")
