import streamlit as st
from PIL import Image
import pandas as pd
import folium
from streamlit_folium import st_folium
import tensorflow as tf
import numpy as np

# ========================= PAGE =========================
st.set_page_config(page_title="Souk Guardian 2030", page_icon="Morocco", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.52), rgba(0,0,0,0.62)),
                    url('https://images.unsplash.com/photo-1559925523-10de9e23cf90?w=1920&q=85')
                    no-repeat center center fixed;
        background-size: cover;
    }

   
    h1 {
        font-size: 8.5rem;      
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg,
            #e31e24 0%,
            #e31e24 38%,
            #ffffff 20%,   
            #006400 62%,
            #006400 70%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 50px rgba(0,0,0,0.8);
        letter-spacing: 12px;
        margin: 3rem 0 1.5rem 0;
        line-height: 1.1;
    }

    .tag {
        font-size: 2.7rem ;
        font-weight: 700;
        text-align: center;
        color: #ffffff ;
        text-shadow: 0 0 30px rgba(0,0,0,0.9);
        margin: 1rem 0 5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>SOUK GUARDIAN 2030</h1>", unsafe_allow_html=True)
st.markdown("<div class='tag'>Prends une photo → Découvre le prix juste → Négocie comme un Marocain</div>", unsafe_allow_html=True)

# ========================= DATA =========================
data = {
    "item_en": ["Copper lantern", "Tajine pot", "Argan oil 100ml", "Handwoven scarf",
                "Ceramic plate", "Silver teapot", "Leather bag", "Spice mix 100g", "Small rug 1x1m"],
    "item_ar": ["فانوس نحاسي", "طاجين فخار", "زيت أركان 100مل", "شال منسوج",
                "طبق سيراميك", "تايبوت فضي", "حقيبة جلدية", "توابل 100غ", "زربية صغيرة 1×1م"],
    "min_price": [120, 80, 150, 70, 50, 300, 250, 30, 800],
    "max_price": [220, 180, 280, 150, 120, 600, 550, 80, 1800]
}
df = pd.DataFrame(data)

darija_lines = [
    "هاد الثمن للسياح فقط؟ غالي بزاف!"
]

# ========================= TFLITE MODEL =========================
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path="souk_items_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()

with open("souk_items_labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

def predict_item(img_pil):
    img = img_pil.convert("RGB").resize((224, 224))
    input_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = np.argmax(predictions)
    return labels[idx], float(predictions[idx])

# ========================= UI =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Photo")
    photo = st.camera_input("Prends une photo de l’article", key="cam")

with col2:
    st.subheader("2. Prix demandé")
    price_input = st.text_input("Exemple : 450", placeholder="400", key="price_input")

    st.subheader("3. Article")
    # Auto-detect and pre-select
    default_idx = 0
    photo_to_use = photo or st.session_state.get("photo")

    if photo_to_use:
        try:
            name, conf = predict_item(Image.open(photo_to_use))
            st.success(f"Je détecte → **{name}** ({conf:.1%})")

            if conf >= 0.70:  # 70%+ = auto-select
                clean_name = " ".join([w for w in name.split() if not w.isdigit()]).strip()
                match = df[df["item_en"].str.contains(clean_name.split()[0], case=False, regex=False)]
                if not match.empty:
                    default_idx = int(match.index[0])
                    st.info("Article auto-sélectionné")
        except:
            pass

  
    selected_idx = st.selectbox(
        "Article (auto-sélectionné si photo claire)",
        options=range(len(df)),
        index=default_idx,
        format_func=lambda x: f"{df.iloc[x]['item_en']} – {df.iloc[x]['item_ar']}"
    )

# ========================= ANALYSE =========================
if st.button("Analyser le prix !", type="primary"):
    if not price_input or not price_input.isdigit():
        st.error("Entre un prix en chiffres !")
    else:
        st.session_state.analyzed = True
        st.session_state.price = int(price_input)
        st.session_state.item_idx = selected_idx
        st.session_state.photo = photo
        st.rerun()

# ========================= RESULTS =========================
if st.session_state.get("analyzed"):
    item = df.iloc[st.session_state.item_idx]
    price = st.session_state.price

    st.markdown("---")
    if st.session_state.photo:
        st.image(st.session_state.photo, use_column_width=True)

    st.subheader(f"{item['item_en']} – {item['item_ar']}")

    if price <= item["max_price"]:
        st.success(f"PRIX JUSTE ! Tu peux payer {price} DH")
    elif price <= item["max_price"] * 1.5:
        st.warning(f"Un peu cher… négocie vers {item['max_price']} DH")
    else:
        st.error(f"ARNAQUE ! Prix réel {item['min_price']}–{item['max_price']} DH")
        st.info("Dis-lui en darija → " + darija_lines[price % len(darija_lines)])

    

    if price > item["max_price"]:
        savings = price - item["max_price"]
        st.success(f"Tu économises **{savings} DH**")

    if st.button("Nouvelle analyse"):
        for k in ["analyzed", "price", "item_idx", "photo"]:
            st.session_state.pop(k, None)
        st.rerun()

# ========================= MAP =========================
st.markdown("---")
st.subheader("Zones à Casablanca")
m = folium.Map(location=[33.5731, -7.5898], zoom_start=12, tiles="cartodbpositron")
folium.CircleMarker([33.595, -7.618], radius=40, color="#e74c3c", fill=True,
                    popup="Prix gonflés", tooltip="Médina").add_to(m)
folium.CircleMarker([33.570, -7.585], radius=35, color="#2ecc71", fill=True,
                    popup="Bons vendeurs", tooltip="Derb Ghallef").add_to(m)
st_folium(m, width=700, height=400, key="permanent_map")

st.caption("Souk Guardian 2030 – Ton bouclier anti-arnaque")
