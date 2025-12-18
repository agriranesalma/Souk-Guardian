import streamlit as st
from PIL import Image
import pandas as pd
import folium
from streamlit_folium import st_folium
import tensorflow as tf
import numpy as np
import math

st.set_page_config(page_title="Atlas Trust Ally", page_icon="🇲🇦", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.68), rgba(0,0,0,0.68)),
                    url('https://images.unsplash.com/photo-1559925523-10de9e23cf90?q=80&w=1064&auto=format&fit=crop')
                    no-repeat center center fixed;
        background-size: cover;
        color: white !important;
    }
    h1 {
        font-size: 12rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #e31e24 0%, #e31e24 38%, #ffffff 20%, #006400 62%, #006400 70%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 50px rgba(0,0,0,0.8);
        letter-spacing: 10px;
        margin: 2rem 0 1rem 0;
    }
    .tag {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #ffffff;
        text-shadow: 0 0 30px rgba(0,0,0,0.9);
        margin: 1rem 0 4rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.8rem;
        font-weight: bold;
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Atlas Trust Ally</h1>", unsafe_allow_html=True)
st.markdown("<div class='tag'>Souks + Taxis → Get the Fair Price in Rabat</div>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Souk Bargain Helper", "Taxi Fare Checker"])

# ========================= SOUK TAB =========================
with tab1:
    st.markdown("### Souk Bargain Helper – Never Overpay in the Medina")

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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Photo")
        photo = st.camera_input("Take a clear photo of the item", key="cam")

    with col2:
        st.subheader("2. Asked Price")
        price_input = st.text_input("Ex: 450 DH", placeholder="400", key="price_input")

        st.subheader("3. Item Type")
        default_idx = 0
        photo_to_use = photo or st.session_state.get("photo")

        if photo_to_use:
            try:
                name, conf = predict_item(Image.open(photo_to_use))

                if conf >= 0.90:
                    st.success(f"Detected → **{name}** ({conf:.1%} confidence)")
                    clean_name = " ".join([w for w in name.split() if not w.isdigit()]).strip()
                    match = df[df["item_en"].str.contains(clean_name.split()[0], case=False, regex=False)]
                    if not match.empty:
                        default_idx = int(match.index[0])
                        st.info("Item auto-selected")
                    else:
                        st.warning("Detected item not in list – choose manually")
                else:
                    st.warning("Photo not clear – please choose item manually")
            except:
                st.warning("Photo not clear – please choose item manually")

        selected_idx = st.selectbox(
            "Confirm or choose item",
            options=range(len(df)),
            index=default_idx,
            format_func=lambda x: f"{df.iloc[x]['item_en']} – {df.iloc[x]['item_ar']}"
        )

    if st.button("Check Price!", type="primary"):
        if not price_input or not price_input.isdigit():
            st.error("Please enter a valid price in numbers!")
        else:
            st.session_state.analyzed = True
            st.session_state.price = int(price_input)
            st.session_state.item_idx = selected_idx
            st.session_state.photo = photo
            st.rerun()

    if st.session_state.get("analyzed"):
        item = df.iloc[st.session_state.item_idx]
        price = st.session_state.price

        st.markdown("---")
        if st.session_state.photo:
            st.image(st.session_state.photo, use_column_width=True)

        st.subheader(f"{item['item_en']} – {item['item_ar']}")

        if price <= item["max_price"]:
            st.success(f"FAIR PRICE! You can pay {price} DH")
        elif price <= item["max_price"] * 1.5:
            st.warning(f"A bit high… bargain down to {item['max_price']} DH")
        else:
            st.error(f"TOO EXPENSIVE! Fair range: {item['min_price']}–{item['max_price']} DH")
            st.info("Say in Darija → " + darija_lines[price % len(darija_lines)])

        if price > item["max_price"]:
            savings = price - item["max_price"]
            st.success(f"You save **{savings} DH** by bargaining!")

        if st.button("New analysis"):
            for k in ["analyzed", "price", "item_idx", "photo"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ========================= EDUCATIONAL SOUK MAP (Option 2) =========================
    st.markdown("---")
    st.subheader("Souk Zones – Bargaining Culture in Rabat")

    m_souk = folium.Map(location=[34.0209, -6.8352], zoom_start=14, tiles="cartodbpositron")

    # Blue circle: Main tourist souk – vibrant, bargaining common
    folium.CircleMarker(
        location=[34.0209, -6.8352],
        radius=60,
        color="#3498db",
        fill=True,
        fill_opacity=0.6,
        popup="Main tourist souk – vibrant atmosphere, lots of choice, bargaining is part of the experience",
        tooltip="Medina of Rabat"
    ).add_to(m_souk)

    # Green circle: Local markets – authentic, often fairer prices
    folium.CircleMarker(
        location=[34.0389, -6.8166],
        radius=50,
        color="#2ecc71",
        fill=True,
        fill_opacity=0.6,
        popup="Local markets – authentic experience, often fairer prices and friendly vendors",
        tooltip="Sale Medina"
    ).add_to(m_souk)

    st_folium(m_souk, width=700, height=400, key="souk_map_educational")

# ========================= TAXI TAB =========================
with tab2:
    st.markdown("### Taxi Fare Checker – Fair Taxi Prices in Rabat")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)

    if "taxi_points" not in st.session_state:
        st.session_state.taxi_points = {"depart": None, "arrival": None}

    popular_places = {
        "Rabat-Salé Airport (RBA)": (34.0511, -6.7515),
        "Rabat Ville Train Station": (34.0135, -6.8322),
        "Rabat Agdal Train Station": (33.9990, -6.8550),
        "Prince Moulay Abdellah Stadium": (34.0085, -6.8750),
        "Medina of Rabat": (34.0209, -6.8352),
        "Kasbah of the Udayas": (34.0251, -6.8378),
        "Hassan Tower": (34.0240, -6.8228),
        "Mausoleum of Mohammed V": (34.0238, -6.8225),
        "Chellah Necropolis": (34.0067, -6.8213),
        "Bouregreg Marina": (34.0235, -6.8280),
        "Royal Palace (Dar al-Makhzen)": (34.0158, -6.8431),
        "Andalusian Gardens": (34.0245, -6.8385),
        "Mohammed VI Tower": (34.0220, -6.8280),
        "Agdal District": (34.0020, -6.8560),
        "Hay Riad District": (34.0000, -6.8200),
        "Sale Medina": (34.0389, -6.8166),
        "Mega Mall Rabat": (33.9570, -6.8700),
        "Arribat Center Mall": (33.9810, -6.8700),
        "Café de France (Medina)": (34.0205, -6.8350),
        "Paul Café Rabat": (34.0150, -6.8500),
        "La Comédie Café": (34.0120, -6.8420),
        "Café Maure (Kasbah)": (34.0255, -6.8380),
        "Le Dhow (Bouregreg)": (34.0230, -6.8285),
        "Café Carrion": (34.0155, -6.8340),
        "Café Weimar": (34.0140, -6.8350),
        "Sofitel Rabat Jardin des Roses": (34.0000, -6.8500),
        "Tour Hassan Palace Hotel": (34.0220, -6.8250),
        "Villa Mandarine": (34.0300, -6.8500),
        "Farah Rabat Hotel": (34.0180, -6.8420),
        "Mohammed VI Museum of Modern Art": (34.0180, -6.8350),
        "National Library of Morocco": (34.0080, -6.8480),
        "Rabat Zoo": (33.9500, -6.8900),
        "Faculty of Medicine Rabat (UM5)": (34.0030, -6.8580),
        "International University of Rabat (UIR)": (33.9800, -6.7400),
        "Hôpital Militaire Mohammed V": (34.0120, -6.8280),
        "Hôpital Cheikh Zaid": (34.0000, -6.8200),
        "Bab er-Rouah": (34.0150, -6.8380),
        "Bab Chellah": (34.0070, -6.8210),
        "Avenue Mohammed VI": (34.0100, -6.8500),
        "Restaurant Dinarjat": (34.0210, -6.8360)
    }

    st.info("Select from the long list or click on the map for any location")

    col1, col2 = st.columns(2)
    with col1:
        depart = st.selectbox("Departure", [""] + list(popular_places.keys()), key="depart_rabat")
        if depart:
            st.session_state.taxi_points["depart"] = popular_places[depart]
            st.success(f"Departure: {depart}")
    with col2:
        arrival = st.selectbox("Arrival", [""] + list(popular_places.keys()), key="arrival_rabat")
        if arrival:
            st.session_state.taxi_points["arrival"] = popular_places[arrival]
            st.success(f"Arrival: {arrival}")

    dep_point = st.session_state.taxi_points["depart"]
    arr_point = st.session_state.taxi_points["arrival"]

    center_coords = arr_point or dep_point or (34.0209, -6.8416)

    m_taxi = folium.Map(location=center_coords, zoom_start=13, tiles="cartodbpositron")
    if dep_point:
        folium.Marker(dep_point, tooltip="Departure", icon=folium.Icon(color="red")).add_to(m_taxi)
    if arr_point:
        folium.Marker(arr_point, tooltip="Arrival", icon=folium.Icon(color="green")).add_to(m_taxi)
        if dep_point:
            folium.PolyLine([dep_point, arr_point], color="blue", weight=6).add_to(m_taxi)

    map_data = st_folium(m_taxi, width=700, height=500, key="taxi_map")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        point = (lat, lon)
        if not dep_point:
            st.session_state.taxi_points["depart"] = point
            st.success("Departure set by click!")
        elif not arr_point:
            st.session_state.taxi_points["arrival"] = point
            st.success("Arrival set by click!")
        st.rerun()

    if dep_point and arr_point:
        col1, col2 = st.columns(2)
        with col1:
            taxi_price = st.text_input("Price asked by driver (DH)", placeholder="150")
        with col2:
            night = st.checkbox("Night trip (after 8 PM) +50%")

        if st.button("Check Taxi Fare!", type="primary"):
            if not taxi_price.isdigit():
                st.error("Enter a valid price")
            else:
                distance = haversine(dep_point[0], dep_point[1], arr_point[0], arr_point[1])
                base_price = max(25, 8 + distance * 8)
                fair_price = int(base_price * 1.5) if night else int(base_price)
                asked = int(taxi_price)

                if any(haversine(p[0], p[1], 34.0511, -6.7515) < 10 for p in [dep_point, arr_point]):
                    st.warning("Airport trip? Use grand taxi – fixed price ~250-300 DH")

                st.write(f"**Distance**: {distance} km | **Fair price**: up to **{fair_price} DH**")
                if asked <= fair_price:
                    st.success("FAIR PRICE!")
                elif asked <= fair_price * 1.4:
                    st.warning("A bit high – bargain down")
                else:
                    st.error("OVERPRICED!")
                    st.info("Say in Darija → This price is for tourists only? Too expensive!")

                if asked > fair_price:
                    st.success(f"You can save **{asked - fair_price} DH** by bargaining!")

        if st.button("New taxi check"):
            st.session_state.taxi_points = {"depart": None, "arrival": None}
            st.rerun()

st.markdown("---")
st.caption("Atlas Trust Ally © 2030 – Your shield against possible overpricing in Rabat's souks and taxis 🇲🇦")
