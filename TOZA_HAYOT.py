import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from PIL import Image
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import lightgbm
import base64
from tensorflow.keras.preprocessing.image import load_img, img_to_array  
from io import BytesIO
import pickle
from tensorflow.keras.models import load_model
import math
import google.generativeai as genai
import os
from datetime import datetime
import datetime
import uuid
import json
# Mahalliy rasmni Base64 formatga o'tkazish
def get_base64_of_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

image_base64 = get_base64_of_image("keraklilar/fonrasm2.jpg") 
# Set page config
st.set_page_config(
    page_title="Agricultural Expert System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)
with open("keraklilar/fonrasm2.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    
# Custom CSS for better styling
st.markdown("""
    <body style="background-color:powderblue";>
    <style>
        h1 {
            font-size: 40px !important;
            color: #111 !important; /* To‘q qora */
            bold: 1px;
        }
        h2 {
            font-size: 30px !important;
            color: #111 !important; /* To‘q qora */
        }
        /* Subheader (h3) uchun yangi uslub */
        h3 {
            font-size: 24px !important;
            color: #111 !important; /* To‘q qora rang */
            font-weight: bold !important;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6) !important; /* Engil qora soya */
            font-family: 'Arial', sans-serif !important;
            background-color: rgba(255, 255, 255, 0.8) !important; /* Shaffof oq fon */
            padding: 8px 12px !important; /* Ichki bo‘shliq */
            border-radius: 5px !important; /* Yumaloq burchaklar */
            display: inline-block !important; /* Matn uzunligiga mos quti */
        }
        p, label {
            font-size: 18px !important;
            color: #111 !important; /* To‘q qora */
        }
        

     .stApp {
            background-color: #FFFFFF !important; /* Oq fon */
            color: black !important;    /* Oq matn */
            font-family: 'Arial', sans-serif;
            background-image:linear-gradient(rgba(255, 255, 255, 0.5), rgba(255, 255, 255, 0.5)),url("https://previews.123rf.com/images/belchonock/belchonock1711/belchonock171101825/90496546-icons-and-field-on-background-concept-of-smart-agriculture-and-modern-technology.jpg");
            background-size: 100% 100%; /* Width 100%, height proportional */
            background-position: right;
            
        }
    /* Yangi klass qo‘shildi */
        .large-text {
            font-size: 24px !important; /* Matn o‘lchami */
            text-align: center !important; /* Markazga tekislash */
        }
    [data-testid="stSidebar"] button {
        width: 100% !important;
        height: 50px !important;
        font-size: 16px !important;
        text-align: center !important;
        background-color: #4CAF50 !important;  /* Yashil */
        color: white !important;
        border-radius: 20px 5px 5px 20px !important;  /* Pastki chap va yuqori o‘ng burchak torroq */
        border: none !important;
        transition: background-color 0.3s ease-in-out, border-radius 0.3s ease-in-out;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #32a5a5 !important;  /* Yashil-moviy aralash rang */
        border-radius: 5px 20px 20px 5px !important;  /* Yuqori chap va pastki o‘ng burchak torayadi */
    }
    [data-testid="stSidebar"] {
        background-image: url("https://your-image-url.com/image.jpg");
        background-size: cover;
    }
    .main-header {
        font-size: 50px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 40px;
    }
    .section-header {
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f1f8e9;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #7cb342;
    }
    
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown("<div class='main-header'>Agricultural Expert System</div>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "1"  # Default to the first page

# Sidebar buttons
with st.sidebar:
    st.markdown('<div class="sidebar-buttons">', unsafe_allow_html=True)
    if st.button("O'simlik kasalliklarini aniqlash"):
        st.session_state.page = "1"
    if st.button("Sug'orish"):
        st.session_state.page = "2"
    if st.button("Tuproq"):
        st.session_state.page = "3"
    if st.button("Xizmatlar"):
        st.session_state.page = "4"
    if st.button("Izohlar"):
        st.session_state.page = "5"
    if st.button("AI yordamchi"):
        st.session_state.page = "6"

#Sidebarga logo qo'shish

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #333;  /* Sidebar orqa foni */
            padding-top: 20px;
        }
        .sidebar-logo {
            display: flex;
            justify-content: center;
            align-items: center;
            padding-bottom: 10px;
        }
        .sidebar-logo img {
            width: 150px;  /* LOGO o‘lchami */
            height: auto;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
 
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">
            <img src="https://png.pngtree.com/element_our/sm/20180411/sm_5ace0628840fa.png" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )



# Soil types and recommendations
soil_types = {
    "Sandy": {"pH": "5.5-7.0", "Water Retention": "Low", "Suitable Crops": "Carrots, Potatoes, Radishes", 
              "Amendments": "Add organic matter, compost, and clay to improve water retention"},
    "Clay": {"pH": "5.5-7.5", "Water Retention": "High", "Suitable Crops": "Broccoli, Cabbage, Beans", 
             "Amendments": "Add organic matter and sand to improve drainage"},
    "Loamy": {"pH": "6.0-7.0", "Water Retention": "Medium", "Suitable Crops": "Most crops thrive", 
              "Amendments": "Add organic matter yearly to maintain fertility"},
    "Silty": {"pH": "6.0-7.0", "Water Retention": "Medium-High", "Suitable Crops": "Most vegetables, fruit trees", 
              "Amendments": "Add compost to improve structure"},
    "Peaty": {"pH": "3.0-5.0", "Water Retention": "High", "Suitable Crops": "Blueberries, Cranberries", 
              "Amendments": "Add lime to raise pH and nitrogen fertilizer"}
}

# Leaf diseases database
leaf_diseases = {
    "Tomato Late Blight": {
        "Symptoms": "Dark brown spots on leaves with white fungal growth underneath", 
        "Cause": "Fungus (Phytophthora infestans)",
        "Treatment": "Remove infected plants, apply copper-based fungicide, ensure proper spacing for air circulation"
    },
    "Powdery Mildew": {
        "Symptoms": "White powdery coating on leaves and stems", 
        "Cause": "Fungal pathogens",
        "Treatment": "Apply sulfur-based fungicide, neem oil, or potassium bicarbonate sprays"
    },
    "Corn Rust": {
        "Symptoms": "Small reddish-brown pustules on leaves", 
        "Cause": "Fungus (Puccinia sorghi)",
        "Treatment": "Plant resistant varieties, apply fungicide early in the season"
    },
    "Bacterial Leaf Spot": {
        "Symptoms": "Dark, angular spots on leaves with yellow halos", 
        "Cause": "Various bacteria",
        "Treatment": "Remove infected leaves, avoid overhead watering, apply copper-based bactericide"
    },
    "Healthy": {
        "Symptoms": "No symptoms of disease", 
        "Cause": "N/A",
        "Treatment": "Continue regular care and monitoring"
    }
}

# Plant care guide
plant_care = {
    "Tomatoes": {
        "Watering": "Regular watering, 1-2 inches per week",
        "Sunlight": "Full sun (6+ hours daily)",
        "Fertilizer": "Balanced fertilizer (10-10-10) at planting, then high phosphorus during fruiting",
        "Special Care": "Stake or cage plants, prune suckers for indeterminate varieties"
    },
    "Corn": {
        "Watering": "1-2 inches per week, critical during silking",
        "Sunlight": "Full sun (6+ hours daily)",
        "Fertilizer": "High nitrogen at planting, side-dress when plants are knee-high",
        "Special Care": "Plant in blocks for proper pollination, not rows"
    },
    "Lettuce": {
        "Watering": "Consistent moisture, avoid wilting",
        "Sunlight": "Partial shade in hot weather",
        "Fertilizer": "Nitrogen-rich fertilizer before planting",
        "Special Care": "Succession planting every 2 weeks for continuous harvest"
    },
    "Beans": {
        "Watering": "1 inch per week, avoid wetting foliage",
        "Sunlight": "Full sun (6+ hours daily)",
        "Fertilizer": "Low nitrogen, as beans fix their own",
        "Special Care": "Provide trellises for pole varieties"
    },
    "Potatoes": {
        "Watering": "1-2 inches per week, critical during flowering",
        "Sunlight": "Full sun (6+ hours daily)",
        "Fertilizer": "Balanced fertilizer at planting, avoid excess nitrogen",
        "Special Care": "Hill soil around plants as they grow"
    }
}

# Latest news on seasonal diseases (example data)
seasonal_news = [
    {
        "title": "Early Blight Alert for Tomato Growers",
        "date": "March 1, 2025",
        "content": "Early blight (Alternaria solani) has been reported in several regions following the recent warm and humid conditions. Symptoms include dark spots with concentric rings on lower leaves. Preventative measures include proper spacing, avoiding overhead watering, and applying fungicides if necessary.",
        "regions_affected": "Midwest, Southeast"
    },
    {
        "title": "Wheat Rust on the Rise",
        "date": "February 15, 2025",
        "content": "Wheat rust infections are increasing due to mild winter conditions. Farmers are advised to scout fields regularly and consider fungicide applications at the first sign of infection.",
        "regions_affected": "Great Plains, Pacific Northwest"
    },
    {
        "title": "New Resistant Corn Variety Released",
        "date": "February 5, 2025",
        "content": "Agricultural scientists have released a new corn variety with improved resistance to common rust and northern leaf blight. Seeds will be available for the next planting season.",
        "regions_affected": "Nationwide"
    }
]





# Leaf Disease Detection
if st.session_state.page == "1":
    # st.markdown("<h2 class='section-header'>Leaf Disease Detection</h2>", unsafe_allow_html=True)
    st.title("Barg orqali kasalliklarni aniqlash")
    model = load_model("keraklilar\improved_cotton_model.h5")  # Replace with your model path

    # Function to preprocess the image (same as before)
    def prepare_image(uploaded_file, target_size=(224, 224)):
        # Vaqtinchalik faylga saqlash
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Tasvirni yuklash va tayyorlash
        img = load_img("temp_image.jpg", target_size=target_size)  # `image.load_img` o‘rniga `load_img`
        img_array = img_to_array(img)  # `image.img_to_array` o‘rniga `img_to_array`
        img_array = img_array / 255.0  # Normalizatsiya
        img_array = np.expand_dims(img_array, axis=0)  # Batch o'lchamini qo'shish
        return img_array

    # Class labels (same as before)
    class_labels = {
        0: "Aphids",
        1: "Army worm",
        2: "Bacterial blight",
        3: "Healthy",
        4: "Powdery mildew",
        5: "Target spot"
    }

    # Streamlit UI
    st.header("Paxta bargi kasalliklarini bashorat qilish")

    # File uploader
    uploaded_file = st.file_uploader("Rasm joylang...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Siz yuklagan rasm", use_container_width=True)

        # Preprocess the image
        img_input = prepare_image(uploaded_file)

        # Make prediction
        predictions = model.predict(img_input)
        max_prob = np.max(predictions[0])
        threshold = 0.7

        # Display prediction results
        if max_prob < threshold:
            st.write("**Prediction:** Barg topilmadi, rasm sifatini yaxshilang Yoki bu kasallik turi train qilinmagan")
        else:
            predicted_index = int(np.argmax(predictions[0]))
            predicted_disease = class_labels[predicted_index]
            st.write(f"**Prediction:** {predicted_disease}")
            st.write(f"**Confidence:** {max_prob:.2f}")
            if predicted_disease=="Aphids":
                st.subheader("Quyidagi vositalarni sinab ko'ring:")
                st.write("Mahsulot nomi: Batalyon insektitsidi , Texnik tarkib :   Tiametoksam 25% WG, Dozalash:  0,5 g / litr suv  ")
                st.divider()
                st.write("Mahsulot nomi: Tafgor insektitsidi , Texnik tarkib :   Dimetoat 30% EC , Dozalash:  2 ml / litr suv   ")
            elif predicted_disease=="Army worm":
                st.subheader("Quyidagi vositalarni sinab ko'ring:")
                st.write("Mahsulot nomi: Batalyon insektitsidi , Texnik tarkib :   Tiametoksam 25% WG, Dozalash:  0,5 g / litr suv  ")
                st.divider()
                st.write("Mahsulot nomi: Tafgor insektitsidi , Texnik tarkib :   Dimetoat 30% EC , Dozalash:  2 ml / litr suv   ")
            elif predicted_disease=="Bacterial blight":
                st.subheader("Quyidagi vositalarni sinab ko'ring:")
                st.write("Mahsulot nomi: Batalyon insektitsidi , Texnik tarkib :   Tiametoksam 25% WG, Dozalash:  0,5 g / litr suv  ")
                st.divider()
                st.write("Mahsulot nomi: Tafgor insektitsidi , Texnik tarkib :   Dimetoat 30% EC , Dozalash:  2 ml / litr suv   ")
            elif predicted_disease=="Powdery mildew":
                st.subheader("Quyidagi vositalarni sinab ko'ring:")
                st.write("Mahsulot nomi: Batalyon insektitsidi , Texnik tarkib :   Tiametoksam 25% WG, Dozalash:  0,5 g / litr suv  ")
                st.divider()
                st.write("Mahsulot nomi: Tafgor insektitsidi , Texnik tarkib :   Dimetoat 30% EC , Dozalash:  2 ml / litr suv   ")   
            elif predicted_disease=="Target spot":
                st.subheader("Quyidagi vositalarni sinab ko'ring:")
                st.write("Mahsulot nomi: Batalyon insektitsidi , Texnik tarkib :   Tiametoksam 25% WG, Dozalash:  0,5 g / litr suv  ")
                st.divider()
                st.write("Mahsulot nomi: Tafgor insektitsidi , Texnik tarkib :   Dimetoat 30% EC , Dozalash:  2 ml / litr suv   ")
elif st.session_state.page =="2":
    
    
    
    @st.cache_data
    def model(path):

        #Dataframe ga yuklash jarayoni
        df=pd.read_csv(path)

        # LabelEncoder ni qo'llash
        le = LabelEncoder()
        CropType_encode = le.fit_transform(df['CropType'])

        df1=df.drop(['CropType'],axis=1)
        df1.insert(0, 'Crop_encode', CropType_encode)
        df1 = df1.sample(frac=1).reset_index(drop=True)
        #Train va test setga ajratish
        x=df1.drop(['Irrigation'],axis=1)
        y=df['Irrigation']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


        # Model yaratish
        dt_model = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=42)

        # Modelni o‘qitish
        dt_model.fit(x_train, y_train)

        # Test ma'lumotlari bo‘yicha bashorat qilish
        # y_pred = dt_model.predict(x_test)

        # Model aniqligini tekshirish
        # accuracy = accuracy_score(y_test, y_pred)
        # print(f"Accuracy: {accuracy * 100:.2f}%")
        return dt_model

    st.title('Sug‘orish')
    with st.form("Fill the form"):
        croptype=st.selectbox("O'simlik turini tanlang",['Wheat', 'Groundnuts', 'Garden Flowers', 'Maize', 'Paddy',
            'Potato', 'Pulse', 'Sugarcane', 'Coffee'])
        hosil={'Wheat': 8, 'Groundnuts': 2, 'Garden Flowers': 1, 'Maize': 3, 'Paddy': 4,
            'Potato': 5, 'Pulse': 6, 'Sugarcane': 7, 'Coffee': 0}

        plant_age=st.number_input("O'simlik necha kunlik?",min_value=0,max_value=300)

        soil_moisture=st.number_input("Tuproq namligi",min_value=0,max_value=1000)

        temperature=st.number_input("Havo harorati",min_value=-10,max_value=60)

        humidity=st.number_input("Havo namligi",min_value=0,max_value=100)
        submitted=st.form_submit_button("Tasdiqlash",type='primary')

    model=model('D:\VS code\greenland\datasets - datasets.csv')

    y_pred=model.predict([[hosil[croptype],plant_age,soil_moisture,temperature,humidity]])
    if submitted:
        if y_pred==1:
            st.markdown("<p class='large-text'><b>Sug'orish zarur</b></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='large-text'><b>Sug'orish zarur emas</b></p>", unsafe_allow_html=True)
            
elif st.session_state.page =="3":
    if 'button' not in st.session_state:
        st.session_state.button = "main"  # Dastlabki holat sifatida "main" qo‘ydim

    # Asosiy sahifa
    col1, col2 = st.columns(2)
    with col1:
        button1 = st.button('Tuproq unumdorligini bashorat qilish', type='primary')
    with col2:
        button2 = st.button('Tuproq uchun mos keluvchi o‘simlik', type='primary')

    # Tugmalar bosilganda holatni o‘zgartirish
    if button1:
        st.session_state.button = "fertility"
    elif button2:
        st.session_state.button = "crop"

    # Tuproq unumdorligini bashorat qilish bo‘limi
    if st.session_state.button == "fertility":
        st.title("Tuproq unumdorligini bashorat qilish")

        # Modelni yuklash
        model = joblib.load("soil_fertility.pkl")

        # Optimal chegaralar
        land_size = st.number_input("Yer o‘lchamini kiriting (sotix):", value=1)
        optimal_thresholds_1 = {"NITROGIN": 250, "FOSFOR": 50, "KALIY": 180, "PH qiymati": 7.0}
        optimal_thresholds_2 = {"ELEKTR O'TKAZUVCHANLIK": 0.7, "ORGANIK MODDA": 1.5, "OLTINGUGURT": 15, "RUX": 1.0}
        optimal_thresholds_3 = {"TEMIR": 3.5, "MIS": 1.2, "MARGANETS": 2.0, "BORN": 0.5}
        optimal_thresholds = {**optimal_thresholds_1, **optimal_thresholds_2, **optimal_thresholds_3}

        # Input maydonlari
        col1, col2, col3 = st.columns(3)
        new_data = {}
        with col1:
            for nutrient in optimal_thresholds_1:
                new_data[nutrient] = st.number_input(f"{nutrient} qiymatini kiriting:", value=optimal_thresholds_1[nutrient])
        with col2:
            for nutrient in optimal_thresholds_2:
                new_data[nutrient] = st.number_input(f"{nutrient} qiymatini kiriting:", value=optimal_thresholds_2[nutrient])
        with col3:
            for nutrient in optimal_thresholds_3:
                new_data[nutrient] = st.number_input(f"{nutrient} qiymatini kiriting:", value=optimal_thresholds_3[nutrient])

        # Bashorat qilish
        if st.button("Bashorat qilish", type='primary'):
            new_df = pd.DataFrame([new_data])
            prediction = model.predict(new_df)[0]
            category_mapping = {0: "Past unumdorlik/sifat", 1: "O‘rta unumdorlik/sifat", 2: "Yuqori unumdorlik/sifat"}
            st.write(f"**Bashorat:** {category_mapping[prediction]}")

            # Ehtimolliklarni ko‘rsatish
            pred_proba = model.predict_proba(new_df)[0]
            for category, proba in enumerate(pred_proba):
                st.write(f"{category_mapping[category]} uchun ehtimollik: {proba * 100:.2f}%")

            # Yetishmayotgan o‘g‘itlarni aniqlash
            def check_nutrient_deficiency(data, thresholds):
                deficiencies = {}
                for nutrient, value in data.items():
                    if nutrient in thresholds and value < thresholds[nutrient]:
                        deficiencies[nutrient] = {"current": value, "optimal": thresholds[nutrient]}
                return deficiencies

            st.subheader("Tuproq uchun yetishmayotgan o‘g‘itlar")
            deficiencies = check_nutrient_deficiency(new_data, optimal_thresholds)
            if deficiencies:
                st.write("**Yer uchun yetishmayotgan o‘g‘it tavsiyalari:**")
                fertilizer_rate = {"NITROGIN": 0.4, "FOSFOR": 0.2, "KALIY": 0.1}
                for nutrient, values in deficiencies.items():
                    if nutrient in fertilizer_rate:
                        deficit = values["optimal"] - values["current"]
                        required_fertilizer = deficit * fertilizer_rate[nutrient] * land_size
                        st.write(f"{nutrient} uchun: {required_fertilizer:.2f} kg")
            else:
                st.write("Tuproqda yetishmayotgan o‘g‘itlar yo‘q.")

    # Tuproq uchun mos keluvchi o‘simlik bo‘limi
    elif st.session_state.button == "crop":
        with open("keraklilar/plant_suggestion.pkl", "rb") as f:
            model = pickle.load(f)

        # O‘simlik tavsiyasi funksiyasi
        def get_crop_recommendations(N, P, K, temperature, humidity, ph, rainfall):
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            crop_probabilities = dict(zip(model.classes_, probabilities))
            sorted_crops = sorted(crop_probabilities.items(), key=lambda item: item[1], reverse=True)
            top_3_crops = [crop[0] for crop in sorted_crops[:3]]
            return prediction, top_3_crops

        # CSS stil
        st.markdown(
            """
            <style>
            input {border: 2px solid #008000 !important; background-color: #90EE90 !important; border-radius: 5px; padding: 5px; color: black;}
            .stApp {background-color: #f8f8f8; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);}
            .stButton>button {background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px;}
            .stSuccess {background-color: #d4edda; border-color: #c3e6cb; color: #155724; padding: 10px; border-radius: 4px;}
            .stInfo {background-color: #d1ecf1; border-color: #bee2e6; color: #0c5460; padding: 10px; border-radius: 4px;}
            </style>
            """,
            unsafe_allow_html=True
        )

        st.title("Tuproq uchun mos keluvchi o‘simlik 🌱")
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
        P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
        K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=25.0)
        humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=60.0)
        ph = st.number_input("pH", min_value=3.5, max_value=9.9, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0)

        if st.button("Predict", type="primary"):
            predicted_crop, top_3_crops = get_crop_recommendations(N, P, K, temperature, humidity, ph, rainfall)
            st.success(f"**Most Suitable Crop:** {predicted_crop}")
            st.info(f"**Top 3 Recommendations:** {', '.join(top_3_crops)}")

    # Agar hech qaysi tugma bosilmagan bo‘lsa
    else:
        st.write("Iltimos, yuqoridagi tugmalardan birini tanlang.")
                
                
elif st.session_state.page =="4":
    
    # Dastur sarlavhasi
    st.title("Qishloq xo'jaligi xizmatlari")
    st.markdown("Bu dastur sizga yaqin atrofdagi qishloq xo'jaligi xizmatlarini topishga yordam beradi.")

    # Foydalanuvchi joylashuvini olish
    user_location = st.text_input("Joylashuvingizni kiriting (masalan: Toshkent, Chilonzor)", "Toshkent")

    # Ma'lumotlar bazasi (sintetik ma'lumotlar)
    data = {
        "tuproq_tahlili": [
            {
                "id": 1,
                "nomi": "BioLab",
                "manzil": "Toshkent sh., Olmazor tumani, Universitet ko'chasi 4",
                "telefon": "+998 71 123-45-67",
                "tahlil_turlari": ["kimyoviy", "fizik", "biologik"],
                "narx": "100,000 - 300,000 so'm",
                "muddat": "3-5 kun",
                "lokatsiya": {"lat": 41.34, "lon": 69.28}
            },
            {
                "id": 2,
                "nomi": "AgriTest",
                "manzil": "Samarqand sh., Registon ko'chasi 12",
                "telefon": "+998 66 234-56-78",
                "tahlil_turlari": ["kimyoviy", "fizik"],
                "narx": "80,000 - 250,000 so'm",
                "muddat": "2-4 kun",
                "lokatsiya": {"lat": 39.65, "lon": 66.96}
            },
            {
                "id": 3,
                "nomi": "FermerLab",
                "manzil": "Namangan sh., Davlatabod tumani, Bobur ko'chasi 15",
                "telefon": "+998 69 345-67-89",
                "tahlil_turlari": ["kimyoviy", "mikrobiologik"],
                "narx": "120,000 - 350,000 so'm",
                "muddat": "4-7 kun",
                "lokatsiya": {"lat": 41.00, "lon": 71.67}
            }
        ],
        
        "ekspertlar": [
            {
                "id": 1,
                "ism": "Abdullayev Akmal",
                "ixtisoslik": ["g'alla ekinlari", "meva daraxtlari"],
                "tajriba": "15 yil",
                "telefon": "+998 90 123-45-67",
                "email": "akmal@agroconsult.uz",
                "konsultatsiya": ["onlayn", "shaxsan"],
                "narx": "200,000 so'm/soat",
                "lokatsiya": {"lat": 41.32, "lon": 69.24}
            },
            {
                "id": 2,
                "ism": "Karimova Nilufar",
                "ixtisoslik": ["sabzavotlar", "issiqxona ekinlari"],
                "tajriba": "10 yil",
                "telefon": "+998 91 234-56-78",
                "email": "nilufar@agronomist.uz",
                "konsultatsiya": ["onlayn"],
                "narx": "180,000 so'm/soat",
                "lokatsiya": {"lat": 40.10, "lon": 67.84}
            },
            {
                "id": 3,
                "ism": "Rahimov Jahongir",
                "ixtisoslik": ["paxta", "g'o'za kasalliklari"],
                "tajriba": "20 yil",
                "telefon": "+998 93 345-67-89",
                "email": "jahongir@cotton-expert.uz",
                "konsultatsiya": ["onlayn", "shaxsan", "dala tashxisi"],
                "narx": "250,000 so'm/soat",
                "lokatsiya": {"lat": 40.78, "lon": 72.35}
            }
        ],
        
        "agrovet_dokonlari": [
            {
                "id": 1,
                "nomi": "AgroMart",
                "manzil": "Toshkent sh., Yunusobod tumani, Amir Temur ko'chasi 105",
                "telefon": "+998 71 234-56-78",
                "mahsulot_turlari": ["o'g'itlar", "urug'lar", "pestitsidlar", "qishloq xo'jaligi texnikasi"],
                "ish_vaqti": "09:00 - 18:00",
                "lokatsiya": {"lat": 41.36, "lon": 69.29}
            },
            {
                "id": 2,
                "nomi": "FermerDukon",
                "manzil": "Andijon sh., Bog'ishamol ko'chasi 45",
                "telefon": "+998 74 345-67-89",
                "mahsulot_turlari": ["o'g'itlar", "urug'lar", "pestitsidlar"],
                "ish_vaqti": "08:00 - 19:00",
                "lokatsiya": {"lat": 40.78, "lon": 72.34}
            },
            {
                "id": 3,
                "nomi": "AgriTools",
                "manzil": "Farg'ona sh., Mustaqillik ko'chasi 18",
                "telefon": "+998 73 456-78-90",
                "mahsulot_turlari": ["qishloq xo'jaligi texnikasi", "ehtiyot qismlar", "o'g'itlar"],
                "ish_vaqti": "08:30 - 17:30",
                "lokatsiya": {"lat": 40.38, "lon": 71.78}
            }
        ]
    }

    # Masofani hisoblash funksiyasi (oddiy yaqinlashish)
    def calculate_distance(loc1, loc2):
        # Oddiy masofani hisoblash (aniq bo'lmasa ham)
        dx = (loc1["lat"] - loc2["lat"]) * 111
        dy = (loc1["lon"] - loc2["lon"]) * 85
        return math.sqrt(dx*dx + dy*dy)

    # Foydalanuvchi uchun joylashuv (haqiqiy joylashuv o'rniga oddiy misol)
    user_locations = {
        "Toshkent": {"lat": 41.3, "lon": 69.24},
        "Samarqand": {"lat": 39.65, "lon": 66.96},
        "Namangan": {"lat": 41.0, "lon": 71.67},
        "Andijon": {"lat": 40.78, "lon": 72.34},
        "Buxoro": {"lat": 39.77, "lon": 64.42},
        "Farg'ona": {"lat": 40.38, "lon": 71.78}
    }

    # Foydalanuvchi joylashuvini aniqlash
    user_loc = {"lat": 41.3, "lon": 69.24}  # standart qiymat
    for city, loc in user_locations.items():
        if city.lower() in user_location.lower():
            user_loc = loc
            break

    # Xizmat turini tanlash
    st.subheader("Xizmat turini tanlang")
    service_type = st.radio(
        "Qanday xizmat kerak?",
        ["Tuproq tahlili laboratoriyasi", "O'simlik kasalligi eksperti", "Agrovet do'koni"]
    )

    # Tanlangan xizmat ma'lumotlarini ko'rsatish
    if st.button("Xizmat tanlash"):
        if service_type == "Tuproq tahlili laboratoriyasi":
            st.subheader("Tuproq tahlili laboratoriyalari")
            
            # Ma'lumotlarni tayyorlash
            labs = data["tuproq_tahlili"]
            
            # Har bir laboratoriyaga masofa qo'shish
            for lab in labs:
                lab["masofa"] = round(calculate_distance(user_loc, lab["lokatsiya"]), 1)
            
            # Masofaga qarab saralash
            labs = sorted(labs, key=lambda x: x["masofa"])
            
            # DataFramega o'tkazish
            df_data = []
            for lab in labs:
                df_data.append({
                    "Nomi": lab["nomi"],
                    "Manzil": lab["manzil"],
                    "Telefon": lab["telefon"],
                    "Tahlil turlari": ", ".join(lab["tahlil_turlari"]),
                    "Narx": lab["narx"],
                    "Muddat": lab["muddat"],
                    "Masofa (km)": lab["masofa"]
                })
            
            df = pd.DataFrame(df_data)
            st.table(df)
        
        elif service_type == "O'simlik kasalligi eksperti":
            st.subheader("O'simlik kasalligi ekspertlari")
            
            # Ma'lumotlarni tayyorlash
            experts = data["ekspertlar"]
            
            # Har bir ekspertga masofa qo'shish
            for expert in experts:
                expert["masofa"] = round(calculate_distance(user_loc, expert["lokatsiya"]), 1)
            
            # Masofaga qarab saralash
            experts = sorted(experts, key=lambda x: x["masofa"])
            
            # DataFramega o'tkazish
            df_data = []
            for expert in experts:
                df_data.append({
                    "Ism": expert["ism"],
                    "Ixtisoslik": ", ".join(expert["ixtisoslik"]),
                    "Tajriba": expert["tajriba"],
                    "Telefon": expert["telefon"],
                    "Email": expert["email"],
                    "Konsultatsiya": ", ".join(expert["konsultatsiya"]),
                    "Narx": expert["narx"],
                    "Masofa (km)": expert["masofa"]
                })
            
            df = pd.DataFrame(df_data)
            st.table(df)
        
        elif service_type == "Agrovet do'koni":
            st.subheader("Agrovet do'konlari")
            
            # Ma'lumotlarni tayyorlash
            shops = data["agrovet_dokonlari"]
            
            # Har bir do'konga masofa qo'shish
            for shop in shops:
                shop["masofa"] = round(calculate_distance(user_loc, shop["lokatsiya"]), 1)
            
            # Masofaga qarab saralash
            shops = sorted(shops, key=lambda x: x["masofa"])
            
            # DataFramega o'tkazish
            df_data = []
            for shop in shops:
                df_data.append({
                    "Nomi": shop["nomi"],
                    "Manzil": shop["manzil"],
                    "Telefon": shop["telefon"],
                    "Mahsulot turlari": ", ".join(shop["mahsulot_turlari"]),
                    "Ish vaqti": shop["ish_vaqti"],
                    "Masofa (km)": shop["masofa"]
                })
            
            df = pd.DataFrame(df_data)
            st.table(df)

    # Dastur haqida ma'lumot
    st.markdown("---")
    st.markdown("<p class='large-text'><b>Ushbu sahifa qishloq xo'jaligi sohasidagi xizmatlarni topishga yordam berish uchun yaratilgan</b></p>", unsafe_allow_html=True)
    #st.caption("Ushbu dastur qishloq xo'jaligi sohasidagi xizmatlarni topishga yordam berish uchun yaratilgan.")


elif st.session_state.page =="5":
        # Postlar va javoblar uchun session state (agar hali aniqlanmagan bo‘lsa)
        

    # Ma'lumotlarni saqlash uchun funksiyalar
    def init_session_state():
        if 'comments' not in st.session_state:
            # Agar comments.csv mavjud bo'lsa, uni o'qiymiz, aks holda bo'sh dataframe yaratamiz
            try:
                st.session_state.comments = pd.read_csv('comments.csv')
            except FileNotFoundError:
                st.session_state.comments = pd.DataFrame(columns=['id', 'parent_id', 'username', 'text', 'timestamp', 'likes'])

    def save_comments():
        # Izohlarni CSV faylga saqlash
        st.session_state.comments.to_csv('comments.csv', index=False)

    def add_comment(username, text, parent_id=None):
        # Yangi izoh qo'shish
        comment_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        new_comment = pd.DataFrame({
            'id': [comment_id],
            'parent_id': [parent_id],
            'username': [username],
            'text': [text],
            'timestamp': [timestamp],
            'likes': [0]
        })
        
        st.session_state.comments = pd.concat([st.session_state.comments, new_comment], ignore_index=True)
        save_comments()
        return comment_id

    def like_comment(comment_id):
        # Izohga like qo'shish
        index = st.session_state.comments[st.session_state.comments['id'] == comment_id].index[0]
        st.session_state.comments.at[index, 'likes'] += 1
        save_comments()

    def get_replies(comment_id):
        # Izohga javoblarni olish
        return st.session_state.comments[st.session_state.comments['parent_id'] == comment_id]

    # Asosiy izohlarni olish (parent_id=None bo'lgan izohlar)
    def get_main_comments():
        return st.session_state.comments[st.session_state.comments['parent_id'].isna()]

        # Izohlarni ko'rsatish uchun funksiya
    def display_comment(comment, level=0):
        comment_id = comment['id']
        username = comment['username']
        text = comment['text']
        timestamp = comment['timestamp']
        likes = comment['likes']
        
        # Padding through indentation in markdown
        indent = "&nbsp;" * (level * 4)
        
        st.markdown(f"{indent}**{username}** - {timestamp}", unsafe_allow_html=True)
        st.markdown(f"{indent}{text}", unsafe_allow_html=True)
        
        # Shu izoh uchun unikal id yaratish
        button_container = f"button_container_{comment_id}_{level}"
        st.markdown(f"<div id='{button_container}'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 8])
        with col1:
            if st.button(f"👍 {likes}", key=f"like_{comment_id}_{level}"):
                like_comment(comment_id)
                st.rerun()
        
        with col2:
            if st.button("Javob berish", key=f"reply_btn_{comment_id}_{level}"):
                st.session_state[f"reply_to_{comment_id}_{level}"] = True
        
        # Javob yozish formasi
        if st.session_state.get(f"reply_to_{comment_id}_{level}", False):
            with st.form(key=f"reply_form_{comment_id}_{level}"):
                reply_username = st.text_input("Foydalanuvchi nomi", key=f"reply_username_{comment_id}_{level}")
                reply_text = st.text_area("Javobingizni yozing", key=f"reply_text_{comment_id}_{level}")
                submit_reply = st.form_submit_button("Javobni yuborish")
                
                if submit_reply and reply_username and reply_text:
                    add_comment(reply_username, reply_text, comment_id)
                    st.session_state[f"reply_to_{comment_id}_{level}"] = False
                    st.rerun()
        
        # Javoblarni ko'rsatish
        replies = get_replies(comment_id)
        if not replies.empty:
            for _, reply in replies.iterrows():
                display_comment(reply, level + 1)

    # Asosiy ilovani yaratish
    def main():
        st.title("Izohlar Tizimi")
        
        # Tab lar orqali sahifalarni boshqarish
        tab1, tab2 = st.tabs(["Izohlar", "Yangi sahifa"])
        
        with tab1:
            st.header("Izohlar sahifasi")
            
            # Yangi izoh qo'shish formasi
            with st.form(key="new_comment_form"):
                st.subheader("Yangi izoh yozing")
                username = st.text_input("Foydalanuvchi nomi")
                comment_text = st.text_area("Izohingizni yozing")
                submit_comment = st.form_submit_button("Izoh qoldirish")
                
                if submit_comment and username and comment_text:
                    add_comment(username, comment_text)
                    st.rerun()
            
            # Mavjud izohlarni ko'rsatish
            st.subheader("Barcha izohlar")
            main_comments = get_main_comments()
            
            if main_comments.empty:
                st.info("Hozircha izohlar yo'q. Birinchi izohni qoldiring!")
            else:
                for _, comment in main_comments.iterrows():
                    display_comment(comment)
                    st.markdown("---")
        
        with tab2:
            st.header("Yangi sahifa")
            st.write("Bu yangi sahifa. Siz bu yerda qo'shimcha funksionallikni qo'shishingiz mumkin.")
            
            # Yangi sahifa uchun kontentni shu yerga qo'shing
            st.subheader("Yangi sahifa kontenti")
            st.write("Bu sahifaning asosiy kontenti.")

    if __name__ == "__main__":
        init_session_state()
        main()

elif st.session_state.page == "6":



    # Session state inizializatsiyasi
    if "active_page" not in st.session_state:
        st.session_state.active_page = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # API kalit
    GEMINI_API_KEY = "AIzaSyCSmZYHuv8uGu2ObL1VqTde6AiGhHLLAOo"

    # Asosiy sarlavha
    st.title("🌱 Eco Assistant")

    # Buttonlar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("AI dan o'simlik kasallik haqida so'rash", type="primary"):
            st.session_state.active_page = "health"
            st.rerun()  # Sahifani qayta yuklash
    with col2:
        if st.button("AI dan o'simlik parvarishi haqida so'rash", type="primary"):
            st.session_state.active_page = "plant"
            st.rerun()  # Sahifani qayta yuklash

    # HEALTH CHATBOT
    if st.session_state.active_page == "health":
        # Stillar
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #2E7D32;
                text-align: center;
                margin-bottom: 20px;
            }
            .sub-header {
                font-size: 1.5rem;
                color: #388E3C;
                margin-bottom: 20px;
            }
            .user-message {
                background-color: #E8F5E9;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .bot-message {
                background-color: #F1F8E9;
                padding: 10px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 5px solid #2E7D32;
            }
        </style>
        """, unsafe_allow_html=True)

        # Sahifa sarlavhasi
        st.markdown("<h1 class='main-header'>🌿 Eco Chat Bot</h1>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>O'simliklardagi kasalliklar haqida malumot beraman</h2>", unsafe_allow_html=True)

        # Bosh sahifaga qaytish
        if st.button("⬅️ Bosh sahifaga qaytish"):
            st.session_state.active_page = None
            st.rerun()

        # Bot haqida ma'lumot
        st.sidebar.markdown("## Bot haqida")
        st.sidebar.info("""
        Bu Eco Chat Bot ekologiya va sog'liq masalalari bo'yicha ma'lumot va maslahatlar beradi. 
            
        Quyidagi mavzular bo'yicha savollar berishingiz mumkin:
        - Ekologik muammolar
        - Atrof-muhit muhofazasi
        - Kasalliklar va ularning davolash choralari
        - Sog'lom turmush tarzi
        """)

        # Funksiyalar
        def initialize_gemini():
            """Gemini API ni ishga tushirish"""
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-pro')
                return model
            except Exception as e:
                st.error(f"API ga ulanishda xatolik: {str(e)}")
                return None

        def get_gemini_response(model, prompt, chat_history):
            """Gemini dan javob olish"""
            try:
                # Maxsus ko'rsatmalar qo'shish
                prompt_with_instructions = f"""
                Sen Eco Chat Bot - ekologiya va sog'liq maslahatchi assistentisan. 
                O'zbek tilida javob ber.
                
                Quyidagi mavzular bo'yicha batafsil javob ber:
                1. Ekologik muammolar va ularning yechimlari
                2. Atrof-muhit muhofazasi usullari
                3. Kasalliklar haqida ma'lumotlar (agar so'ralsa)
                4. Kasalliklarni davolash choralari (agar so'ralsa)
                5. Sog'lom turmush tarzi masalahatlari
                
                Javobni aniq, ilmiy asoslangan va tushunarli tarzda ber. 
                Kasalliklar haqida gapirganingda, bu tibbiy maslahat emasligini, 
                jiddiy holatlarda albatta shifokorga murojaat qilish kerakligini ta'kidla.
                
                Foydalanuvchi savoli: {prompt}
                """
                
                # Chat ni boshlash
                response = model.generate_content(prompt_with_instructions)
                return response.text
            except Exception as e:
                return f"Xatolik yuz berdi: {str(e)}"

        # Modelni yuklash
        model = initialize_gemini()
        if model is None:
            st.error("API kaliti xato yoki model yuklanmadi. Kodda API kalitini tekshiring.")

        # Chat formasi
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Savolingizni kiriting:", key="user_input")
            submit_button = st.form_submit_button("Yuborish")

        # Foydalanuvchi savolini yuborish
        if submit_button and user_input:
            if model:
                # Savolni chat tarixiga qo'shish
                current_time = datetime.datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": current_time
                })
                
                # Javobni olish
                with st.spinner("Javob tayyorlanmoqda..."):
                    bot_response = get_gemini_response(model, user_input, st.session_state.chat_history)
                
                # Javobni chat tarixiga qo'shish
                current_time = datetime.datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": bot_response,
                    "timestamp": current_time
                })
            else:
                st.error("Model yuklanmagan")

        # Chat tarixini ko'rsatish
        if st.session_state.chat_history:
            st.markdown("### Suhbat tarixi")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class='user-message'>
                        <b>Siz ({message["timestamp"]}):</b><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='bot-message'>
                        <b>Eco Bot ({message["timestamp"]}):</b><br>
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)

        # Qo'shimcha ma'lumotlar
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <small>© 2025 Eco Chat Bot. Maxsus maslahatlar uchun mutaxasisga murojaat qiling.</small>
        </div>
        """, unsafe_allow_html=True)

    # O'SIMLIK MASLAHATCHISI
    elif st.session_state.active_page == "plant":
        st.title("🌱 O'simlik parvarishi maslahatchisi")
        st.markdown("O'simlik nomini kiriting va uning parvarishi bo'yicha maslahatlar oling")

        # Bosh sahifaga qaytish
        if st.button("⬅️ Bosh sahifaga qaytish"):
            st.session_state.active_page = None
            st.rerun()

        # Gemini API kalitini o'rnatish
        def setup_genai():
            # API kalitini to'g'ridan-to'g'ri kodda saqlash
            genai.configure(api_key=GEMINI_API_KEY)
            return genai.GenerativeModel('gemini-1.5-pro')

        # Gemini modelini o'rnatish
        model = setup_genai()

        # Form yaratish
        with st.form(key="plant_form"):
            # O'simlik turi kiritish
            plant_name = st.text_input("O'simlik nomini kiriting (masalan: atirgul, kaktus, fikus):")
            
            # Yuborish tugmasi
            submit_button = st.form_submit_button(label="Yuborish")

        # Form yuborilganda
        if submit_button and plant_name:
            with st.spinner(f"{plant_name} uchun parvarish ma'lumotlarini olinmoqda..."):
                # O'simlik ma'lumotlarini olish uchun so'rov
                prompt = f""" 
                Menga quyidagi o'simlik uchun batafsil parvarish ma'lumotlarini taqdim eting: {plant_name}.
                Iltimos, ma'lumotlarni quyidagi aniq tuzilishda taqdim eting:

                1. Optimal saqlash harorati: O'simlik uchun ideal harorat sharoitlari.
                2. Sug'orish jadvali: Qanday chastotada va qancha suv berish tavsiya etiladi.
                3. O'g'itlash: Qaysi turdagi o'g'itlar, qancha miqdorda va qachon berilishi kerakligi haqida aniq tavsiyalar.
                4. Yorug'lik: O'simlik uchun ideal yorug'lik sharoitlari va yorug'lik miqdori.

                Har bir bo'lim uchun aniq va amaliy ma'lumot bering.
                Javobingizni o'zbek tilida bering.
                """
                
                try:
                    response = model.generate_content(prompt)
                    plant_info = response.text
                    
                    # Ma'lumotlarni chiroyli ko'rsatish uchun
                    st.subheader(f"{plant_name.capitalize()} parvarishi bo'yicha maslahatlar")
                    
                    # Gemini javobini ko'rsatish
                    st.markdown(plant_info)
                    
                    # Qo'shimcha foydalanuvchi tajribasi uchun
                    st.success("Ma'lumotlar muvaffaqiyatli olindi!")
                    
                    # Rasmni qidirish havolasi
                    plant_name_encoded = plant_name.replace(" ", "+")
                    search_link = f"https://www.google.com/search?q={plant_name_encoded}+plant&tbm=isch"
                    st.markdown(f"[{plant_name.capitalize()} rasmlarini ko'rish uchun bosing]({search_link})")
                    
                except Exception as e:
                    st.error(f"Xatolik yuz berdi: {str(e)}")
                    st.info("Iltimos, API kalitini tekshiring yoki boshqa o'simlik nomini kiriting.")

        # Footer
        st.markdown("---")
        st.markdown("🌿 O'simlik parvarishi maslahatchisi © 2025")