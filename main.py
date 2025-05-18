import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import os
import joblib # For loading scikit-learn models
import datetime # For date operations

# Specific imports for each app (add all unique imports from all 6 projects here)
try:
    from inference_sdk import InferenceHTTPClient # For Injera App (Project 1)
except ImportError:
    # This will be handled within the injera app function if needed
    INFERENCE_SDK_AVAILABLE = False
else:
    INFERENCE_SDK_AVAILABLE = True

try:
    import google.generativeai as genai # For Chatbot (Project 5)
except ImportError:
    # This will be handled within the chatbot app function if needed
    GENAI_AVAILABLE = False
else:
    GENAI_AVAILABLE = True

# Scikit-learn imports (used by multiple apps)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


# --- Global Page Configuration ---
st.set_page_config(
    page_title="ሁሉን አቀፍ የግብርና ስርዓት",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper function for constructing file paths relative to this main script ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_project_file_path(project_subfolder_name, filename):
    # If project_subfolder_name is empty, it means the file is in BASE_DIR (e.g. assets)
    if not project_subfolder_name:
        return os.path.join(BASE_DIR, filename)
    return os.path.join(BASE_DIR, project_subfolder_name, filename)

# ==============================================================================
# መተግበሪያ 1: የእንጀራ ጥራት ምርመራ
# ==============================================================================
def run_injera_quality_app():
    st.header("🔍 የእንጀራ ጥራት ምርመራ")
    st.markdown("የእንጀራ ፎቶ በመስቀል ወይም ካሜራ በመጠቀም በማንሳት ጥራቱን ይወቁ።")

    if not INFERENCE_SDK_AVAILABLE:
        st.error("የ 'inference_sdk' ፓኬጅ አልተጫነም። እባክዎ ይጫኑት፦ `pip install inference-sdk`")
        st.stop()

    ROBOFLOW_API_KEY_INJERA_ENV = os.environ.get("ROBOFLOW_API_KEY_INJERA")
    DEFAULT_API_KEY_INJERA = st.secrets.get("DEFAULT_ROBOFLOW_API_KEY_INJERA", "YOUR_DEFAULT_ROBOFLOW_KEY_HERE") # Example default, user should change
    DEFAULT_MODEL_ID_INJERA = "injera_quality/5" # User should verify/update this
    DEFAULT_API_URL_INJERA = "https://detect.roboflow.com" # Common URL, adjust if needed

    st.sidebar.subheader("የእንጀራ ምርመራ ማዋቀሪያ")
    api_key_injera = st.sidebar.text_input(
        "የሮቦፍሎው ኤፒአይ ቁልፍ (ለእንጀራ)",
        value=st.secrets.get("ROBOFLOW_API_KEY_INJERA", ROBOFLOW_API_KEY_INJERA_ENV or DEFAULT_API_KEY_INJERA),
        type="password", key="app1_injera_api_key"
    )
    model_id_injera = st.sidebar.text_input(
        "የሮቦፍሎው ሞዴል መለያ (ለእንጀራ)", value=DEFAULT_MODEL_ID_INJERA, key="app1_injera_model_id"
    )
    api_url_injera = st.sidebar.text_input(
        "የሮቦፍሎው ኤፒአይ ዩአርኤል (ለእንጀራ)", value=DEFAULT_API_URL_INJERA, key="app1_injera_api_url"
    )

    CLIENT_INJERA = None
    if api_key_injera and api_key_injera != "YOUR_DEFAULT_ROBOFLOW_KEY_HERE": # Avoid using placeholder key
        try:
            CLIENT_INJERA = InferenceHTTPClient(api_url=api_url_injera, api_key=api_key_injera)
        except Exception as e:
            st.sidebar.error(f"የሮቦፍሎው ደንበኛ (እንጀራ) ለመጀመር አልተቻለም: {e}")
    elif api_key_injera == "YOUR_DEFAULT_ROBOFLOW_KEY_HERE":
        st.sidebar.warning("እባክዎ ትክክለኛ የሮቦፍሎው ኤፒአይ ቁልፍ ያስገቡ።")


    st.subheader("የእንጀራ ምስል ያቅርቡ")
    image_source_injera = st.radio(
        "የምስል ምንጭ ይምረጡ (እንጀራ)፦", ("ምስል ይስቀሉ", "በካሜራ ፎቶ ያንሱ"),
        horizontal=True, key="app1_injera_img_source", label_visibility="collapsed"
    )

    img_bytes_for_processing_injera = None
    source_image_display_injera = None

    if image_source_injera == "ምስል ይስቀሉ":
        img_file_buffer_injera = st.file_uploader(
            "የእንጀራ ምስልዎን ይስቀሉ (JPG, PNG, JPEG)፦", type=["jpg", "png", "jpeg"], key="app1_injera_uploader"
        )
        if img_file_buffer_injera:
            img_bytes_for_processing_injera = img_file_buffer_injera.getvalue()
            source_image_display_injera = Image.open(img_file_buffer_injera)
    elif image_source_injera == "በካሜራ ፎቶ ያንሱ":
        camera_img_buffer_injera = st.camera_input("ፎቶ ለማንሳት ይጫኑ (እንጀራ)፦", key="app1_injera_camera")
        if camera_img_buffer_injera:
            img_bytes_for_processing_injera = camera_img_buffer_injera.getvalue()
            source_image_display_injera = Image.open(camera_img_buffer_injera)

    def translate_class_name_amharic_injera(class_name_en):
        translations = {"good": "ጥሩ", "bad": "መጥፎ", "fair": "ከፊል ጥሩ"}
        return translations.get(class_name_en.lower(), class_name_en)

    if source_image_display_injera:
        col1_injera, col2_injera = st.columns(2)
        with col1_injera:
            st.image(source_image_display_injera, caption="የእርስዎ የእንጀራ ምስል", use_column_width=True)
        with col2_injera:
            st.subheader("የምርመራ ውጤቶች")
            if CLIENT_INJERA and img_bytes_for_processing_injera:
                if st.button("🔬 የእንጀራን ጥራት ይመርምሩ", use_container_width=True, key="app1_injera_inspect_btn"):
                    with st.spinner("እየተመረመረ ነው... እባክዎ ይጠብቁ።"):
                        try:
                            pil_image_to_infer_injera = Image.open(BytesIO(img_bytes_for_processing_injera))
                            result_injera = CLIENT_INJERA.infer(pil_image_to_infer_injera, model_id=model_id_injera)
                            st.success("ምርመራው ተጠናቋል!")
                            st.write("---")
                            if isinstance(result_injera, dict) and 'predictions' in result_injera: # Object detection
                                predictions_injera = result_injera.get('predictions', [])
                                if predictions_injera:
                                    st.write(f"**የተገኙ ነገሮች/አካባቢዎች ({len(predictions_injera)})፦**")
                                    for pred_injera in predictions_injera:
                                        pred_class_en_injera = pred_injera.get('class', "N/A")
                                        confidence_injera = pred_injera.get('confidence', 0)
                                        pred_class_am_injera = translate_class_name_amharic_injera(pred_class_en_injera)
                                        st.write(f"- **{pred_class_am_injera}** (የመተማመን ደረጃ፦ {confidence_injera*100:.2f}%)")
                                        if pred_class_en_injera.lower() == "good": st.balloons()
                                else: st.write("በውጤቱ ውስጥ ምንም ግምቶች አልተገኙም።")
                            # Handling for classification model structure (if top-level is list of dicts or dict with 'top' key)
                            elif isinstance(result_injera, dict) and 'top' in result_injera and 'confidence' in result_injera: # Classification
                                pred_class_en_injera = result_injera.get('top', "N/A")
                                confidence_injera = result_injera.get('confidence', 0)
                                pred_class_am_injera = translate_class_name_amharic_injera(pred_class_en_injera)
                                st.metric(
                                        label=f"የተገመተው ጥራት፦ **{pred_class_am_injera}**",
                                        value=f"{confidence_injera*100:.2f}% የመተማመን ደረጃ"
                                    )
                                if pred_class_en_injera.lower() == "good": st.balloons()
                                elif pred_class_en_injera.lower() == "bad": st.warning("ይህ እንጀራ ዝቅተኛ ጥራት ያለው ሊሆን ይችላል።")

                            elif isinstance(result_injera, list) and result_injera and 'class' in result_injera[0]: # Also can be object detection
                                st.write(f"**የተገኙ ነገሮች/አካባቢዎች ({len(result_injera)})፦**")
                                for pred_injera in result_injera:
                                    pred_class_en_injera = pred_injera.get('class', "N/A")
                                    confidence_injera = pred_injera.get('confidence', 0)
                                    pred_class_am_injera = translate_class_name_amharic_injera(pred_class_en_injera)
                                    st.write(f"- **{pred_class_am_injera}** (የመተማመን ደረጃ፦ {confidence_injera*100:.2f}%)")
                            else:
                                st.info("ግምቶችን መተንተን አልተቻለም። 'ጥሬ ውጤት' ይመልከቱ።")
                            with st.expander("ጥሬ የሮቦፍሎው ውጤት (እንግሊዝኛ)", expanded=False):
                                st.json(result_injera)
                        except Exception as e_injera:
                            st.error(f"በኢንፈረንስ ወቅት ስህተት ተከስቷል (እንጀራ)፦ {e_injera}")
            elif not CLIENT_INJERA:
                st.warning("የሮቦፍሎው ደንበኛ (እንጀራ) አልተጀመረም። ማዋቀሪያውን ያረጋግጡ።")
    else:
        st.info("እባክዎ ምስል ይስቀሉ ወይም ፎቶ ያንሱ (ለእንጀራ)።")
    st.markdown("--- \n በ [Roboflow](https://roboflow.com) የተጎላበተ")


# ==============================================================================
# መተግበሪያ 2: የወተት ጥራት ትንበያ
# ==============================================================================
def run_milk_spoilage_app():
    st.header("🥛 የወተት ጥራት እና የመበላሸት ትንበያ")
    st.markdown("የወተትዎን መረጃ በማስገባት የጥራት ደረጃውን ይተንብዩ።")

    DATA_FILE_PATH_MILK = get_project_file_path("02_milk_spoilage_prediction", "milknew.csv")
    MODEL_FILE_PATH_MILK = get_project_file_path("02_milk_spoilage_prediction", "milk_quality_rf_model.joblib")
    SCALER_FILE_PATH_MILK = get_project_file_path("02_milk_spoilage_prediction", "milk_quality_scaler.joblib")

    @st.cache_data
    def load_and_preprocess_data_milk(file_path):
        try:
            df = pd.read_csv(file_path)
            df['Grade'] = df['Grade'].map({'high': 2, 'medium': 1, 'low': 0})
            if 'Fat ' in df.columns: df.rename(columns={'Fat ': 'Fat'}, inplace=True)
            # Handle potential missing values robustly
            for col in df.select_dtypes(include=np.number).columns:
                if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include='object').columns: #Though 'Grade' is now numeric
                if df[col].isnull().any(): df[col].fillna(df[col].mode()[0], inplace=True)
            return df
        except FileNotFoundError:
            st.error(f"የወተት ዳታ ፋይል አልተገኘም፦ {file_path}")
            return None
        except Exception as e:
            st.error(f"የወተት ዳታ በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}")
            return None

    df_milk = load_and_preprocess_data_milk(DATA_FILE_PATH_MILK)

    @st.cache_resource
    def train_or_load_model_and_scaler_milk(data_frame):
        if data_frame is None: return None, None
        
        # Ensure target 'Grade' is not in X_milk
        if 'Grade' not in data_frame.columns:
            st.error("የዒላማ ዓምድ 'Grade' በወተት ዳታ ውስጥ የለም።")
            return None, None
            
        X_milk = data_frame.drop('Grade', axis=1, errors='ignore') # errors='ignore' if Grade was already removed
        y_milk = data_frame['Grade']

        # Verify X_milk columns (should be 7 features if Grade is target)
        expected_features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat', 'Turbidity', 'Colour']
        if not all(feat in X_milk.columns for feat in expected_features) or len(X_milk.columns) != len(expected_features):
            st.error(f"የወተት ዳታ ግብዓት ዓምዶች ትክክል አይደሉም። የተጠበቁት: {expected_features}, የተገኙት: {X_milk.columns.tolist()}")
            return None, None

        if os.path.exists(MODEL_FILE_PATH_MILK) and os.path.exists(SCALER_FILE_PATH_MILK):
            try:
                model = joblib.load(MODEL_FILE_PATH_MILK)
                scaler = joblib.load(SCALER_FILE_PATH_MILK)
                return model, scaler
            except Exception as e:
                st.sidebar.warning(f"የተቀመጠ የወተት ሞዴል/ስኬለር መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...")

        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_milk, y_milk, test_size=0.2, random_state=42, stratify=y_milk)
        scaler_m = MinMaxScaler()
        X_train_scaled_m = scaler_m.fit_transform(X_train_m)
        X_test_scaled_m = scaler_m.transform(X_test_m)
        model_m = RandomForestClassifier(max_depth=12, random_state=0) # Example depth
        model_m.fit(X_train_scaled_m, y_train_m)
        accuracy_m = accuracy_score(y_test_m, model_m.predict(X_test_scaled_m))
        st.sidebar.metric(label="የወተት ሞዴል ትክክለኛነት (አዲስ የሰለጠነ)", value=f"{accuracy_m*100:.2f}%", key="app2_milk_accuracy_retrained")
        try:
            joblib.dump(model_m, MODEL_FILE_PATH_MILK)
            joblib.dump(scaler_m, SCALER_FILE_PATH_MILK)
        except Exception as e_save: st.sidebar.error(f"የወተት ሞዴል/ስኬለር ማስቀመጥ አልተቻለም፦ {e_save}")
        return model_m, scaler_m

    if df_milk is None: st.stop()
    model_milk, scaler_milk = train_or_load_model_and_scaler_milk(df_milk)
    if not model_milk or not scaler_milk: st.error("የወተት ጥራት ትንበያ ሞዴል ወይም ስኬለር መጫን አልተቻለም።"); st.stop()

    st.sidebar.subheader("የወተት መረጃ ያስገቡ")
    def user_input_features_milk(data_frame_for_ranges):
        # Use data_frame_for_ranges to get min/max for sliders to avoid errors if df_milk is None initially
        ph_m = st.sidebar.slider('ፒኤች (pH)', float(data_frame_for_ranges['pH'].min()), float(data_frame_for_ranges['pH'].max()), float(data_frame_for_ranges['pH'].mean()), 0.1, key="app2_milk_ph")
        temp_m = st.sidebar.slider('የሙቀት መጠን (°C)', int(data_frame_for_ranges['Temprature'].min()), int(data_frame_for_ranges['Temprature'].max()), int(data_frame_for_ranges['Temprature'].mean()), key="app2_milk_temp")
        taste_m = st.sidebar.selectbox('ጣዕም (0=ክፉ, 1=ጥሩ)', (0, 1), index=int(data_frame_for_ranges['Taste'].mode()[0]), key="app2_milk_taste")
        odor_m = st.sidebar.selectbox('ሽታ (0=የለም, 1=አለ)', (0, 1), index=int(data_frame_for_ranges['Odor'].mode()[0]), key="app2_milk_odor")
        fat_m = st.sidebar.selectbox('የስብ መጠን (0=ዝቅተኛ, 1=ከፍተኛ)', (0, 1), index=int(data_frame_for_ranges['Fat'].mode()[0]), key="app2_milk_fat")
        turb_m = st.sidebar.selectbox('ደብዛውነት (0=የለም, 1=አለ)', (0, 1), index=int(data_frame_for_ranges['Turbidity'].mode()[0]), key="app2_milk_turb")
        colour_m = st.sidebar.slider('ቀለም (እሴት)', int(data_frame_for_ranges['Colour'].min()), int(data_frame_for_ranges['Colour'].max()), int(data_frame_for_ranges['Colour'].mean()), key="app2_milk_colour")
        data = {'pH': ph_m, 'Temprature': temp_m, 'Taste': taste_m, 'Odor': odor_m, 'Fat': fat_m, 'Turbidity': turb_m, 'Colour': colour_m}
        return pd.DataFrame(data, index=[0])

    input_df_milk = user_input_features_milk(df_milk)
    st.subheader('እርስዎ ያስገቡት የወተት መረጃ፦')
    st.write(input_df_milk)

    if st.button("🥛 የወተትን ጥራት ተንብይ", key="app2_milk_predict_btn", use_container_width=True):
        input_df_scaled_milk = scaler_milk.transform(input_df_milk)
        prediction_milk = model_milk.predict(input_df_scaled_milk)
        prediction_proba_milk = model_milk.predict_proba(input_df_scaled_milk)
        st.subheader('የትንበያ ውጤት (ወተት)፦')
        grade_map_amharic_milk = {2: "ከፍተኛ ጥራት (High)", 1: "መካከለኛ ጥራት (Medium)", 0: "ዝቅተኛ ጥራት (Low)"}
        predicted_grade_amharic_milk = grade_map_amharic_milk.get(prediction_milk[0], "ያልታወቀ")
        st.success(f"የተገመተው የወተት ጥራት ደረጃ፦ **{predicted_grade_amharic_milk}**")
        if prediction_milk[0] == 0: st.warning("ይህ ወተት ዝቅተኛ ጥራት ያለው ወይም የመበላሸት ስጋት ሊኖረው ይችላል።")
        elif prediction_milk[0] == 2: st.balloons()
        st.subheader('የመተማመን ደረጃ ለእያንዳንዱ የጥራት መደብ (ወተት)፦')
        proba_df_milk = pd.DataFrame({
            "የጥራት መደብ": [grade_map_amharic_milk[0], grade_map_amharic_milk[1], grade_map_amharic_milk[2]], # Order to match probabilities
            "የመተማመን ዕድል": [f"{p*100:.2f}%" for p in prediction_proba_milk[0]]
        })
        st.table(proba_df_milk)
# ==============================================================================
# መተግበሪያ 3: የማዳበሪያ ምክረ ሀሳብ
# ==============================================================================
def run_fertilizer_recommendation_app():
    st.header("🌱 የማዳበሪያ ምክረ ሀሳብ ስርዓት")
    st.markdown("የአካባቢዎን እና የሰብልዎን መረጃ በማስገባት ተስማሚውን ማዳበሪያ ይወቁ።")

    # This should be the column name the *saved model expects*.
    # Based on the error, it's 'Temparature' (with 'a').
    MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME = 'Temparature'

    DATA_FILE_PATH_FERT = get_project_file_path("03_fertilizer_recommendation", "fertilizer_recommendation_data.csv")
    MODEL_PIPELINE_FILE_PATH_FERT = get_project_file_path("03_fertilizer_recommendation", "fertilizer_model_pipeline.joblib")
    TARGET_ENCODER_FILE_PATH_FERT = get_project_file_path("03_fertilizer_recommendation", "fertilizer_target_encoder.joblib")

    @st.cache_data
    def load_data_fert(file_path):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            # Standardize temperature column name to what the model expects
            current_temp_col_in_csv = None
            if MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME in df.columns:
                current_temp_col_in_csv = MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME
            elif 'Temprature' in df.columns: # Check for 'e' spelling
                current_temp_col_in_csv = 'Temprature'
            elif 'Temperature' in df.columns: # Check for standard 'e' spelling (Temperature)
                current_temp_col_in_csv = 'Temperature'
            
            if current_temp_col_in_csv and current_temp_col_in_csv != MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME:
                #st.warning(f"የዓምድ ስም '{current_temp_col_in_csv}' ወደ '{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}' ተቀይሯል (ለወጥነት)።")
                df.rename(columns={current_temp_col_in_csv: MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}, inplace=True)
            elif not current_temp_col_in_csv:
                st.error(f"ስህተት፦ በዳታ ፋይሉ ውስጥ ወሳኝ የሆነው የሙቀት መጠን ዓምድ ('{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}', 'Temprature', or 'Temperature') አልተገኘም።")
                return None

            if df.isnull().sum().any():
                for col in df.select_dtypes(include=np.number).columns: df[col].fillna(df[col].median(), inplace=True)
                for col in df.select_dtypes(include='object').columns: df[col].fillna(df[col].mode()[0], inplace=True)
            return df
        except FileNotFoundError: st.error(f"የማዳበሪያ ዳታ ፋይል አልተገኘም፦ {file_path}"); return None
        except Exception as e: st.error(f"የማዳበሪያ ዳታ በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}"); return None

    df_fertilizer = load_data_fert(DATA_FILE_PATH_FERT)

    @st.cache_resource
    def train_or_load_model_pipeline_fert(data_frame):
        if data_frame is None: return None, None, None, None, None
        target_col_fert = 'Fertilizer Name'
        
        if MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME not in data_frame.columns:
            st.error(f"ስህተት፦ '{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}' ዓምድ በዳታ ፍሬሙ ውስጥ የለም (ለሞዴል ስልጠና)።")
            return None, None, None, None, None

        X_fert = data_frame.drop(target_col_fert, axis=1)
        y_raw_fert = data_frame[target_col_fert]
        target_encoder_fert = LabelEncoder()
        y_fert = target_encoder_fert.fit_transform(y_raw_fert)
        
        categorical_features_fert = ['Soil Type', 'Crop Type']
        numerical_features_fert = X_fert.select_dtypes(include=np.number).columns.tolist()
        
        if MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME not in numerical_features_fert:
            # This might happen if the column is object type, load_data_fert should prevent this.
            st.error(f"'{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}' ዓምድ እንደ ቁጥር አልተገኘም (ለስልጠና)። የዳታ ጭነትን ያረጋግጡ።")
            return None, None, None, None, None

        preprocessor_fert = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features_fert),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features_fert)],
            remainder='drop') # Changed to 'drop' for robustness
        
        model_pipeline_fert = Pipeline(steps=[
            ('preprocessor', preprocessor_fert),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))])

        if os.path.exists(MODEL_PIPELINE_FILE_PATH_FERT) and os.path.exists(TARGET_ENCODER_FILE_PATH_FERT):
            try:
                loaded_pipeline = joblib.load(MODEL_PIPELINE_FILE_PATH_FERT)
                loaded_target_encoder = joblib.load(TARGET_ENCODER_FILE_PATH_FERT)
                return loaded_pipeline, loaded_target_encoder, data_frame['Soil Type'].unique(), data_frame['Crop Type'].unique(), data_frame
            except Exception as e: st.sidebar.warning(f"የተቀመጠ የማዳበሪያ ሞዴል/ኢንኮደር መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...")

        try:
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42, stratify=y_fert)
            model_pipeline_fert.fit(X_train_f, y_train_f)
        except Exception as e_fit:
            st.error(f"የማዳበሪያ ሞዴልን በማሰልጠን ላይ ስህተት ተከስቷል፦ {e_fit}")
            st.error(f"ለስልጠና የቀረቡት የ X ዓምዶች፦ {X_fert.columns.tolist()}")
            return None, None, None, None, None

        accuracy_f = model_pipeline_fert.score(X_test_f, y_test_f)
        st.sidebar.metric(label="የማዳበሪያ ሞዴል ትክክለኛነት (አዲስ የሰለጠነ)", value=f"{accuracy_f*100:.2f}%", key="app3_fert_accuracy_retrained")
        try:
            joblib.dump(model_pipeline_fert, MODEL_PIPELINE_FILE_PATH_FERT)
            joblib.dump(target_encoder_fert, TARGET_ENCODER_FILE_PATH_FERT)
        except Exception as e_save: st.sidebar.error(f"የማዳበሪያ ሞዴል/ኢንኮደር ማስቀመጥ አልተቻለም፦ {e_save}")
        return model_pipeline_fert, target_encoder_fert, data_frame['Soil Type'].unique(), data_frame['Crop Type'].unique(), data_frame

    if df_fertilizer is None:
        st.error("የማዳበሪያ ዳታ መጫን ስላልተቻለ ይህ መሳሪያ አይሰራም።")
        st.stop()
        
    pipeline_fert, LEncoder_fert, unique_soil_types_fert, unique_crop_types_fert, df_for_ranges_fert = train_or_load_model_pipeline_fert(df_fertilizer)
    
    if not pipeline_fert or not LEncoder_fert or df_for_ranges_fert is None :
        st.error("የማዳበሪያ ሞዴል ወይም አስፈላጊ መረጃዎች አልተጫኑም።")
        st.stop()

    st.sidebar.subheader("የማዳበሪያ መረጃ ያስገቡ")
    def get_user_inputs_fert(soil_types_list, crop_types_list, data_for_input_ranges):
        user_inputs = {} # This dictionary will store inputs with correct keys
        
        # Temperature input: Use MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME for key and access
        if MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME not in data_for_input_ranges.columns:
            st.sidebar.error(f"ስህተት፦ '{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}' ዓምድ በዳታው (ለወሰኖች) ውስጥ የለም። ነባሪ እሴቶች ጥቅም ላይ ይውላሉ።")
            temp_f_val = st.sidebar.slider(f'የአየር ሙቀት መጠን (°C) [{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}]', 10, 40, 25, key="app3_fert_temp_main_fallback")
        else:
            temp_f_val = st.sidebar.slider(f'የአየር ሙቀት መጠን (°C) [{MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME}]', 
                                            int(data_for_input_ranges[MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME].min()), 
                                            int(data_for_input_ranges[MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME].max()), 
                                            int(data_for_input_ranges[MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME].mean()), 
                                            key="app3_fert_temp_main")
        user_inputs[MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME] = temp_f_val
        
        # Other inputs (assuming their names are consistent and correct in data_for_input_ranges)
        user_inputs['Humidity'] = st.sidebar.slider('የአየር ንብረት እርጥበት (%)', int(data_for_input_ranges.get('Humidity', pd.Series([0,100,50])).min()), int(data_for_input_ranges.get('Humidity', pd.Series([0,100,50])).max()), int(data_for_input_ranges.get('Humidity', pd.Series([0,100,50])).mean()), key="app3_fert_hum_main")
        user_inputs['Moisture'] = st.sidebar.slider('የአፈር እርጥበት (%)', int(data_for_input_ranges.get('Moisture', pd.Series([0,100,50])).min()), int(data_for_input_ranges.get('Moisture', pd.Series([0,100,50])).max()), int(data_for_input_ranges.get('Moisture', pd.Series([0,100,50])).mean()), key="app3_fert_moist_main")
        user_inputs['Soil Type'] = st.sidebar.selectbox('የአፈር አይነት', sorted(list(soil_types_list)) if soil_types_list is not None else [], key="app3_fert_soil_main")
        user_inputs['Crop Type'] = st.sidebar.selectbox('የሰብል አይነት', sorted(list(crop_types_list)) if crop_types_list is not None else [], key="app3_fert_crop_main")
        user_inputs['Nitrogen'] = st.sidebar.slider('የናይትሮጅን መጠን (kg/ha)', int(data_for_input_ranges.get('Nitrogen', pd.Series([0,100,20])).min()), int(data_for_input_ranges.get('Nitrogen', pd.Series([0,100,20])).max()), int(data_for_input_ranges.get('Nitrogen', pd.Series([0,100,20])).mean()), key="app3_fert_n_main")
        user_inputs['Potassium'] = st.sidebar.slider('የፖታሲየም መጠን (kg/ha)', int(data_for_input_ranges.get('Potassium', pd.Series([0,100,10])).min()), int(data_for_input_ranges.get('Potassium', pd.Series([0,100,10])).max()), int(data_for_input_ranges.get('Potassium', pd.Series([0,100,10])).mean()), key="app3_fert_k_main")
        user_inputs['Phosphorous'] = st.sidebar.slider('የፎስፈረስ መጠን (kg/ha)', int(data_for_input_ranges.get('Phosphorous', pd.Series([0,100,10])).min()), int(data_for_input_ranges.get('Phosphorous', pd.Series([0,100,10])).max()), int(data_for_input_ranges.get('Phosphorous', pd.Series([0,100,10])).mean()), key="app3_fert_p_main")
        
        # Create DataFrame using the keys from user_inputs dictionary
        # This ensures column names in the DataFrame match the keys used above.
        return pd.DataFrame({k: [v] for k, v in user_inputs.items()})

    if unique_soil_types_fert is not None and unique_crop_types_fert is not None and df_for_ranges_fert is not None:
        input_df_fert = get_user_inputs_fert(unique_soil_types_fert, unique_crop_types_fert, df_for_ranges_fert)
        st.subheader('እርስዎ ያስገቡት መረጃ (ለማዳበሪያ)፦')
        st.dataframe(input_df_fert) # Displaying the DF sent for prediction
        
        if st.button('💡 የማዳበሪያ ምክረ ሀሳብ አግኝ', use_container_width=True, key="app3_fert_recommend_btn_main"):
            with st.spinner("ምክረ ሀሳብ እየተዘጋጀ ነው..."):
                try:
                    # Ensure the input DataFrame columns match what the pipeline expects.
                    # The pipeline's preprocessor's `ColumnTransformer` is fitted on specific column names.
                    # If MODEL_EXPECTED_TEMPERATURE_COLUMN_NAME was 'Temparature' (a), and input_df_fert has it, it should work.
                    
                    prediction_encoded_fert = pipeline_fert.predict(input_df_fert)
                    predicted_fertilizer_name_fert = LEncoder_fert.inverse_transform(prediction_encoded_fert)
                    st.subheader('የማዳበሪያ ምክረ ሀሳብ፦')
                    st.success(f"ለእርስዎ ሁኔታ ተስማሚ የሆነው ማዳበሪያ፦ **{predicted_fertilizer_name_fert[0]}** ነው")
                    st.balloons()
                except ValueError as ve: # Catch specific ValueError for missing columns
                    st.error(f"ምክረ ሀሳብ በማዘጋጀት ላይ ሳለ ስህተት ተከስቷል (ማዳበሪያ)፦ {ve}")
                    st.error(f"የቀረበው ግብዓት ዓምዶች (ለትንበያ)፦ {input_df_fert.columns.tolist()}")
                    # Try to get expected feature names from the pipeline if possible
                    if hasattr(pipeline_fert, 'feature_names_in_'):
                        st.error(f"ሞዴሉ የሚጠብቃቸው የግብዓት ዓምዶች (ከፓይፕላይን)፦ {pipeline_fert.feature_names_in_}")
                    elif hasattr(pipeline_fert.named_steps.get('preprocessor'), 'get_feature_names_out'):
                         try:
                             st.error(f"ሞዴሉ የሚጠብቃቸው የግብዓት ዓምዶች (ከፕሪፕሮሰሰር)፦ {pipeline_fert.named_steps['preprocessor'].get_feature_names_out()}")
                         except:
                             st.error("የፕሪፕሮሰሰር የግብዓት ዓምዶችን ማግኘት አልተቻለም።")
                    else:
                        st.error("የሞዴሉን (preprocessor) የግብዓት ዓምዶች ማወቅ አልተቻለም።")

                except Exception as e_predict: 
                    st.error(f"ምክረ ሀሳብ በማዘጋጀት ላይ ሳለ ያልታወቀ ስህተት ተከስቷል (ማዳበሪያ)፦ {e_predict}")
                    st.error(f"የቀረበው ግብዓት ዓምዶች፦ {input_df_fert.columns.tolist()}")
    else:
        st.error("የአፈር ወይም የሰብል አይነቶችን (ለማዳበሪያ) ከዳታው ማግኘት አልተቻለም።")

# ==============================================================================
# መተግበሪያ 4: የሰብል ምርት ትንበያ
# ==============================================================================
def run_crop_yield_app():
    st.header("🌾 የሰብል ምርት ትንበያ ስርዓት")
    st.markdown("የተለያዩ መረጃዎችን በማስገባት የሚጠበቀውን የሰብል ምርታማነት (Yield) ይተንብዩ።")

    DATA_FILE_PATH_YIELD = get_project_file_path("04_crop_yield_prediction", "crop_yield_data.csv")
    MODEL_PIPELINE_FILE_PATH_YIELD = get_project_file_path("04_crop_yield_prediction", "crop_yield_model_pipeline.joblib")

    @st.cache_data
    def load_data_yield(file_path):
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            for col in df.select_dtypes(include=np.number).columns:
                if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)
            for col in df.select_dtypes(include='object').columns:
                if df[col].isnull().any(): df[col].fillna(df[col].mode()[0], inplace=True)
            if 'Production' in df.columns and df['Production'].dtype == 'object':
                df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
                numeric_production = df['Production'].dropna()
                df['Production'].fillna(numeric_production.median() if not numeric_production.empty else 0, inplace=True)
            if 'Yield' not in df.columns:
                if 'Production' in df.columns and 'Area' in df.columns:
                    df['Yield'] = df.apply(lambda row: row['Production'] / row['Area'] if row['Area'] != 0 else 0, axis=1)
                    df['Yield'].replace([np.inf, -np.inf], np.nan, inplace=True)
                    df['Yield'].fillna(df['Yield'].median(), inplace=True)
                else:
                    st.error("የሰብል ምርት ዳታ 'Yield' ዓምድ ወይም እሱን ለመፍጠር የሚያስችሉ 'Production' እና 'Area' ዓምዶች የሉትም።")
                    return None
            return df
        except FileNotFoundError: st.error(f"የሰብል ምርት ዳታ ፋይል አልተገኘም፦ {file_path}"); return None
        except Exception as e: st.error(f"የሰብል ምርት ዳታ በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}"); return None

    df_yield_data = load_data_yield(DATA_FILE_PATH_YIELD)

    @st.cache_resource
    def train_or_load_yield_model_pipeline(data_frame):
        if data_frame is None: return None, None, None, None, None, None
        target_col_y = 'Yield'
        # Ensure Yield is present before dropping
        if target_col_y not in data_frame.columns:
            st.error(f"ዒላማ ዓምድ '{target_col_y}' በሰብል ምርት ዳታ ውስጥ የለም።")
            return None, None, None, None, None, None

        features_to_drop_y = [target_col_y, 'Production'] # Production also often dropped if Yield is derived
        X_y = data_frame.drop(columns=[col for col in features_to_drop_y if col in data_frame.columns], axis=1)
        y_y = data_frame[target_col_y]
        
        categorical_features_y = X_y.select_dtypes(include='object').columns.tolist()
        numerical_features_y = X_y.select_dtypes(include=np.number).columns.tolist()
        
        if not numerical_features_y and not categorical_features_y : # Check if X_y is empty
            st.error("ምንም የግብዓት ዓምዶች (features) ለሰብል ምርት ሞዴል አልተገኙም።")
            return None, None, None, None, None, None

        preprocessor_y = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features_y),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features_y)],
            remainder='drop') # Drop any other columns not specified
        
        model_pipeline_y = Pipeline(steps=[
            ('preprocessor', preprocessor_y),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

        if os.path.exists(MODEL_PIPELINE_FILE_PATH_YIELD):
            try:
                loaded_pipeline = joblib.load(MODEL_PIPELINE_FILE_PATH_YIELD)
                # To get feature names, it's best if they were saved during training
                # For now, derive from data_frame like before
                temp_X_y = data_frame.drop(columns=[col for col in features_to_drop_y if col in data_frame.columns], axis=1)
                feature_names_out = temp_X_y.columns.tolist()
                return (loaded_pipeline, data_frame['Crop'].unique(), data_frame['Season'].unique(), data_frame['State'].unique(), data_frame, feature_names_out)
            except Exception as e: st.sidebar.warning(f"የተቀመጠ የሰብል ምርት ሞዴል መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...")

        X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_y, y_y, test_size=0.2, random_state=42)
        model_pipeline_y.fit(X_train_y, y_train_y)
        r2_y = model_pipeline_y.score(X_test_y, y_test_y)
        rmse_y = np.sqrt(mean_squared_error(y_test_y, model_pipeline_y.predict(X_test_y)))
        st.sidebar.metric(label="የሰብል ምርት ሞዴል R² (አዲስ የሰለጠነ)", value=f"{r2_y:.3f}", key="app4_yield_r2_retrained")
        st.sidebar.metric(label="የሰብል ምርት ሞዴል RMSE (አዲስ የሰለጠነ)", value=f"{rmse_y:.3f}", key="app4_yield_rmse_retrained")
        try: joblib.dump(model_pipeline_y, MODEL_PIPELINE_FILE_PATH_YIELD)
        except Exception as e_save: st.sidebar.error(f"የሰብል ምርት ሞዴል ማስቀመጥ አልተቻለም፦ {e_save}")
        return (model_pipeline_y, data_frame['Crop'].unique(), data_frame['Season'].unique(), data_frame['State'].unique(), data_frame, X_y.columns.tolist())

    if df_yield_data is None: st.error("የሰብል ምርት ዳታ መጫን ስላልተቻለ ይህ መሳሪያ አይሰራም።"); st.stop()
    pipeline_y, unique_crops_y, unique_seasons_y, unique_states_y, df_for_ranges_y, feature_names_y = train_or_load_yield_model_pipeline(df_yield_data)
    
    if not pipeline_y or not feature_names_y or df_for_ranges_y is None:
        st.error("የሰብል ምርት ትንበያ ሞዴል ወይም አስፈላጊ መረጃዎች አልተጫኑም።")
        st.stop()


    st.sidebar.subheader("የሰብል ምርት ትንበያ መረጃ ያስገቡ")
    def get_user_yield_inputs(crops_list, seasons_list, states_list, data_for_input_ranges, model_expected_features):
        inputs = {}
        current_real_year_y = datetime.date.today().year
        
        # Ensure inputs are only created for features the model expects
        if 'Crop_Year' in model_expected_features:
            min_crop_year_data_y = int(data_for_input_ranges['Crop_Year'].min()) if 'Crop_Year' in data_for_input_ranges.columns and not data_for_input_ranges['Crop_Year'].empty else current_real_year_y - 10
            max_crop_year_data_y = int(data_for_input_ranges['Crop_Year'].max()) if 'Crop_Year' in data_for_input_ranges.columns and not data_for_input_ranges['Crop_Year'].empty else current_real_year_y
            default_crop_year_val_y = min(current_real_year_y, max_crop_year_data_y)
            max_input_year_y = max(max_crop_year_data_y, current_real_year_y + 2)
            inputs['Crop_Year'] = st.sidebar.number_input('የሰብል አመት (ለምርት)', min_value=min_crop_year_data_y, max_value=max_input_year_y, value=default_crop_year_val_y, step=1, key="app4_yield_year")
        
        if 'Crop' in model_expected_features: inputs['Crop'] = st.sidebar.selectbox('የሰብል አይነት (ለምርት)', sorted(list(crops_list)) if crops_list is not None else [], key="app4_yield_crop")
        if 'Season' in model_expected_features: inputs['Season'] = st.sidebar.selectbox('ወቅት (ለምርት)', sorted(list(seasons_list)) if seasons_list is not None else [], key="app4_yield_season")
        if 'State' in model_expected_features: inputs['State'] = st.sidebar.selectbox('ግዛት/ክልል (ለምርት)', sorted(list(states_list)) if states_list is not None else [], key="app4_yield_state")

        num_inputs_def_y = {
            'Area': ('የለማ መሬት ስፋት (ሄክታር)', 1000.0, 100.0, "%.2f"),
            'Annual_Rainfall': ('አመታዊ የዝናብ መጠን (mm)', 1200.0, 50.0, "%.2f"),
            'Fertilizer': ('የማዳበሪያ መጠን (kg)', 50000.0, 1000.0, "%.2f"),
            'Pesticide': ('የፀረ-ተባይ መጠን (kg/L)', 500.0, 100.0, "%.2f")}
        
        for feature, (label, def_val, step, fmt) in num_inputs_def_y.items():
            if feature in model_expected_features:
                min_val = float(data_for_input_ranges[feature].min()) if feature in data_for_input_ranges.columns and not data_for_input_ranges[feature].empty else 0.0
                max_val = float(data_for_input_ranges[feature].max()) if feature in data_for_input_ranges.columns and not data_for_input_ranges[feature].empty else def_val * 5
                mean_val = float(data_for_input_ranges[feature].mean()) if feature in data_for_input_ranges.columns and not data_for_input_ranges[feature].empty else def_val
                inputs[feature] = st.sidebar.number_input(label, min_value=min_val, max_value=max_val, value=mean_val, step=step, format=fmt, key=f"app4_yield_{feature}")
        
        # Create DataFrame with only the features the model expects, in the correct order.
        user_data_dict = {key: [inputs.get(key)] for key in model_expected_features}

        # Handle cases where a model feature might not have a UI element (e.g., if logic error)
        for feature in model_expected_features:
            if feature not in user_data_dict or user_data_dict[feature][0] is None:
                # Fill with NaN or a sensible default if a feature was missed by UI for some reason
                # This should ideally not happen if model_expected_features drives UI creation
                user_data_dict[feature] = [np.nan]
                st.warning(f"የሰብል ምርት ግብዓት '{feature}' አልተገኘም ወይም ባዶ ነው። በ NaN ይተካል።")


        return pd.DataFrame(user_data_dict)[model_expected_features] # Enforce order

    input_df_y = get_user_yield_inputs(unique_crops_y, unique_seasons_y, unique_states_y, df_for_ranges_y, feature_names_y)
    st.subheader('እርስዎ ያስገቡት መረጃ (ለምርት ትንበያ)፦')
    st.dataframe(input_df_y)
    if st.button('🌾 ምርታማነትን ተንብይ', use_container_width=True, key="app4_yield_predict_btn"):
        with st.spinner("የምርት ትንበያ እየተሰራ ነው..."):
            try:
                predicted_yield_y = pipeline_y.predict(input_df_y)
                final_yield_y = max(0, predicted_yield_y[0]) # Ensure non-negative yield
                st.subheader('የተተነበየው የሰብል ምርታማነት (Yield)፦')
                st.success(f"በግምት የሚጠበቀው ምርታማነት፦ **{final_yield_y:.3f}** (መለኪያ አሃዱ በዋናው ዳታ መሰረት)")
            except Exception as e: st.error(f"የምርት ትንበያ ላይ ስህተት ተከስቷል፦ {e}")


# ==============================================================================
# መተግበሪያ 5: የአማርኛ AI ቻትቦት (Gemini)
# ==============================================================================
def run_chatbot_app():
    st.header("🤖 የግብርና እና ምግብ አማካሪ (AI Chatbot)")
    st.caption("በአማርኛ ስለ ግብርና እና ምግብ ጉዳዮች ይጠይቁ")

    if not GENAI_AVAILABLE:
        st.error("የ 'google-generativeai' ፓኬጅ አልተጫነም። እባክዎ ይጫኑት፦ `pip install google-generativeai`")
        st.stop()

    GEMINI_API_KEY_CHATBOT = st.secrets.get("GEMINI_API_KEY_CHATBOT_MAIN", st.secrets.get("GEMINI_API_KEY")) # Try specific then general
    if not GEMINI_API_KEY_CHATBOT: GEMINI_API_KEY_CHATBOT = os.environ.get("GEMINI_API_KEY_CHATBOT_MAIN", os.environ.get("GEMINI_API_KEY"))
    
    if not GEMINI_API_KEY_CHATBOT:
        st.sidebar.subheader("የ Gemini ኤፒአይ ቁልፍ (ለቻትቦት)")
        GEMINI_API_KEY_CHATBOT_INPUT = st.sidebar.text_input("የ Gemini ኤፒአይ ቁልፍዎን እዚህ ያስገቡ፦", type="password", key="app5_gemini_api_key")
        if GEMINI_API_KEY_CHATBOT_INPUT: GEMINI_API_KEY_CHATBOT = GEMINI_API_KEY_CHATBOT_INPUT
        else: st.warning("⚠️ የ Gemini ኤፒአይ ቁልፍ (ለቻትቦት) አልተገኘም።"); st.stop()
    
    try: genai.configure(api_key=GEMINI_API_KEY_CHATBOT)
    except Exception as e: st.error(f"የ Gemini ኤፒአይ ቁልፍን (ለቻትቦት) በማዋቀር ላይ ስህተት ተከስቷል፦ {e}"); st.stop()

    MODEL_NAME_CHATBOT = "gemini-1.5-flash" # Or other suitable model
    SYSTEM_PROMPT_AMHARIC_CHATBOT = """ሰላም! አንተ የምትናገረው በአማርኛ ብቻ ነው። እኔ ስለ ኢትዮጵያ ግብርና፣ አዝመራ አመራረት፣ የእንስሳት እርባታ ዘዴዎች、 የአፈርና ውሃ አያያዝ、 የሰብል ተባይና በሽታ ቁጥጥር、 ዘመናዊ የግብርና ቴክኖሎጂዎች、 የምግብ አይነቶች、 የምግብ ዝግጅት、 የምግብ ደህንነት、 እና ስነ-ምግብ ጉዳዮች መረጃ ለመስጠትና ለመወያየት የተዘጋጀሁ የሰው ሰራሽ የማሰብ ችሎታ ረዳት ነኝ። እባክዎን ጥያቄዎን በእነዚህ ርዕሶች ዙሪያ ብቻ ያቅርቡ። ከእነዚህ ርዕሶች ውጪ ለሚቀርቡ ጥያቄዎች መልስ ለመስጠትም ሆነ ለመወያየት አልተፈቀደልኝም። በግብርና ወይም በምግብ ነክ ጉዳይ ላይ ምን ልርዳዎት?"""


    try:
        if "app5_chat_session" not in st.session_state:
            model_chatbot = genai.GenerativeModel(MODEL_NAME_CHATBOT, system_instruction=SYSTEM_PROMPT_AMHARIC_CHATBOT)
            st.session_state.app5_chat_session = model_chatbot.start_chat(history=[])
    except Exception as e: st.error(f"የ Gemini ሞዴልን (ለቻትቦት) በማስጀመር ላይ ስህተት ተከስቷል፦ {e}"); st.stop()

    if "app5_chat_messages" not in st.session_state: st.session_state.app5_chat_messages = []

    for message in st.session_state.app5_chat_messages:
        with st.chat_message(message["role"]): st.markdown(message["parts"][0]["text"])

    user_prompt_chatbot = st.chat_input("ጥያቄዎን እዚህ በአማርኛ ይጻፉ (ለቻትቦት)...", key="app5_chat_input")
    if user_prompt_chatbot:
        st.session_state.app5_chat_messages.append({"role": "user", "parts": [{"text": user_prompt_chatbot}]})
        with st.chat_message("user"): st.markdown(user_prompt_chatbot)
        
        with st.chat_message("model"):
            message_placeholder_chatbot = st.empty()
            full_response_text_chatbot = ""
            try:
                response_chatbot = st.session_state.app5_chat_session.send_message(user_prompt_chatbot, stream=True)
                for chunk in response_chatbot:
                    text_part = None
                    if hasattr(chunk, 'text') and chunk.text:
                        text_part = chunk.text
                    elif hasattr(chunk, 'parts') and chunk.parts and hasattr(chunk.parts[0], 'text') and chunk.parts[0].text:
                        text_part = chunk.parts[0].text
                    
                    if text_part:
                        full_response_text_chatbot += text_part
                        message_placeholder_chatbot.markdown(full_response_text_chatbot + "▌")
                message_placeholder_chatbot.markdown(full_response_text_chatbot)
            except Exception as e_chat:
                full_response_text_chatbot = f"የቻትቦት ምላሽ በማግኘት ላይ ሳለ ስህተት ተከስቷል፦ {e_chat}"
                message_placeholder_chatbot.error(full_response_text_chatbot)
        st.session_state.app5_chat_messages.append({"role": "model", "parts": [{"text": full_response_text_chatbot}]})

    if st.sidebar.button("የቻትቦት ውይይቱን አጽዳ", key="app5_clear_chat_btn"):
        st.session_state.app5_chat_messages = []
        if "app5_chat_session" in st.session_state: del st.session_state.app5_chat_session 
        st.rerun()


# ==============================================================================
# መተግበሪያ 6: የግብርና እቅድ አስመሳይ
# ==============================================================================
def run_agri_planner_app():
    st.header("🛠️ የግብርና እቅድ አስመሳይ እና ማመቻቻ")
    st.markdown("የተለያዩ የግብርና ግብዓቶችን እና የመዝሪያ ጊዜዎችን በማስገባት የሚጠበቀውን የሰብል ምርታማነት (Yield) ለመገመት ይረዳዎታል።")

    # Planner uses the model from App 4 (Crop Yield Prediction)
    MODEL_PIPELINE_FILE_PATH_PLANNER = get_project_file_path("04_crop_yield_prediction", "crop_yield_model_pipeline.joblib")
    DATA_FILE_PATH_PLANNER_INFO = get_project_file_path("04_crop_yield_prediction", "crop_yield_data.csv") # For input ranges and unique values

    @st.cache_resource
    def load_planner_model_and_data_info(model_path, data_path):
        model_p = None; df_info_p = None; features_p_list = None
        if not os.path.exists(model_path):
            st.error(f"የሰብል ምርት ትንበያ ሞዴል ፋይል (ለእቅድ) አልተገኘም፦ {model_path}"); return None, None, None
        try: model_p = joblib.load(model_path)
        except Exception as e: st.error(f"የትንበያ ሞዴሉን (ለእቅድ) በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}"); return None, None, None

        if os.path.exists(data_path):
            try:
                # Load data similar to App 4 for consistency
                df_temp_p = pd.read_csv(data_path)
                df_temp_p.columns = df_temp_p.columns.str.strip()
                # Basic cleaning for info extraction
                for col in df_temp_p.select_dtypes(include=np.number).columns:
                    if df_temp_p[col].isnull().any(): df_temp_p[col].fillna(df_temp_p[col].median(), inplace=True)
                for col in df_temp_p.select_dtypes(include='object').columns:
                    if df_temp_p[col].isnull().any(): df_temp_p[col].fillna(df_temp_p[col].mode()[0], inplace=True)

                if 'Production' in df_temp_p.columns and df_temp_p['Production'].dtype == 'object':
                    df_temp_p['Production'] = pd.to_numeric(df_temp_p['Production'], errors='coerce')
                    numeric_prod_p = df_temp_p['Production'].dropna()
                    df_temp_p['Production'].fillna(numeric_prod_p.median() if not numeric_prod_p.empty else 0, inplace=True)

                if 'Yield' not in df_temp_p.columns and 'Production' in df_temp_p.columns and 'Area' in df_temp_p.columns:
                     df_temp_p['Yield'] = df_temp_p.apply(lambda row: row['Production'] / row['Area'] if row['Area'] != 0 else 0, axis=1)
                     df_temp_p['Yield'].replace([np.inf, -np.inf], np.nan, inplace=True)
                     df_temp_p['Yield'].fillna(df_temp_p['Yield'].median(), inplace=True)
                
                features_to_drop_p_info = ['Yield', 'Production']
                df_info_p = df_temp_p.copy() # Use a copy for info, original df_temp_p might be modified
                features_p_list = df_temp_p.drop(columns=[col for col in features_to_drop_p_info if col in df_temp_p.columns], axis=1).columns.tolist()

            except Exception as e_data: st.warning(f"የቀድሞውን ዳታ ፋይል (ለእቅድ) ለማጣቀሻ መጫን አልተቻለም፦ {e_data}")
        else: st.warning(f"የቀድሞው ዳታ ፋይል (ለእቅድ) ለማጣቀሻ አልተገኘም።")
        return model_p, df_info_p, features_p_list # df_info_p is the full dataframe for ranges

    pipeline_planner, df_info_planner, model_features_planner = load_planner_model_and_data_info(MODEL_PIPELINE_FILE_PATH_PLANNER, DATA_FILE_PATH_PLANNER_INFO)

    if not pipeline_planner: st.error("የእቅድ አስመሳይ ሞዴል መጫን አልተቻለም።"); st.stop()
    
    if not model_features_planner:
        # Fallback if features_p_list couldn't be determined
        # This should match the features the model from App 4 was trained on.
        # Crucially, decide if 'Planting_Month_Num' is part of this model.
        # If not, the form needs to use 'Season'.
        model_features_planner = ['Crop_Year', 'Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
        # If Planting_Month_Num is a desired feature, the model needs to be retrained with it.
        # For now, let's assume the model from App 4 does NOT use Planting_Month_Num explicitly.
        # model_features_planner.append('Planting_Month_Num') # Add this if model supports it
        st.warning("የእቅድ አስመሳይ ሞዴል ግብዓት ዓምዶችን ማግኘት አልተቻለም። ነባሪ ዓምዶች (ያለ መዝሪያ ወር ቁጥር) ጥቅም ላይ ይውላሉ።")
    
    if df_info_planner is None:
        st.warning("የእቅድ አስመሳይ የዳታ መረጃ (ለወሰኖች) መጫን አልተቻለም። ነባሪ የግብዓት ወሰኖች ጥቅም ላይ ይውላሉ።")


    st.subheader("የግብርና እቅድ መለኪያዎችን ያስገቡ")
    if 'app6_scenarios' not in st.session_state: st.session_state.app6_scenarios = []
    if 'app6_scenario_counter' not in st.session_state: st.session_state.app6_scenario_counter = 0

    # Use a unique key for the form based on scenario_counter to ensure it's fresh
    with st.form(key=f"app6_scenario_form_{st.session_state.app6_scenario_counter}"):
        st.write(f"አዲስ እቅድ (Scenario) #{len(st.session_state.app6_scenarios) + 1}")
        current_scenario_inputs_p = {}
        form_cols_p = st.columns(3) # Adjust number of columns as needed
        col_idx_p = 0

        # Define unique values for dropdowns, with fallbacks if df_info_planner is None
        unique_crops_p = sorted(df_info_planner['Crop'].unique()) if df_info_planner is not None and 'Crop' in df_info_planner.columns else ["ስንዴ", "ጤፍ", "በቆሎ"]
        unique_seasons_p = sorted(df_info_planner['Season'].unique()) if df_info_planner is not None and 'Season' in df_info_planner.columns else ["ከረምት", "በልግ", "ሙሉ አመት"]
        unique_states_p = sorted(df_info_planner['State'].unique()) if df_info_planner is not None and 'State' in df_info_planner.columns else ["ኦሮሚያ", "አማራ", "ደቡብ"]
        
        # planting_months_am_p is for display if we map a numeric month. The model itself might use 'Season'.
        planting_months_am_p = {"መስከረም": 9, "ጥቅምት": 10, "ህዳር": 11, "ታህሳስ": 12, "ጥር": 1, "የካቲት": 2, "መጋቢት": 3, "ሚያዝያ": 4, "ግንቦት": 5, "ሰኔ": 6, "ሀምሌ": 7, "ነሐሴ": 8}


        current_real_year_p = datetime.date.today().year
        if 'Crop_Year' in model_features_planner:
            min_cy_p = int(df_info_planner['Crop_Year'].min()) if df_info_planner is not None and 'Crop_Year' in df_info_planner.columns and not df_info_planner['Crop_Year'].empty else current_real_year_p - 5
            max_cy_data_p = int(df_info_planner['Crop_Year'].max()) if df_info_planner is not None and 'Crop_Year' in df_info_planner.columns and not df_info_planner['Crop_Year'].empty else current_real_year_p
            def_cy_p = min(current_real_year_p, max_cy_data_p)
            max_in_cy_p = max(max_cy_data_p, current_real_year_p + 1)
            current_scenario_inputs_p['Crop_Year'] = form_cols_p[col_idx_p % 3].number_input("የሰብል አመት (ለእቅድ)", min_value=min_cy_p, max_value=max_in_cy_p, value=def_cy_p, step=1, key=f"app6_year_{st.session_state.app6_scenario_counter}")
            col_idx_p += 1

        # If model uses 'Planting_Month_Num'
        if 'Planting_Month_Num' in model_features_planner:
            sel_month_p_name = form_cols_p[col_idx_p % 3].selectbox("የመዝሪያ ወር (ለእቅድ)", list(planting_months_am_p.keys()), key=f"app6_month_{st.session_state.app6_scenario_counter}")
            current_scenario_inputs_p['Planting_Month_Num'] = planting_months_am_p[sel_month_p_name]
            col_idx_p += 1
        # Else if model uses 'Season' (more likely for App 4 model)
        elif 'Season' in model_features_planner:
             current_scenario_inputs_p['Season'] = form_cols_p[col_idx_p % 3].selectbox("ወቅት (ለእቅድ)", unique_seasons_p, key=f"app6_season_{st.session_state.app6_scenario_counter}")
             col_idx_p +=1
        
        if 'Crop' in model_features_planner: 
            current_scenario_inputs_p['Crop'] = form_cols_p[col_idx_p % 3].selectbox("የሰብል አይነት (ለእቅድ)", unique_crops_p, key=f"app6_crop_{st.session_state.app6_scenario_counter}")
            col_idx_p += 1
        if 'State' in model_features_planner: 
            current_scenario_inputs_p['State'] = form_cols_p[col_idx_p % 3].selectbox("ክልል/ግዛት (ለእቅድ)", unique_states_p, key=f"app6_state_{st.session_state.app6_scenario_counter}")
            col_idx_p += 1

        num_inputs_def_p = {
            'Area': ('የለማ መሬት ስፋት (ሄክታር)', 10.0, 1.0, "%.2f"), 
            'Annual_Rainfall': ('አመታዊ የዝናብ መጠን (mm)', 1000.0, 50.0, "%.2f"), 
            'Fertilizer': ('የማዳበሪያ መጠን (kg)', 10000.0, 1000.0, "%.2f"), 
            'Pesticide': ('የፀረ-ተባይ መጠን (kg/L)', 100.0, 10.0, "%.2f")
        }
        for feature, (label, def_val, step, fmt) in num_inputs_def_p.items():
            if feature in model_features_planner:
                min_val_p = float(df_info_planner[feature].min()) if df_info_planner is not None and feature in df_info_planner.columns and not df_info_planner[feature].empty else 0.0
                max_val_p = float(df_info_planner[feature].max()) if df_info_planner is not None and feature in df_info_planner.columns and not df_info_planner[feature].empty else def_val * 5 # Max 5 times default if no data
                mean_val_p = float(df_info_planner[feature].mean()) if df_info_planner is not None and feature in df_info_planner.columns and not df_info_planner[feature].empty else def_val
                current_scenario_inputs_p[feature] = form_cols_p[col_idx_p % 3].number_input(label, min_value=min_val_p, max_value=max_val_p, value=mean_val_p, step=step, format=fmt, key=f"app6_{feature}_{st.session_state.app6_scenario_counter}")
                col_idx_p += 1
        
        submit_button_planner = st.form_submit_button(label="➕ ይህንን እቅድ ጨምር እና ተንብይ (ለአስመሳይ)")

    if submit_button_planner:
        # Create a DataFrame for prediction based on model_features_planner
        scenario_data_for_prediction = {}
        for feature_name in model_features_planner:
            if feature_name in current_scenario_inputs_p:
                scenario_data_for_prediction[feature_name] = [current_scenario_inputs_p[feature_name]]
            else:
                scenario_data_for_prediction[feature_name] = [np.nan] # Should be handled by preprocessor if trained to do so
                st.warning(f"የእቅድ ግብዓት '{feature_name}' በቅጹ ውስጥ አልተገኘም። በ NaN ይሞላል።")
        
        scenario_df_p = pd.DataFrame(scenario_data_for_prediction)[model_features_planner] # Ensure order

        try:
            with st.spinner("የምርታማነት ትንበያ (ለእቅድ) እየተሰራ ነው..."):
                predicted_yield_p = pipeline_planner.predict(scenario_df_p)
                final_yield_p = max(0, predicted_yield_p[0]) # Ensure non-negative
            
            # Store the original inputs and the prediction for display
            scenario_to_store = current_scenario_inputs_p.copy()
            scenario_to_store['Predicted_Yield'] = round(final_yield_p, 3)
            
            st.session_state.app6_scenarios.append(scenario_to_store)
            st.session_state.app6_scenario_counter += 1 # To refresh the form
            st.success(f"እቅድ #{len(st.session_state.app6_scenarios)} ተጨምሯል! የተገመተው ምርታማነት፦ {final_yield_p:.3f}")
            st.rerun() # Rerun to update the displayed scenarios and clear the form
        except Exception as e_p_predict: 
            st.error(f"የእቅድ ትንበያ ላይ ስህተት ተከስቷል፦ {e_p_predict}")
            st.error(f"ለትንበያ የቀረቡ ዓምዶች (ለእቅድ አስመሳይ)፦ {scenario_df_p.columns.tolist()}")


    if st.session_state.app6_scenarios:
        st.subheader("የግብርና እቅድ ማነጻጸሪያ")
        scenarios_display_df_p = pd.DataFrame(st.session_state.app6_scenarios)
        
        # Prepare columns for display, including mapping month number to name if used
        display_cols_order_p = []
        if 'Planting_Month_Num' in scenarios_display_df_p.columns:
            num_to_month_am_p = {v: k for k, v in planting_months_am_p.items()}
            # Ensure the column exists before trying to map
            if 'Planting_Month_Num' in scenarios_display_df_p:
                 scenarios_display_df_p['የመዝሪያ ወር'] = scenarios_display_df_p['Planting_Month_Num'].map(num_to_month_am_p)
                 display_cols_order_p.append('የመዝሪያ ወር')
        
        # Add other input features that were part of the scenario
        for feature_p in current_scenario_inputs_p.keys(): # Use keys from the last submitted scenario as a guide
            if feature_p not in ['Predicted_Yield', 'Planting_Month_Num'] and feature_p in scenarios_display_df_p.columns:
                 display_cols_order_p.append(feature_p)

        if 'Predicted_Yield' in scenarios_display_df_p.columns: 
            display_cols_order_p.append('Predicted_Yield')
        
        # Ensure all columns in display_cols_order_p actually exist in scenarios_display_df_p
        final_display_columns_p = [col for col in display_cols_order_p if col in scenarios_display_df_p.columns]
        
        if final_display_columns_p:
            if 'Predicted_Yield' in final_display_columns_p:
                st.dataframe(scenarios_display_df_p[final_display_columns_p].sort_values(by='Predicted_Yield', ascending=False).reset_index(drop=True))
            else:
                 st.dataframe(scenarios_display_df_p[final_display_columns_p].reset_index(drop=True))
        else:
            st.write("ለማሳየት ምንም የተዘጋጁ የእቅድ ዓምዶች የሉም።")


        if st.button("🗑️ ሁሉንም እቅዶች አጽዳ (ለአስመሳይ)", key="app6_clear_scenarios_btn"):
            st.session_state.app6_scenarios = []
            st.session_state.app6_scenario_counter = 0 # Reset counter to ensure form key is fresh
            st.rerun()
    else: st.info("እስካሁን ምንም የግብርና እቅድ (ለአስመሳይ) አላከሉም። ከላይ ካለው ቅጽ በመሙላት ይጀምሩ።")


# ==============================================================================
# --- ዋና የመተግበሪያ አሰሳ (Navigation) ---
# ==============================================================================
if 'main_selected_app_name' not in st.session_state:
    st.session_state.main_selected_app_name = "🏠 የመነሻ ገጽ"

# Construct path for logo using the helper function
logo_path = get_project_file_path("assets", "smart_agri_logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=100)
else:
    st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=100) # Fallback

st.sidebar.title("የመሳሪያዎች ምርጫ")

app_options_main = {
    "🏠 የመነሻ ገጽ": None,
    "1. የእንጀራ ጥራት ምርመራ": run_injera_quality_app,
    "2. የወተት ጥራት ትንበያ": run_milk_spoilage_app,
    "3. የማዳበሪያ ምክረ ሀሳብ": run_fertilizer_recommendation_app,
    "4. የሰብል ምርት ትንበያ": run_crop_yield_app,
    "5. የግብርና አማካሪ (AI Chatbot)": run_chatbot_app,
    "6. የግብርና እቅድ አስመሳይ": run_agri_planner_app
}

# Callback to update selected app name in session state
def update_main_selected_app():
    st.session_state.main_selected_app_name = st.session_state._main_selectbox_app_selection

# Use st.session_state.main_selected_app_name to set the index for the selectbox
try:
    current_selection_index = list(app_options_main.keys()).index(st.session_state.main_selected_app_name)
except ValueError:
    current_selection_index = 0 # Default to home page if current selection is invalid

st.sidebar.selectbox(
    "እባክዎ ሊጠቀሙበት የሚፈልጉትን መሳሪያ ይምረጡ፦",
    list(app_options_main.keys()),
    index=current_selection_index,
    on_change=update_main_selected_app,
    key="_main_selectbox_app_selection" 
)

# Display the selected app
if st.session_state.main_selected_app_name == "🏠 የመነሻ ገጽ":
    st.header("🌾 እንኳን ወደ ሁሉን አቀፍ የግብርና እና ምግብ ስርዓት በደህና መጡ!")
    st.markdown("""
    ይህ መድረክ የተለያዩ የግብርና እና ምግብ ነክ ችግሮችን ለመፍታት የሚያግዙ ዘመናዊ የቴክኖሎጂ መፍትሄዎችን ያቀርባል።
    ከግራ በኩል ካለው ምናሌ በመምረጥ የሚፈልጉትን መሳሪያ መጠቀም ይችላሉ።

    **የሚገኙ መሳሪያዎች ዝርዝር፦**
    - **የእንጀራ ጥራት ምርመራ:** የእንጀራ ፎቶ በመጠቀም የጥራት ደረጃውን በፍጥነት ይወቁ።
    - **የወተት ጥራት ትንበያ:** ጥሬ ወተት ላይ ያሉ መረጃዎችን በመጠቀም የመበላሸት ስጋቱን ይተንብዩ።
    - **የማዳበሪያ ምክረ ሀሳብ:** ለአፈርዎ እና ለሰብልዎ አይነት ተስማሚ የሆነውን ማዳበሪያ ይወቁ።
    - **የሰብል ምርት ትንበያ:** የተለያዩ ግብዓቶችን መሠረት በማድረግ የሚጠበቀውን ምርት ይገምቱ።
    - **የግብርና አማካሪ (AI Chatbot):** ስለ ግብርና እና ምግብ ነክ ጉዳዮች በአማርኛ ጥያቄዎችን ይጠይቁና መልስ ያግኙ።
    - **የግብርና እቅድ አስመሳይ:** የተለያዩ የግብርና ዕቅዶችን በማስመሰል የተሻለውን ምርጫ እንዲያደርጉ ያግዛል።

    ይህ ፕሮጀክት የተዘጋጀው የግብርናውን ዘርፍ በቴክኖሎጂ ለማገዝ ነው።
    """)
    st.image(logo_path if os.path.exists(logo_path) else "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
else:
    app_function = app_options_main.get(st.session_state.main_selected_app_name)
    if app_function:
        app_function()
    else:
        st.error("የማይታወቅ ምርጫ። እባክዎ እንደገና ይሞክሩ።")

st.sidebar.markdown("---")
st.sidebar.info("© 2024 ዘመናዊ የግብርና መፍትሄዎች")