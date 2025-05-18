import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # For saving and loading the model
import os

# --- Page Setup and Title ---
st.set_page_config(page_title="የወተት ጥራት ትንበያ", layout="wide")
st.title("🥛 የወተት ጥራት እና የመበላሸት ትንበያ")
st.markdown("የወተትዎን መረጃ በማስገባት የጥራት ደረጃውን ይተንብዩ።")

# --- Data Loading and Preparation ---
DATA_FILE_PATH = 'milknew.csv' # Place this file in the same directory

@st.cache_data # Cache data to prevent reloading
def load_and_preprocess_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Convert 'Grade'
        df['Grade'] = df['Grade'].map({'high': 2, 'medium': 1, 'low': 0})
        # Correcting the space in the 'Fat ' column name
        if 'Fat ' in df.columns:
            df.rename(columns={'Fat ': 'Fat'}, inplace=True)
        # Check for missing values (good practice, though not present in this dataset)
        if df.isnull().sum().any():
            st.warning("በዳታው ውስጥ የጎደሉ እሴቶች አሉ፣ ይህም የትንበያውን ትክክለኛነት ሊቀንስ ይችላል።")
            # df = df.dropna() # or other imputation method
        return df
    except FileNotFoundError:
        st.error(f"ስህተት፦ የዳታ ፋይል '{file_path}' አልተገኘም። እባክዎ ፋይሉ መኖሩን ያረጋግጡ።")
        return None
    except Exception as e:
        st.error(f"ዳታውን በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}")
        return None

df_milk = load_and_preprocess_data(DATA_FILE_PATH)

# --- Model Training or Loading ---
MODEL_FILE_PATH = 'milk_quality_rf_model.joblib'
SCALER_FILE_PATH = 'milk_quality_scaler.joblib'

@st.cache_resource # Cache model and scaler (load only once)
def train_or_load_model_and_scaler(data_frame):
    if data_frame is None:
        return None, None

    X = data_frame.iloc[:, :-1]
    y = data_frame.iloc[:, -1]

    # Load model and scaler if they already exist
    if os.path.exists(MODEL_FILE_PATH) and os.path.exists(SCALER_FILE_PATH):
        try:
            model = joblib.load(MODEL_FILE_PATH)
            scaler = joblib.load(SCALER_FILE_PATH)
            # Ensure scaler is fitted (to transform new data)
            # It might be necessary to check the original X_train data size
            # X_train_temp, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            # scaler.fit(X_train_temp) # or use the pre-fitted one
            st.sidebar.success("የሰለጠነ ሞዴል እና ስኬለር ተጭኗል።")
            return model, scaler
        except Exception as e:
            st.sidebar.warning(f"የተቀመጠውን ሞዴል/ስኬለር መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...")

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random_state for reproducible results

    # Scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest model (high accuracy)
    # depth_vec = np.arange(1, 20, 1) # For finding best depth
    # For simplicity, we use a common good depth or the one found in notebook (e.g., 10-15)
    best_depth = 12 # Good depth found from notebook or experience
    model = RandomForestClassifier(max_depth=best_depth, random_state=0)
    model.fit(X_train_scaled, y_train)

    # Evaluate model accuracy
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.sidebar.metric(label="የሙከራ ሞዴል ትክክለኛነት", value=f"{accuracy*100:.2f}%")

    # Save the model and scaler
    try:
        joblib.dump(model, MODEL_FILE_PATH)
        joblib.dump(scaler, SCALER_FILE_PATH)
        st.sidebar.success("ሞዴል እና ስኬለር ሰልጥነው ተቀምጠዋል።")
    except Exception as e:
        st.sidebar.error(f"ሞዴሉን/ስኬለሩን ማስቀመጥ አልተቻለም፦ {e}")

    return model, scaler

if df_milk is not None:
    model, scaler = train_or_load_model_and_scaler(df_milk)
else:
    model, scaler = None, None
    st.stop() # Stop the app if data is not available

# --- User Inputs ---
st.sidebar.header("የወተት መረጃ ያስገቡ")

def user_input_features(data_frame):
    # Get approximate ranges from the data
    ph_min, ph_max = float(data_frame['pH'].min()), float(data_frame['pH'].max())
    temp_min, temp_max = int(data_frame['Temprature'].min()), int(data_frame['Temprature'].max())
    # Taste, Odor, Fat, Turbidity are 0 or 1
    # Colour min, max
    colour_min, colour_max = int(data_frame['Colour'].min()), int(data_frame['Colour'].max())

    ph = st.sidebar.slider('ፒኤች (pH)', ph_min, ph_max, float(data_frame['pH'].mean()), 0.1)
    temprature = st.sidebar.slider('የሙቀት መጠን (°C)', temp_min, temp_max, int(data_frame['Temprature'].mean()))
    taste = st.sidebar.selectbox('ጣዕም (0=ክፉ, 1=ጥሩ)', (0, 1))
    odor = st.sidebar.selectbox('ሽታ (0=የለም, 1=አለ)', (0, 1))
    fat = st.sidebar.selectbox('የስብ መጠን (0=ዝቅተኛ, 1=ከፍተኛ)', (0, 1)) # 'Fat ' was the original column name
    turbidity = st.sidebar.selectbox('ደብዛውነት (0=የለም, 1=አለ)', (0, 1))
    colour = st.sidebar.slider('ቀለም (እሴት)', colour_min, colour_max, int(data_frame['Colour'].mean()))

    data = {
        'pH': ph,
        'Temprature': temprature,
        'Taste': taste,
        'Odor': odor,
        'Fat': fat, # Ensure this matches the renamed column
        'Turbidity': turbidity,
        'Colour': colour
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features(df_milk)

# --- Displaying Input Data ---
st.subheader('እርስዎ ያስገቡት የወተት መረጃ፦')
st.write(input_df)

# --- Prediction and Result Display ---
if model and scaler:
    # Scale the input data
    input_df_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_df_scaled)
    prediction_proba = model.predict_proba(input_df_scaled)

    st.subheader('የትንበያ ውጤት፦')
    grade_map_amharic = {2: "ከፍተኛ ጥራት (High)", 1: "መካከለኛ ጥራት (Medium)", 0: "ዝቅተኛ ጥራት (Low)"}
    predicted_grade_amharic = grade_map_amharic.get(prediction[0], "ያልታወቀ")

    st.success(f"የተገመተው የወተት ጥራት ደረጃ፦ **{predicted_grade_amharic}**")

    if prediction[0] == 2: # High
        st.balloons()
    elif prediction[0] == 0: # Low
        st.warning("ይህ ወተት ዝቅተኛ ጥራት ያለው ወይም የመበላሸት ስጋት ሊኖረው ይችላል።")

    st.subheader('የመተማመን ደረጃ (Confidence) ለእያንዳንዱ የጥራት መደብ፦')
    proba_df = pd.DataFrame({
        "የጥራት መደብ": [grade_map_amharic[0], grade_map_amharic[1], grade_map_amharic[2]],
        "የመተማመን ዕድል": [f"{p*100:.2f}%" for p in prediction_proba[0]]
    })
    st.table(proba_df)

else:
    st.error("ሞዴሉ ወይም ስኬለሩ አልተጫነም። እባክዎ እንደገና ይሞክሩ ወይም የዳታ ፋይሉን ያረጋግጡ።")

# (Optional) Display general data overview
st.markdown("---")
if st.checkbox('የመጀመሪያውን የወተት ዳታ ናሙና ይመልከቱ (በእንግሊዝኛ)'):
    st.subheader('የወተት ዳታ ናሙና (ከ milknew.csv):')
    st.write(df_milk.head())
    # Display meaning of 'Grade'
    st.caption("የ'Grade' ትርጉም በዳታው ውስጥ፦ 2=High (ከፍተኛ), 1=Medium (መካከለኛ), 0=Low (ዝቅተኛ)")

