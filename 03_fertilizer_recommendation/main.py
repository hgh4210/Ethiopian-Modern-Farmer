import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To convert text data to numbers
from sklearn.compose import ColumnTransformer # To process different columns in different ways
from sklearn.pipeline import Pipeline # To create a sequence of processing and model steps
from sklearn.metrics import accuracy_score # To measure the model's accuracy
import joblib # To save and load the model
import os # To work with the file system

# --- Page Setup and Title (in Amharic) ---
st.set_page_config(page_title="የማዳበሪያ ምክረ ሀሳብ", layout="wide")
st.title("🌱 የማዳበሪያ ምክረ ሀሳብ ስርዓት")
st.markdown("የአካባቢዎን እና የሰብልዎን መረጃ በማስገባት ተስማሚውን ማዳበሪያ ይወቁ።")

# --- File Paths (Constants) ---
DATA_FILE_PATH = 'fertilizer_recommendation_data.csv'
MODEL_PIPELINE_FILE_PATH = 'fertilizer_model_pipeline.joblib'
TARGET_ENCODER_FILE_PATH = 'fertilizer_target_encoder.joblib'

@st.cache_data # To cache the data so it doesn't reload repeatedly (for speed)
def load_data(file_path):
    # This function loads data from a CSV file
    try:
        df = pd.read_csv(file_path)
        # Clean column names from extra spaces (if any)
        df.columns = df.columns.str.strip()
        # Check for missing values and fill them in a simple way (this is for demonstration; a better method is needed in practice)
        if df.isnull().sum().any():
            st.warning("በዳታው ውስጥ የጎደሉ እሴቶች ተገኝተዋል። በቀላል መንገድ ተሞልተዋል።") # Warning: Missing values found in the data. They have been filled in a simple way.
            for col in df.select_dtypes(include=np.number).columns: # Numerical columns
                df[col].fillna(df[col].median(), inplace=True) # Fill with median
            for col in df.select_dtypes(include='object').columns: # Text columns
                df[col].fillna(df[col].mode()[0], inplace=True) # Fill with the most frequent value
        return df
    except FileNotFoundError: # If the file is not found
        st.error(f"ስህተት፦ የዳታ ፋይል '{file_path}' አልተገኘም። ፋይሉ መኖሩን ያረጋግጡ።") # Error: Data file '{file_path}' not found. Please ensure the file exists.
        return None
    except Exception as e: # If any other error occurs
        st.error(f"ዳታውን በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}") # Error occurred while loading data: {e}
        return None

df_fertilizer = load_data(DATA_FILE_PATH) # Load data

# --- Train Model or Load Pre-trained Model ---
@st.cache_resource # To cache the trained model/pipeline so it doesn't reload repeatedly (for speed)
def train_or_load_model_pipeline(data_frame):
    # This function trains the model or loads it if it's already trained
    if data_frame is None: # If there's no data, nothing can be done
        return None, None, None, None, None

    # The target variable (the column we are predicting) is 'Fertilizer Name'
    target_col = 'Fertilizer Name'
    X = data_frame.drop(target_col, axis=1) # Input columns (features)
    y_raw = data_frame[target_col] # Target column (labels)

    # Convert the target variable (fertilizer names) to numbers (Label Encoding)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    # Identify categorical (text) and numerical columns
    categorical_features = ['Soil Type', 'Crop Type'] # Text inputs
    numerical_features = X.select_dtypes(include=np.number).columns.tolist() # Numerical inputs

    # Create a data processing pipeline
    # ColumnTransformer: Helps to prepare different columns in different ways
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features), # Pass through numbers as they are (or StandardScaler/MinMaxScaler can be used)
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features) # One-Hot Encode text columns
        ],
        remainder='passthrough' # Pass through other unspecified columns as they are
    )

    # Create the full pipeline (preprocessing and model)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # Step 1: Data processing
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')) # Step 2: Random Forest model
    ])

    # Load the model if it has already been trained and saved
    if os.path.exists(MODEL_PIPELINE_FILE_PATH) and os.path.exists(TARGET_ENCODER_FILE_PATH):
        try:
            loaded_pipeline = joblib.load(MODEL_PIPELINE_FILE_PATH)
            loaded_target_encoder = joblib.load(TARGET_ENCODER_FILE_PATH)
            st.sidebar.success("የሰለጠነ ሞዴል እና ኢንኮደር ተጭኗል።") # Trained model and encoder loaded.
            # For testing, show the accuracy of the trained pipeline (this is optional)
            # X_train_temp, X_test_temp, _, y_test_encoded_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            # accuracy_loaded = loaded_pipeline.score(X_test_temp, y_test_encoded_temp)
            # st.sidebar.metric(label="የተጫነው ሞዴል ትክክለኛነት", value=f"{accuracy_loaded*100:.2f}%") # Accuracy of the loaded model
            return loaded_pipeline, loaded_target_encoder, data_frame[categorical_features[0]].unique(), data_frame[categorical_features[1]].unique(), data_frame # Return pipeline, encoder, and unique values
        except Exception as e:
            st.sidebar.warning(f"የተቀመጠውን ሞዴል/ኢንኮደር መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...") # Could not load saved model/encoder ({e}). Training a new one...

    # Train the model (if not previously trained)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 'stratify=y' ensures the target distribution is similar in training and test data
    model_pipeline.fit(X_train, y_train) # Train the model on the training data

    # Evaluate the model's accuracy on the test data
    accuracy = model_pipeline.score(X_test, y_test)
    st.sidebar.metric(label="የሙከራ ሞዴል ትክክለኛነት", value=f"{accuracy*100:.2f}%") # Test model accuracy

    # Save the model and target encoder for future use
    try:
        joblib.dump(model_pipeline, MODEL_PIPELINE_FILE_PATH)
        joblib.dump(target_encoder, TARGET_ENCODER_FILE_PATH)
        st.sidebar.success("ሞዴል እና ኢንኮደር ሰልጥነው ተቀምጠዋል።") # Model and encoder trained and saved.
    except Exception as e:
        st.sidebar.error(f"ሞዴሉን/ኢንኮደሩን ማስቀመጥ አልተቻለም፦ {e}") # Could not save model/encoder: {e}

    return model_pipeline, target_encoder, data_frame[categorical_features[0]].unique(), data_frame[categorical_features[1]].unique(), data_frame

# Load or train the model/pipeline
if df_fertilizer is not None: # Only if the data was loaded correctly
    pipeline, LEncoder, unique_soil_types, unique_crop_types, df_for_ranges = train_or_load_model_pipeline(df_fertilizer)
else: # If the data was not loaded
    pipeline, LEncoder, unique_soil_types, unique_crop_types, df_for_ranges = None, None, [], [], None
    st.stop() # Stop the application

# --- User Inputs (in Amharic) ---
st.sidebar.header("መረጃ ያስገቡ") # Enter Information

def get_user_inputs(soil_types_list, crop_types_list, data_for_input_ranges):
    # This function collects user inputs from the sidebar
    # Get approximate ranges for numerical inputs from the main data (for sliders)
    temp_min, temp_max = int(data_for_input_ranges['Temprature'].min()), int(data_for_input_ranges['Temprature'].max())
    hum_min, hum_max = int(data_for_input_ranges['Humidity'].min()), int(data_for_input_ranges['Humidity'].max())
    moist_min, moist_max = int(data_for_input_ranges['Moisture'].min()), int(data_for_input_ranges['Moisture'].max())
    n_min, n_max = int(data_for_input_ranges['Nitrogen'].min()), int(data_for_input_ranges['Nitrogen'].max())
    k_min, k_max = int(data_for_input_ranges['Potassium'].min()), int(data_for_input_ranges['Potassium'].max())
    p_min, p_max = int(data_for_input_ranges['Phosphorous'].min()), int(data_for_input_ranges['Phosphorous'].max())

    # Collect inputs using Streamlit controls (sliders, selectbox)
    Temprature = st.sidebar.slider('የአየር ሙቀት መጠን (°C)', temp_min, temp_max, int(data_for_input_ranges['Temprature'].mean())) # Air Temperature (°C)
    humidity = st.sidebar.slider('የአየር ንብረት እርጥበት (%)', hum_min, hum_max, int(data_for_input_ranges['Humidity'].mean())) # Air Humidity (%)
    moisture = st.sidebar.slider('የአፈር እርጥበት (%)', moist_min, moist_max, int(data_for_input_ranges['Moisture'].mean())) # Soil Moisture (%)
    soil_type = st.sidebar.selectbox('የአፈር አይነት', sorted(list(soil_types_list))) # Soil Type
    crop_type = st.sidebar.selectbox('የሰብል አይነት', sorted(list(crop_types_list))) # Crop Type
    nitrogen = st.sidebar.slider('የናይትሮጅን መጠን (kg/ha)', n_min, n_max, int(data_for_input_ranges['Nitrogen'].mean())) # Nitrogen Amount (kg/ha)
    potassium = st.sidebar.slider('የፖታሲየም መጠን (kg/ha)', k_min, k_max, int(data_for_input_ranges['Potassium'].mean())) # Potassium Amount (kg/ha)
    phosphorous = st.sidebar.slider('የፎስፈረስ መጠን (kg/ha)', p_min, p_max, int(data_for_input_ranges['Phosphorous'].mean())) # Phosphorous Amount (kg/ha)

    # Organize the collected inputs into a Pandas DataFrame
    user_data = pd.DataFrame({
        'Temprature': [Temprature],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Soil Type': [soil_type],
        'Crop Type': [crop_type],
        'Nitrogen': [nitrogen],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous]
    })
    return user_data

# Get user inputs
if unique_soil_types is not None and unique_crop_types is not None and df_for_ranges is not None:
    input_df = get_user_inputs(unique_soil_types, unique_crop_types, df_for_ranges)
else:
    st.error("የአፈር ወይም የሰብል አይነቶችን ከዳታው ማግኘት አልተቻለም።") # Could not get soil or crop types from the data.
    st.stop()

# --- Display Input Data (in Amharic) ---
st.subheader('እርስዎ ያስገቡት መረጃ፦') # Information you entered:
st.dataframe(input_df) # Display the DataFrame in a nice table

# --- Prediction and Result Display (in Amharic) ---
if pipeline and LEncoder and input_df is not None:
    if st.button('💡 ምክረ ሀሳብ አግኝ', use_container_width=True): # Get Recommendation
        with st.spinner("ምክረ ሀሳብ እየተዘጋጀ ነው..."): # Recommendation is being prepared...
            try:
                # Make a prediction using the pipeline
                prediction_encoded = pipeline.predict(input_df)
                # Convert the predicted numerical value back to the original fertilizer name
                predicted_fertilizer_name = LEncoder.inverse_transform(prediction_encoded)

                st.subheader('የማዳበሪያ ምክረ ሀሳብ፦') # Fertilizer Recommendation:
                st.success(f"ለእርስዎ ሁኔታ ተስማሚ የሆነው ማዳበሪያ፦ **{predicted_fertilizer_name[0]}** ነው") # The suitable fertilizer for your condition is: **{predicted_fertilizer_name[0]}**
                st.balloons() # To add a little joy!

                # (Optional) Display confidence level (if the model supports predict_proba)
                if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
                    prediction_probabilities = pipeline.predict_proba(input_df)
                    st.subheader("የመተማመን ደረጃ ለእያንዳንዱ የማዳበሪያ አይነት (በግምት)፦") # Confidence level for each fertilizer type (approximate):
                    # Get class names from the encoder
                    class_names = LEncoder.classes_
                    proba_df = pd.DataFrame(prediction_probabilities, columns=class_names)
                    # Show the top few most likely ones
                    top_n = 3
                    top_probas = proba_df.T.sort_values(by=0, ascending=False).head(top_n)
                    top_probas.columns = ["የመሆን እድል (%)"] # Probability (%)
                    top_probas["የመሆን እድል (%)"] = top_probas["የመሆን እድል (%)"].apply(lambda x: f"{x*100:.2f}%")
                    st.table(top_probas)

            except Exception as e:
                st.error(f"ምክረ ሀሳብ በማዘጋጀት ላይ ሳለ ስህተት ተከስቷል፦ {e}") # An error occurred while preparing the recommendation: {e}
else:
    st.error("ሞዴሉ ወይም ኢንኮደሩ አልተጫነም። እባክዎ እንደገና ይሞክሩ ወይም የዳታ ፋይሉን ያረጋግጡ።") # Model or encoder not loaded. Please try again or check the data file.

# (Optional) Display an overview of the data
st.markdown("---")
if st.checkbox('የመጀመሪያውን የማዳበሪያ ዳታ ናሙና ይመልከቱ (በእንግሊዝኛ)'): # View a sample of the original fertilizer data (in English)
    st.subheader('የማዳበሪያ ዳታ ናሙና (ከ fertilizer_recommendation_data.csv):') # Fertilizer Data Sample (from fertilizer_recommendation_data.csv):
    if df_fertilizer is not None:
        st.dataframe(df_fertilizer.head())
    else:
        st.write("ዳታ አልተጫነም።") # Data not loaded.

st.markdown("---")
