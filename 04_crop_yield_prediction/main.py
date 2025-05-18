import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # For Regression problem
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # To prepare text and numerical data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error # To evaluate the Regression model
import joblib
import os

# --- Page Setup and Title (in Amharic) ---
st.set_page_config(page_title="የሰብል ምርት ትንበያ", layout="wide")
st.title("🌾 የሰብል ምርት ትንበያ ስርዓት")
st.markdown("የተለያዩ መረጃዎችን በማስገባት የሚጠበቀውን የሰብል ምርታማነት (Yield) ይተንብዩ።")

# --- File Paths (Constants) ---
DATA_FILE_PATH = 'crop_yield_data.csv'
MODEL_PIPELINE_FILE_PATH = 'crop_yield_model_pipeline.joblib'
# For this problem, LabelEncoder is not needed for the target because Yield is a number

@st.cache_data # To cache the data so it doesn't reload repeatedly
def load_data(file_path):
    # This function loads data from a CSV file
    try:
        df = pd.read_csv(file_path)
        # Clean column names from extra spaces
        df.columns = df.columns.str.strip()
        # Handle missing values (this is basic; a better method might be needed)
        # For example, fill numbers with median, texts with mode
        for col in df.select_dtypes(include=np.number).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include='object').columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        # If 'Production' column has non-numeric values (e.g., 'unknown'), correct them
        if 'Production' in df.columns and df['Production'].dtype == 'object':
            df['Production'] = pd.to_numeric(df['Production'], errors='coerce').fillna(df['Production'].median() if pd.to_numeric(df['Production'], errors='coerce').isnull().sum() < len(df) else 0)

        # 'Yield' is the target variable; create or verify it if it doesn't exist
        if 'Yield' not in df.columns:
            if 'Production' in df.columns and 'Area' in df.columns and not df['Area'].eq(0).any(): # Ensure Area is not zero
                 df['Yield'] = df['Production'] / df['Area']
                 df['Yield'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinity values
                 df['Yield'].fillna(df['Yield'].median(), inplace=True) # Fill NaNs with median if any
            else:
                st.error("የ'Yield' ዓምድ በዳታው ውስጥ የለም፣ እና ከ'Production' እና 'Area' መፍጠር አልተቻለም።") # 'Yield' column is not in the data, and could not be created from 'Production' and 'Area'.
                return None
        return df
    except FileNotFoundError:
        st.error(f"ስህተት፦ የዳታ ፋይል '{file_path}' አልተገኘም። ፋይሉ መኖሩን ያረጋግጡ።") # Error: Data file '{file_path}' not found. Please ensure the file exists.
        return None
    except Exception as e:
        st.error(f"ዳታውን በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}") # Error occurred while loading data: {e}
        return None

df_yield = load_data(DATA_FILE_PATH)

# --- Train Model or Load Pre-trained Model ---
@st.cache_resource # To cache the trained model/pipeline so it doesn't reload repeatedly
def train_or_load_yield_model_pipeline(data_frame):
    # This function trains the model or loads it if it's already trained
    if data_frame is None:
        return None, None, None, None, None, None # Added more None for new returns

    # Target variable is 'Yield'
    target_col = 'Yield'
    # Remove 'Production' from input columns (as it's highly correlated with Yield and might not be an input for prediction)
    # We might also remove 'Area' since Yield = Production / Area
    features_to_drop = [target_col, 'Production'] # 'Area' can also be included
    X = data_frame.drop(columns=[col for col in features_to_drop if col in data_frame.columns], axis=1)
    y = data_frame[target_col]

    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include='object').columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Create a data processing pipeline
    # Scale numbers with StandardScaler; encode texts with OneHotEncoder
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Full pipeline (preprocessing and Regression model)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) # n_jobs=-1 uses all CPU cores
    ])

    # Load the model if it has already been trained and saved
    if os.path.exists(MODEL_PIPELINE_FILE_PATH):
        try:
            loaded_pipeline = joblib.load(MODEL_PIPELINE_FILE_PATH)
            st.sidebar.success("የሰለጠነ የሰብል ምርት ትንበያ ሞዴል ተጭኗል።") # Trained crop yield prediction model loaded.
            # For testing, show the R2 score of the trained pipeline (optional)
            # X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42)
            # r2_loaded = loaded_pipeline.score(X_test_temp, y_test_temp) # For regressor, score() returns R2
            # st.sidebar.metric(label="የተጫነው ሞዴል R² ውጤት", value=f"{r2_loaded:.3f}") # R² score of the loaded model
            return (loaded_pipeline,
                    data_frame['Crop'].unique(),
                    data_frame['Season'].unique(),
                    data_frame['State'].unique(),
                    data_frame, # For input ranges
                    X.columns.tolist() # To return input columns
                   )
        except Exception as e:
            st.sidebar.warning(f"የተቀመጠውን ሞዴል መጫን አልተቻለም ({e})። አዲስ በማሰልጠን ላይ...") # Could not load saved model ({e}). Training a new one...

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.sidebar.text("ሞዴሉ እየሰለጠነ ነው... እባክዎ ይጠብቁ።") # Model is training... please wait.
    model_pipeline.fit(X_train, y_train)
    st.sidebar.text("የሞዴል ስልጠና ተጠናቋል።") # Model training completed.

    # Evaluate model performance (R-squared and RMSE)
    r2 = model_pipeline.score(X_test, y_test) # .score() for Regressor returns R2
    y_pred_test = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    st.sidebar.metric(label="የሙከራ ሞዴል R² ውጤት", value=f"{r2:.3f}") # Test model R² score
    st.sidebar.metric(label="የሙከራ ሞዴል RMSE", value=f"{rmse:.3f}") # Test model RMSE


    # Save the model for future use
    try:
        joblib.dump(model_pipeline, MODEL_PIPELINE_FILE_PATH)
        st.sidebar.success("የሰብል ምርት ትንበያ ሞዴል ሰልጥኖ ተቀምጧል።") # Crop yield prediction model trained and saved.
    except Exception as e:
        st.sidebar.error(f"ሞዴሉን ማስቀመጥ አልተቻለም፦ {e}") # Could not save model: {e}

    return (model_pipeline,
            data_frame['Crop'].unique(),
            data_frame['Season'].unique(),
            data_frame['State'].unique(),
            data_frame, # For input ranges
            X.columns.tolist() # To return input columns
           )

# Load or train the model/pipeline
if df_yield is not None:
    pipeline, unique_crops, unique_seasons, unique_states, df_for_ranges, feature_names_for_input = train_or_load_yield_model_pipeline(df_yield)
else:
    pipeline, unique_crops, unique_seasons, unique_states, df_for_ranges, feature_names_for_input = None, [], [], [], None, []
    st.stop()

# --- User Inputs (in Amharic) ---
st.sidebar.header("የትንበያ መረጃ ያስገቡ") # Enter Prediction Information

def get_user_yield_inputs(crops_list, seasons_list, states_list, data_for_input_ranges, available_features):
    # This function collects user inputs from the sidebar
    inputs = {}

    # Get approximate ranges for numerical inputs from the main data (for sliders)
    # And prepare optional text inputs
    default_year = 2020 # or the last year from the data
    if 'Crop_Year' in available_features:
        year_min = int(data_for_input_ranges['Crop_Year'].min()) if 'Crop_Year' in data_for_input_ranges else 1990
        year_max = int(data_for_input_ranges['Crop_Year'].max()) if 'Crop_Year' in data_for_input_ranges else default_year + 5
        inputs['Crop_Year'] = st.sidebar.number_input('የሰብል አመት', min_value=year_min, max_value=year_max, value=default_year) # Crop Year

    if 'Crop' in available_features:
        inputs['Crop'] = st.sidebar.selectbox('የሰብል አይነት', sorted(list(crops_list))) # Crop Type
    if 'Season' in available_features:
        inputs['Season'] = st.sidebar.selectbox('ወቅት', sorted(list(seasons_list))) # Season
    if 'State' in available_features:
        inputs['State'] = st.sidebar.selectbox('ግዛት/ክልል', sorted(list(states_list))) # State/Region

    if 'Area' in available_features:
        area_min = float(data_for_input_ranges['Area'].min()) if 'Area' in data_for_input_ranges else 0.0
        area_max = float(data_for_input_ranges['Area'].max()) if 'Area' in data_for_input_ranges else 100000.0
        area_mean = float(data_for_input_ranges['Area'].mean()) if 'Area' in data_for_input_ranges else 1000.0
        inputs['Area'] = st.sidebar.number_input('የለማ መሬት ስፋት (ሄክታር)', min_value=area_min, max_value=area_max, value=area_mean, step=100.0, format="%.2f") # Cultivated Land Area (hectares)

    if 'Annual_Rainfall' in available_features:
        rf_min = float(data_for_input_ranges['Annual_Rainfall'].min()) if 'Annual_Rainfall' in data_for_input_ranges else 0.0
        rf_max = float(data_for_input_ranges['Annual_Rainfall'].max()) if 'Annual_Rainfall' in data_for_input_ranges else 5000.0
        rf_mean = float(data_for_input_ranges['Annual_Rainfall'].mean()) if 'Annual_Rainfall' in data_for_input_ranges else 1000.0
        inputs['Annual_Rainfall'] = st.sidebar.number_input('አመታዊ የዝናብ መጠን (mm)', min_value=rf_min, max_value=rf_max, value=rf_mean, step=50.0, format="%.2f") # Annual Rainfall (mm)

    if 'Fertilizer' in available_features:
        fert_min = float(data_for_input_ranges['Fertilizer'].min()) if 'Fertilizer' in data_for_input_ranges else 0.0
        fert_max = float(data_for_input_ranges['Fertilizer'].max()) if 'Fertilizer' in data_for_input_ranges else 10000000.0
        fert_mean = float(data_for_input_ranges['Fertilizer'].mean()) if 'Fertilizer' in data_for_input_ranges else 100000.0
        inputs['Fertilizer'] = st.sidebar.number_input('የማዳበሪያ መጠን (kg)', min_value=fert_min, max_value=fert_max, value=fert_mean, step=1000.0, format="%.2f") # Fertilizer Amount (kg)

    if 'Pesticide' in available_features:
        pest_min = float(data_for_input_ranges['Pesticide'].min()) if 'Pesticide' in data_for_input_ranges else 0.0
        pest_max = float(data_for_input_ranges['Pesticide'].max()) if 'Pesticide' in data_for_input_ranges else 50000.0
        pest_mean = float(data_for_input_ranges['Pesticide'].mean()) if 'Pesticide' in data_for_input_ranges else 1000.0
        inputs['Pesticide'] = st.sidebar.number_input('የፀረ-ተባይ መጠን (kg/L)', min_value=pest_min, max_value=pest_max, value=pest_mean, step=100.0, format="%.2f") # Pesticide Amount (kg/L)

    # Organize the collected inputs into a Pandas DataFrame (aligning with the model's input columns)
    user_data_list = {key: [inputs.get(key)] for key in feature_names_for_input if key in inputs}
    user_data_df = pd.DataFrame(user_data_list)

    # Ensure all input columns that were present during model training are included
    # Fill missing ones (those not provided by the user but expected by the model) with NaN (the pipeline will handle them)
    for col in feature_names_for_input:
        if col not in user_data_df.columns:
            user_data_df[col] = np.nan # or a suitable default value

    return user_data_df[feature_names_for_input] # Ensure columns are in the same order as during training

# Get user inputs
if unique_crops is not None and unique_seasons is not None and unique_states is not None and df_for_ranges is not None and feature_names_for_input:
    input_df = get_user_yield_inputs(unique_crops, unique_seasons, unique_states, df_for_ranges, feature_names_for_input)
else:
    st.error("ልዩ የሆኑ የግብዓት አማራጮችን ወይም የግብዓት ዓምዶችን ከዳታው ማግኘት አልተቻለም።") # Could not get unique input options or input columns from the data.
    if not feature_names_for_input: st.warning("የሞዴሉ የግብዓት ዓምዶች ዝርዝር ባዶ ነው።") # The model's input column list is empty.
    st.stop()

# --- Display Input Data (in Amharic) ---
st.subheader('እርስዎ ያስገቡት የትንበያ መረጃ፦') # Prediction information you entered:
st.dataframe(input_df)

# --- Prediction and Result Display (in Amharic) ---
if pipeline and input_df is not None:
    if st.button('🌾 ምርታማነትን ተንብይ', use_container_width=True): # Predict Productivity
        with st.spinner("ትንበያ እየተሰራ ነው..."): # Prediction is being made...
            try:
                # Make a prediction using the pipeline
                predicted_yield = pipeline.predict(input_df)

                st.subheader('የተተነበየው የሰብል ምርታማነት (Yield)፦') # Predicted Crop Productivity (Yield):
                # Ensure productivity is not below zero
                final_yield = max(0, predicted_yield[0])
                st.success(f"በግምት የሚጠበቀው ምርታማነት፦ **{final_yield:.3f}** (መለኪያ አሃዱ በዋናው ዳታ መሰረት)") # Approximately expected productivity: **{final_yield:.3f}** (unit of measurement according to the original data)
                st.caption("(ይህ ትንበያ ሲሆን ትክክለኛው ውጤት በተለያዩ ሁኔታዎች ሊለያይ ይችላል። መለኪያ አሃዱ (ለምሳሌ ቶን/ሄክታር) በዋናው ዳታ 'Yield' ዓምድ መለኪያ መሰረት ነው።)") # (This is a prediction, and the actual result may vary depending on various conditions. The unit of measurement (e.g., tons/hectare) is based on the 'Yield' column unit in the original data.)

                # (Optional) Additional explanation or advice
                if final_yield > (df_for_ranges['Yield'].mean() + df_for_ranges['Yield'].std()): # If above average and standard deviation
                    st.info("ይህ ትንበያ ከአማካይ በላይ የሆነ ምርታማነትን ያሳያል። ጥሩ ሁኔታዎች ከቀጠሉ ከፍተኛ ምርት ሊጠበቅ ይችላል።") # This prediction indicates above-average productivity. High yield can be expected if good conditions continue.
                elif final_yield < (df_for_ranges['Yield'].mean() - df_for_ranges['Yield'].std()): # If below average and standard deviation
                    st.warning("ይህ ትንበያ ከአማካይ በታች የሆነ ምርታማነትን ያሳያል። የምርት ማነቆዎችን መገምገም ጠቃሚ ሊሆን ይችላል።") # This prediction indicates below-average productivity. Evaluating production bottlenecks may be useful.


            except Exception as e:
                st.error(f"ትንበያ በማዘጋጀት ላይ ሳለ ስህተት ተከስቷል፦ {e}") # An error occurred while making the prediction: {e}
                st.code(f"የተጠቃሚ ግብዓት ዓምዶች ብዛት: {len(input_df.columns)}") # Number of user input columns:
                st.code(f"የተጠቃሚ ግብዓት ዓምዶች: {input_df.columns.tolist()}") # User input columns:
                st.code(f"ሞዴሉ የሚጠብቃቸው የግብዓት ዓምዶች: {feature_names_for_input}") # Input columns expected by the model:


else:
    st.error("የትንበያ ሞዴሉ አልተጫነም ወይም የተጠቃሚ ግብዓት ትክክል አይደለም።") # The prediction model is not loaded or the user input is incorrect.

# (Optional) Display an overview of the data
st.markdown("---")
if st.checkbox('የመጀመሪያውን የሰብል ምርት ዳታ ናሙና ይመልከቱ (በእንግሊዝኛ)'): # View a sample of the original crop yield data (in English)
    st.subheader('የሰብል ምርት ዳታ ናሙና (ከ crop_yield_data.csv):') # Crop Yield Data Sample (from crop_yield_data.csv):
    if df_yield is not None:
        st.dataframe(df_yield.head())
    else:
        st.write("ዳታ አልተጫነም።") # Data not loaded.

st.markdown("---")
