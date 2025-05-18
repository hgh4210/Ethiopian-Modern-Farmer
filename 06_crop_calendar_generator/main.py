import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime # To get the current year

# --- Page Setup and Title (in Amharic) ---
st.set_page_config(page_title="የግብርና እቅድ አስመሳይ", layout="wide")
st.title("🛠️ የግብርና እቅድ አስመሳይ እና ማመቻቻ") # Agricultural Plan Simulator and Optimizer
st.markdown("""
ይህ መሳሪያ የተለያዩ የግብርና ግብዓቶችን እና የመዝሪያ ጊዜዎችን በማስገባት የሚጠበቀውን የሰብል ምርታማነት (Yield) ለመገመት ይረዳዎታል።
የተለያዩ አማራጮችን በማወዳደር የተሻለውን የግብርና እቅድ ለማውጣት ይጠቀሙበት።
**ማሳሰቢያ:** ይህ አስመሳይ ከዚህ በፊት በሰለጠነው የሰብል ምርት ትንበያ ሞዴል ላይ የተመሰረተ ነው።
""")
# This tool helps you estimate the expected crop productivity (Yield) by entering various agricultural inputs and planting times.
# Use it to compare different options and develop the best agricultural plan.
# **Note:** This simulator is based on the previously trained crop yield prediction model.

# --- Load the Trained Crop Yield Prediction Model ---
MODEL_PIPELINE_FILE_PATH_FROM_PROJ4 = os.path.join("..", "04_crop_yield_prediction", "crop_yield_model_pipeline.joblib")
DATA_FILE_PATH_FROM_PROJ4 = os.path.join("..", "04_crop_yield_prediction", "crop_yield_data.csv")

@st.cache_resource
def load_prediction_model_and_data_info(model_path, data_path):
    model_pipeline = None
    df_for_info = None
    feature_names = None

    if not os.path.exists(model_path):
        st.error(f"ስህተት፦ የሰብል ምርት ትንበያ ሞዴል ፋይል ({model_path}) አልተገኘም። እባክዎ መጀመሪያ የፕሮጀክት 4ን ሞዴል ያሰልጥኑ።") # Error: Crop yield prediction model file ({model_path}) not found. Please train Project 4's model first.
        return None, None, None

    try:
        model_pipeline = joblib.load(model_path)
        st.sidebar.success("የሰብል ምርት ትንበያ ሞዴል ተጭኗል።") # Crop yield prediction model loaded.
    except Exception as e:
        st.error(f"የትንበያ ሞዴሉን በመጫን ላይ ሳለ ስህተት ተከስቷል፦ {e}") # An error occurred while loading the prediction model: {e}
        return None, None, None

    if os.path.exists(data_path):
        try:
            df_temp = pd.read_csv(data_path)
            df_temp.columns = df_temp.columns.str.strip()
            # Correct non-numeric values in 'Production' column
            if 'Production' in df_temp.columns and df_temp['Production'].dtype == 'object':
                 df_temp['Production'] = pd.to_numeric(df_temp['Production'], errors='coerce')
                 # Calculate Production median only from numeric values
                 numeric_production = df_temp['Production'].dropna()
                 if not numeric_production.empty:
                     df_temp['Production'].fillna(numeric_production.median(), inplace=True)
                 else: # If all Production are NaN (or if it was empty initially)
                     df_temp['Production'].fillna(0, inplace=True) # or another suitable value


            # 'Yield' is the target variable; create or verify it if it doesn't exist
            if 'Yield' not in df_temp.columns:
                if 'Production' in df_temp.columns and 'Area' in df_temp.columns :
                    # Prevent division by zero for Area
                    df_temp['Yield'] = df_temp.apply(lambda row: row['Production'] / row['Area'] if row['Area'] != 0 else 0, axis=1)
                    df_temp['Yield'].replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_temp['Yield'].fillna(df_temp['Yield'].median(), inplace=True)

            features_to_drop_info = ['Yield', 'Production']
            df_for_info = df_temp.drop(columns=[col for col in features_to_drop_info if col in df_temp.columns], axis=1)
            feature_names = df_for_info.columns.tolist()
        except Exception as e:
            st.warning(f"የቀድሞውን ዳታ ፋይል ({data_path}) ለማጣቀሻ መጫን አልተቻለም፦ {e}") # Could not load the previous data file ({data_path}) for reference: {e}
    else:
        st.warning(f"የቀድሞው ዳታ ፋይል ({data_path}) ለማጣቀሻ አልተገኘም። የግብዓት አማራጮች ውስን ይሆናሉ።") # The previous data file ({data_path}) for reference was not found. Input options will be limited.

    return model_pipeline, df_for_info, feature_names

pipeline, df_info, model_features = load_prediction_model_and_data_info(MODEL_PIPELINE_FILE_PATH_FROM_PROJ4, DATA_FILE_PATH_FROM_PROJ4)

if not pipeline:
    st.stop()

if not model_features and df_info is not None:
    model_features = df_info.columns.tolist()
elif not model_features and df_info is None:
    st.warning("የሞዴሉን የግብዓት ዓምዶች ማወቅ አልተቻለም። ነባሪ ዓምዶች ጥቅም ላይ ይውላሉ።") # Could not determine the model's input columns. Default columns will be used.
    model_features = ['Crop_Year', 'Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

st.header("1. የእቅድ መለኪያዎችን ያስገቡ") # Enter Plan Parameters
st.markdown("ለተለያዩ የመዝሪያ ወራት、 የሰብል አይነቶች、 ወዘተ... የሚጠበቀውን ምርት ለማየት ከታች ያሉትን መረጃዎች ያስተካክሉ።") # Adjust the information below to see the expected yield for different planting months, crop types, etc.

if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []
if 'scenario_counter' not in st.session_state:
    st.session_state.scenario_counter = 0

# --- Single Plan (Scenario) Input Form ---
# Using a unique key for the form based on scenario_counter to ensure it's fresh after rerun
with st.form(key=f"scenario_form_{st.session_state.scenario_counter}"):
    st.subheader(f"አዲስ እቅድ (Scenario) #{len(st.session_state.scenarios) + 1}") # New Plan (Scenario)
    current_scenario_inputs = {}
    cols_per_row = 3
    form_cols = st.columns(cols_per_row)
    col_idx = 0

    unique_crops = sorted(df_info['Crop'].unique()) if df_info is not None and 'Crop' in df_info.columns else ["ስንዴ", "ጤፍ", "በቆሎ"] # Wheat, Teff, Maize
    unique_seasons = sorted(df_info['Season'].unique()) if df_info is not None and 'Season' in df_info.columns else ["ከረምት", "በልግ", "ሙሉ አመት"] # Kiremt, Belg, Full Year
    unique_states = sorted(df_info['State'].unique()) if df_info is not None and 'State' in df_info.columns else ["ኦሮሚያ", "አማራ", "ደቡብ"] # Oromia, Amhara, SNNPR

    planting_months_am = {
        "መስከረም": 9, "ጥቅምት": 10, "ህዳር": 11, "ታህሳስ": 12, # September to December
        "ጥር": 1, "የካቲት": 2, "መጋቢት": 3, "ሚያዝያ": 4,   # January to April
        "ግንቦት": 5, "ሰኔ": 6, "ሀምሌ": 7, "ነሐሴ": 8      # May to August
    }

    # --- Crop_Year Input Handling ---
    if 'Crop_Year' in model_features:
        current_real_year = datetime.date.today().year
        if df_info is not None and 'Crop_Year' in df_info.columns:
            min_crop_year_data = int(df_info['Crop_Year'].min())
            max_crop_year_data = int(df_info['Crop_Year'].max())
            # Default value should not exceed max_value from data if data's max is less than current year
            default_crop_year_val = min(current_real_year, max_crop_year_data)
            # Max value for input can be data's max or a bit into future if data max is old
            max_input_year = max(max_crop_year_data, current_real_year + 2) # Allow 2 years into future
        else: # No df_info for Crop_Year
            min_crop_year_data = current_real_year - 10
            max_input_year = current_real_year + 2
            default_crop_year_val = current_real_year
            max_crop_year_data = default_crop_year_val # Fallback if no data

        current_scenario_inputs['Crop_Year'] = form_cols[col_idx % cols_per_row].number_input(
            "የሰብል አመት", # Crop Year
            min_value=min_crop_year_data,
            max_value=max_input_year, # This is the corrected max_value for the widget
            value=default_crop_year_val, # This is the corrected default value
            step=1,
            key=f"year_{st.session_state.scenario_counter}"
        )
        col_idx += 1


    if 'Planting_Month_Num' in model_features :
        selected_planting_month_am = form_cols[col_idx % cols_per_row].selectbox("የመዝሪያ ወር", list(planting_months_am.keys()), key=f"month_{st.session_state.scenario_counter}") # Planting Month
        current_scenario_inputs['Planting_Month_Num'] = planting_months_am[selected_planting_month_am]
        col_idx += 1
    elif 'Season' in model_features: # If model uses Season directly
        current_scenario_inputs['Season'] = form_cols[col_idx % cols_per_row].selectbox("ወቅት", unique_seasons, key=f"season_{st.session_state.scenario_counter}") # Season
        col_idx += 1


    if 'Crop' in model_features:
        current_scenario_inputs['Crop'] = form_cols[col_idx % cols_per_row].selectbox("የሰብል አይነት", unique_crops, key=f"crop_{st.session_state.scenario_counter}") # Crop Type
        col_idx += 1
    if 'State' in model_features:
        current_scenario_inputs['State'] = form_cols[col_idx % cols_per_row].selectbox("ክልል/ግዛት", unique_states, key=f"state_{st.session_state.scenario_counter}") # Region/State
        col_idx += 1

    num_inputs_def = {
        'Area': ('የለማ መሬት ስፋት (ሄክታር)', 1000.0, 100.0, "%.2f"), # Cultivated Land Area (hectares)
        'Annual_Rainfall': ('አመታዊ የዝናብ መጠን (mm)', 1200.0, 50.0, "%.2f"), # Annual Rainfall (mm)
        'Fertilizer': ('የማዳበሪያ መጠን (kg)', 50000.0, 1000.0, "%.2f"), # Fertilizer Amount (kg)
        'Pesticide': ('የፀረ-ተባይ መጠን (kg/L)', 500.0, 100.0, "%.2f") # Pesticide Amount (kg/L)
    }

    for feature, (label, default_val, step_val, format_val) in num_inputs_def.items():
        if feature in model_features:
            min_val = float(df_info[feature].min()) if df_info is not None and feature in df_info.columns and not df_info[feature].empty else 0.0
            max_val = float(df_info[feature].max()) if df_info is not None and feature in df_info.columns and not df_info[feature].empty else default_val * 5
            mean_val = float(df_info[feature].mean()) if df_info is not None and feature in df_info.columns and not df_info[feature].empty else default_val
            current_scenario_inputs[feature] = form_cols[col_idx % cols_per_row].number_input(
                label, min_value=min_val, max_value=max_val, value=mean_val, step=step_val, format=format_val, key=f"{feature}_{st.session_state.scenario_counter}"
            )
            col_idx += 1

    # --- Submit Button inside the form ---
    submit_button = st.form_submit_button(label="➕ ይህንን እቅድ ጨምር እና ተንብይ") # Add this plan and predict


# --- Logic after form submission (outside the form block) ---
if submit_button: # This 'submit_button' variable is now defined from within the form
    # The current_scenario_inputs dictionary is populated within the form block
    # So, we need to ensure it's correctly captured or re-accessed if necessary
    # For simplicity, the logic for processing current_scenario_inputs after submit is fine here
    # as long as current_scenario_inputs was fully built before the st.form_submit_button line.

    scenario_df_list = {key: [current_scenario_inputs.get(key)] for key in model_features if key in current_scenario_inputs}
    scenario_df = pd.DataFrame(scenario_df_list)

    for col in model_features:
        if col not in scenario_df.columns:
            # This might happen if a model_feature wasn't included in the form's inputs
            # For example, if 'Planting_Month_Num' is a model_feature but not 'Season', and vice versa
            st.warning(f"'{col}' የተባለው የሞዴል ግብዓት በቅጹ (form) ውስጥ አልተገኘም። በ NaN ይሞላል።") # The model input '{col}' was not found in the form. It will be filled with NaN.
            scenario_df[col] = np.nan


    # Ensure column order matches model_features
    try:
        scenario_df = scenario_df[model_features]
    except KeyError as e:
        st.error(f"የዓምድ ቅደም ተከተል በማስተካከል ላይ ስህተት ተከስቷል: {e}") # An error occurred while adjusting column order: {e}
        st.error(f"ሊሆኑ የሚችሉ ዓምዶች በ scenario_df: {scenario_df.columns.tolist()}") # Possible columns in scenario_df:
        st.error(f"ሞዴሉ የሚጠብቃቸው ዓምዶች: {model_features}") # Columns expected by the model:
        st.stop()


    try:
        with st.spinner("የምርታማነት ትንበያ እየተሰራ ነው..."): # Productivity prediction is in progress...
            predicted_yield = pipeline.predict(scenario_df)
            final_yield = max(0, predicted_yield[0])

        # Add the successfully processed inputs (which are in current_scenario_inputs) to session_state
        # Add the prediction to this dictionary before appending
        current_scenario_inputs_with_yield = current_scenario_inputs.copy() # Start with inputs
        current_scenario_inputs_with_yield['Predicted_Yield'] = round(final_yield, 3)

        st.session_state.scenarios.append(current_scenario_inputs_with_yield)
        st.session_state.scenario_counter += 1
        st.success(f"እቅድ {len(st.session_state.scenarios)} ተጨምሯል! የተገመተው ምርታማነት፦ {final_yield:.3f}") # Plan {len(st.session_state.scenarios)} added! Estimated productivity: {final_yield:.3f}
        st.rerun()
    except Exception as e:
        st.error(f"ትንበያ ላይ ስህተት ተከስቷል፦ {e}") # An error occurred during prediction: {e}
        st.error(f"የተዘጋጀው ዳታ ዓምዶች: {scenario_df.columns.tolist()}") # Prepared data columns:
        st.error(f"ሞዴሉ የሚጠብቃቸው ዓምዶች: {model_features}") # Columns expected by the model:


# --- Display Added Scenarios and Their Predictions ---
if st.session_state.scenarios:
    st.header("2. የእቅድ ማነጻጸሪያ") # Plan Comparison
    # Create DataFrame from the list of dictionaries in session_state
    scenarios_display_df = pd.DataFrame(st.session_state.scenarios)

    # Prepare display columns
    display_cols_order = []
    # Add 'የመዝሪያ ወር' if applicable
    if 'Planting_Month_Num' in scenarios_display_df.columns:
        num_to_month_am = {v: k for k, v in planting_months_am.items()}
        scenarios_display_df['የመዝሪያ ወር'] = scenarios_display_df['Planting_Month_Num'].map(num_to_month_am) # Planting Month
        display_cols_order.append('የመዝሪያ ወር')

    # Add other model features that were part of the input form
    for feature in model_features:
        if feature in scenarios_display_df.columns and feature != 'Planting_Month_Num': # Avoid duplicate or numeric month
            display_cols_order.append(feature)

    # Add 'Predicted_Yield'
    if 'Predicted_Yield' in scenarios_display_df.columns:
        display_cols_order.append('Predicted_Yield')
    else:
        st.warning("የትንበያ ውጤት ('Predicted_Yield') በእቅዶቹ ውስጥ አልተገኘም።") # Prediction result ('Predicted_Yield') not found in the plans.


    # Ensure all columns in display_cols_order actually exist in scenarios_display_df
    final_display_columns = [col for col in display_cols_order if col in scenarios_display_df.columns]

    if final_display_columns and 'Predicted_Yield' in final_display_columns:
        st.dataframe(scenarios_display_df[final_display_columns].sort_values(by='Predicted_Yield', ascending=False).reset_index(drop=True))
    elif final_display_columns: # If no Predicted_Yield but other columns exist
        st.dataframe(scenarios_display_df[final_display_columns].reset_index(drop=True))
    else:
        st.write("ለማሳየት ምንም የተዘጋጁ የእቅድ ዓምዶች የሉም።") # There are no prepared plan columns to display.


    if st.button("🗑️ ሁሉንም እቅዶች አጽዳ"): # Clear All Plans
        st.session_state.scenarios = []
        st.session_state.scenario_counter = 0
        st.rerun()
else:
    st.info("እስካሁን ምንም አይነት የግብርና እቅድ (scenario) አላከሉም። ከላይ ካለው ቅጽ በመሙላት ይጀምሩ።") # You have not added any agricultural plans (scenarios) yet. Start by filling out the form above.

st.sidebar.markdown("---")
if st.sidebar.button("🔄 መተግበሪያውን እንደገና አስጀምር (Reset)"): # Reset Application
    for key in list(st.session_state.keys()):
        if key.startswith("scenario_") or key == "scenarios" or key == "scenario_counter":
            del st.session_state[key]
    st.rerun()