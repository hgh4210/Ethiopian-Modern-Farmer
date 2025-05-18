import streamlit as st
from PIL import Image
from io import BytesIO
import os

# Attempt to import the inference_sdk
try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    st.error(
        "'inference_sdk' አልተጫነም። " # User-facing error
        "እባክዎ በሚከተለው መንገድ ይጫኑት፦ pip install inference-sdk"
    )
    st.stop()

# Roboflow Configuration (remains mostly technical)
ROBOFLOW_API_KEY_ENV = os.environ.get("ROBOFLOW_API_KEY")
DEFAULT_API_KEY = ""
DEFAULT_MODEL_ID = "injera_quality/5"
DEFAULT_API_URL = "https://infer.roboflow.com"

# Streamlit App UI - All user-facing text here is Amharic
st.set_page_config(page_title="የእንጀራ ጥራት ምርመራ", layout="centered") # User-facing
st.title("🔍 የእንጀራ ጥራት ምርመራ") # User-facing
st.markdown( # User-facing
    "የእንጀራ ፎቶ ይስቀሉ ወይም ፎቶ ያንሱ። "
    "ሲስተሙ የሮቦፍሎው ሞዴልን በመጠቀም ጥራቱን ይመረምራል።"
)

# Sidebar for Configuration - Labels are Amharic
st.sidebar.header("የሮቦፍሎው ማዋቀሪያ") # User-facing
api_key_input = st.sidebar.text_input( # User-facing label
    "የሮቦፍሎው ኤፒአይ ቁልፍ",
    value=ROBOFLOW_API_KEY_ENV if ROBOFLOW_API_KEY_ENV else DEFAULT_API_KEY,
    type="password",
    help="የእርስዎ የሮቦፍሎው ኤፒአይ ቁልፍ። ከሮቦፍሎው አካውንትዎ ያግኙት።" # User-facing help text
)
model_id_input = st.sidebar.text_input( # User-facing label
    "የሮቦፍሎው ሞዴል መለያ",
    value=DEFAULT_MODEL_ID,
    help="በሮቦፍሎው ላይ የሰለጠነው ሞዴልዎ መለያ (ለምሳሌ፦ የፕሮጀክት_ስም/ስሪት)።" # User-facing help text
)
api_url_input = st.sidebar.text_input( # User-facing label
    "የሮቦፍሎው ኤፒአይ ዩአርኤል",
    value=DEFAULT_API_URL,
    help="ለሮቦፍሎው ኢንፈረንስ የኤፒአይ መድረሻ።" # User-facing help text
)

# Initialize Roboflow Client - User-facing messages are Amharic
CLIENT = None
if api_key_input and api_key_input != "YOUR_DEFAULT_KEY_IF_NOT_SET" and api_key_input != DEFAULT_API_KEY or ROBOFLOW_API_KEY_ENV :
    try:
        CLIENT = InferenceHTTPClient(
            api_url=api_url_input,
            api_key=api_key_input
        )
        st.sidebar.success("የሮቦፍሎው ደንበኛ ተጀምሯል።") # User-facing
    except Exception as e:
        st.sidebar.error(f"የሮቦፍሎው ደንበኛን ለመጀመር አልተቻለም፦ {e}") # User-facing
        CLIENT = None
else:
    st.sidebar.warning( # User-facing
        "ነባሪ ወይም የጎደለ የኤፒአይ ቁልፍ እየተጠቀሙ ነው። "
        "ኢንፈረንስ እንዲሰራ እባክዎ ትክክለኛ የሮቦፍሎው ኤፒአይ ቁልፍ ያስገቡ።"
    )

# Image Input - User-facing labels/text are Amharic
st.subheader("የእንጀራ ምስል ያቅርቡ") # User-facing
image_source = st.radio( # User-facing prompt & options
    "የምስል ምንጭ ይምረጡ፦",
    ("ምስል ይስቀሉ", "በካሜራ ፎቶ ያንሱ"),
    horizontal=True,
    label_visibility="collapsed"
)

img_file_buffer = None
img_bytes_for_processing = None
source_image_display = None

if image_source == "ምስል ይስቀሉ": # This string matches the Amharic option
    img_file_buffer = st.file_uploader( # User-facing label
        "ምስልዎን ይስቀሉ (JPG, PNG, JPEG)፦",
        type=["jpg", "png", "jpeg"]
    )
    if img_file_buffer is not None:
        img_bytes_for_processing = img_file_buffer.getvalue()
        source_image_display = Image.open(img_file_buffer)

elif image_source == "በካሜራ ፎቶ ያንሱ": # This string matches the Amharic option
    camera_img_buffer = st.camera_input("ፎቶ ለማንሳት ይጫኑ፦") # User-facing label
    if camera_img_buffer is not None:
        img_bytes_for_processing = camera_img_buffer.getvalue()
        source_image_display = Image.open(camera_img_buffer)

# Translation function for Roboflow classes
def translate_class_name_amharic(class_name_en):
    translations = {
        "good": "ጥሩ",
        "bad": "መጥፎ",
        "fair": "ከፊል ጥሩ", # Example
        # Add more English_class: "Amharic_translation" pairs as needed
    }
    return translations.get(class_name_en.lower(), class_name_en) # Return original if no translation

# Display Image and Process - User-facing text is Amharic
if source_image_display:
    col1, col2 = st.columns(2)
    with col1:
        st.image(source_image_display, caption="የእርስዎ የእንጀራ ምስል", use_column_width=True) # User-facing caption

    with col2:
        st.subheader("የምርመራ ውጤቶች") # User-facing
        if CLIENT and img_bytes_for_processing:
            if st.button("🔬 ጥራት ይመርምሩ", use_container_width=True): # User-facing button text
                with st.spinner("እየተመረመረ ነው... እባክዎ ይጠብቁ።"): # User-facing spinner text
                    try:
                        pil_image_to_infer = Image.open(BytesIO(img_bytes_for_processing))
                        result = CLIENT.infer(pil_image_to_infer, model_id=model_id_input)

                        st.success("ምርመራው ተጠናቋል!") # User-facing
                        st.write("---")

                        # Displaying Results - User-facing parts are Amharic
                        if isinstance(result, dict) and 'predictions' in result:
                            predictions = result.get('predictions', [])
                            if predictions:
                                top_prediction = predictions[0]
                                pred_class_en = top_prediction.get('class', "N/A")
                                confidence = top_prediction.get('confidence', 0)
                                pred_class_am = translate_class_name_amharic(pred_class_en)

                                st.metric( # User-facing label
                                    label=f"የተገመተው ጥራት፦ **{pred_class_am}**",
                                    value=f"{confidence*100:.2f}% የመተማመን ደረጃ"
                                )
                                if pred_class_en.lower() == "good":
                                    st.balloons()
                                elif pred_class_en.lower() == "bad":
                                    st.warning("ይህ እንጀራ በሞዴሉ መሰረት ዝቅተኛ ጥራት ያለው ሊሆን ይችላል።") # User-facing
                            else:
                                st.write("በውጤቱ ውስጥ ምንም ግምቶች አልተገኙም።") # User-facing

                        elif isinstance(result, list) and result and 'class' in result[0]:
                            st.write(f"**የተገኙ ነገሮች/አካባቢዎች ({len(result)})፦**") # User-facing
                            for i, pred in enumerate(result):
                                pred_class_en = pred.get('class', "N/A")
                                confidence = pred.get('confidence', 0)
                                pred_class_am = translate_class_name_amharic(pred_class_en)
                                st.write( # User-facing
                                    f"- **{pred_class_am}** (የመተማመን ደረጃ፦ {confidence*100:.2f}%)"
                                )
                        else:
                            st.info( # User-facing
                                "ግምቶችን በራስ-ሰር መተንተን አልተቻለም። "
                                "አወቃቀሩን ለመረዳት እና የማሳያውን አመክንዮ ለማዘመን ከዚህ በታች ያለውን 'ጥሬ የሮቦፍሎው ውጤት' ይመልከቱ።"
                            )
                        st.write("---")
                        st.write("**ጥሬ የሮቦፍሎው ውጤት (እንግሊዝኛ):**") # User-facing label, notes output is English
                        st.json(result) # Raw JSON output remains as is

                    except Exception as e:
                        st.error(f"በኢንፈረንስ ወቅት ስህተት ተከስቷል፦ {e}") # User-facing
                        st.error( # User-facing
                            "እባክዎ የኤፒአይ ቁልፍዎን፣ የሞዴል መለያዎን፣ የኔትወርክ ግንኙነትዎን ያረጋግጡ፣ "
                            "እና የሮቦፍሎው ሞዴሉ በትክክል መሰማራቱን ያረጋግጡ።"
                        )
        elif not CLIENT:
            st.warning("የሮቦፍሎው ደንበኛ አልተጀመረም። እባክዎ በጎን አሞሌው ላይ ያለውን የኤፒአይ ቁልፍ ያረጋግጡ።") # User-facing
        else:
            st.info("ምስል ይስቀሉ ወይም ፎቶ ያንሱ እና 'ጥራት ይመርምሩ' የሚለውን ይጫኑ።") # User-facing
else:
    st.info("እባክዎ ከላይ ባሉት አማራጮች ምስል ይስቀሉ ወይም ፎቶ ያንሱ።") # User-facing

