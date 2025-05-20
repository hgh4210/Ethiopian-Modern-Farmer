import streamlit as st
import google.generativeai as genai
import os

# --- Page Setup and Title ---
st.set_page_config(page_title="የግብርና አማካሪ ቻትቦት", layout="centered", initial_sidebar_state="collapsed")
st.title("🤖 የግብርና እና ምግብ አማካሪ (AI Chatbot)")
st.caption("በአማርኛ ስለ ግብርና እና ምግብ ጉዳዮች ይጠይቁ") # Ask about agriculture and food issues in Amharic

# --- Gemini API Key Configuration ---
# 1. From Streamlit Secrets (for deployment)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

# 2. From Environment Variable (for local development)
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 3. User input (for demo/quick testing - not recommended for production)
if not GEMINI_API_KEY:
    st.sidebar.subheader("የ Gemini ኤፒአይ ቁልፍ") # Gemini API Key
    GEMINI_API_KEY_INPUT = st.sidebar.text_input("ኤፒአይ ቁልፍዎን እዚህ ያስገቡ፦", type="password", key="api_key_input_chat") # Enter your API key here:
    if GEMINI_API_KEY_INPUT:
        GEMINI_API_KEY = GEMINI_API_KEY_INPUT
    else:
        st.warning("⚠️ የ Gemini ኤፒአይ ቁልፍ አልተገኘም። እባክዎ በጎን አሞሌው ላይ ያስገቡ ወይም በ Streamlit Secrets/Environment Variables ያዘጋጁ።") # Gemini API key not found. Please enter it in the sidebar or set it up in Streamlit Secrets/Environment Variables.
        st.stop() # Stop the app if no key

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"የ Gemini ኤፒአይ ቁልፍን በማዋቀር ላይ ስህተት ተከስቷል፦ {e}") # Error configuring Gemini API key: {e}
    st.stop()

# --- Selecting and Configuring Gemini Model ---
# Use models better suited for conversation (e.g., 'gemini-1.5-flash' or 'gemini-pro')
MODEL_NAME = "gemini-1.5-flash" # or 'gemini-pro'

# System Prompt (in Amharic)
SYSTEM_PROMPT_AMHARIC = """ሰላም! እኔ ስለ ኢትዮጵያ ግብርና፣ አዝመራ አመራረት፣ የእንስሳት እርባታ ዘዴዎች、 የአፈርና ውሃ አያያዝ、 የሰብል ተባይና በሽታ ቁጥጥር、 ዘመናዊ የግብርና ቴክኖሎጂዎች、 የምግብ አይነቶች、 የምግብ ዝግጅት、 የምግብ ደህንነት、 እና ስነ-ምግብ ጉዳዮች መረጃ ለመስጠትና ለመወያየት የተዘጋጀሁ የሰው ሰራሽ የማሰብ ችሎታ ረዳት ነኝ። እባክዎን ጥያቄዎን በእነዚህ ርዕሶች ዙሪያ ብቻ ያቅርቡ። ከእነዚህ ርዕሶች ውጪ ለሚቀርቡ ጥያቄዎች መልስ ለመስጠትም ሆነ ለመወያየት አልተፈቀደልኝም። በግብርና ወይም በምግብ ነክ ጉዳይ ላይ ምን ልርዳዎት?"""
# Hello! I am an AI assistant designed to provide information and discuss Ethiopian agriculture, crop production, animal husbandry methods, soil and water management, crop pest and disease control, modern agricultural technologies, food types, food preparation, food safety, and nutrition issues. Please ask your questions only around these topics. I am not allowed to answer or discuss questions outside of these topics. What can I help you with regarding agriculture or food-related matters?

# Initialize the conversational model with the system prompt
try:
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_PROMPT_AMHARIC,
        # (Optional) Safety settings - may not be very necessary for agricultural content
        # safety_settings={
        #     'HATE': 'BLOCK_NONE',
        #     'HARASSMENT': 'BLOCK_NONE',
        #     'DANGEROUS' : 'BLOCK_NONE'
        # }
    )
    chat_session = model.start_chat(history=[]) # Start a chat session
except Exception as e:
    st.error(f"የ Gemini ሞዴልን በማስጀመር ላይ ስህተት ተከስቷል፦ {e}") # Error initializing Gemini model: {e}
    st.stop()


# --- Managing Chat History in Session State ---
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    # (Optional) Initial welcome message from the chatbot
    # st.session_state.chat_messages.append({"role": "model", "parts": [{"text": SYSTEM_PROMPT_AMHARIC.split("!")[0] + "! ምን ልርዳዎት?"}]}) # What can I help you with?


# --- Displaying Chat Messages ---
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0]["text"])


# --- Receiving User Input ---
user_prompt = st.chat_input("ጥያቄዎን እዚህ በአማርኛ ይጻፉ...") # Write your question here in Amharic...

if user_prompt:
    # Add user's message to history and display it
    st.session_state.chat_messages.append({"role": "user", "parts": [{"text": user_prompt}]})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Get and display response from Gemini
    with st.chat_message("model"):
        message_placeholder = st.empty()
        full_response_text = ""
        try:
            # Gemini API call - using chat history
            # For a direct chat session, use `send_message`
            response = chat_session.send_message(user_prompt, stream=True)

            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response_text += chunk.text
                    message_placeholder.markdown(full_response_text + "▌") # To show "typing"
                elif hasattr(chunk, 'parts'): # Sometimes it might have 'parts'
                     for part in chunk.parts:
                         if hasattr(part, 'text') and part.text:
                            full_response_text += part.text
                            message_placeholder.markdown(full_response_text + "▌")
            message_placeholder.markdown(full_response_text) # Display the final full response
        except Exception as e:
            full_response_text = f"ውይይቱን በማስኬድ ላይ ሳለ ስህተት ተከስቷል፦ {e}" # An error occurred while processing the chat: {e}
            message_placeholder.error(full_response_text)

    # Add model's response to history
    st.session_state.chat_messages.append({"role": "model", "parts": [{"text": full_response_text}]})

# (Optional) Button to clear chat history
if st.sidebar.button("ውይይቱን አጽዳ"): # Clear Chat
    st.session_state.chat_messages = []
    # chat_session = model.start_chat(history=[]) # A new chat session can be started
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"የሚጠቀመው ሞዴል: {MODEL_NAME}") 
