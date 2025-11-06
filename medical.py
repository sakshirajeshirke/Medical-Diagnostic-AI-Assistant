import streamlit as st
from PIL import Image
import re
import os
from dotenv import load_dotenv
import json
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from groq import Groq
import numpy as np
import cv2

# =============================================================================
# ⚙️ HANDLE OPTIONAL DEPENDENCIES
# =============================================================================

# --- PDF SUPPORT ---
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- EasyOCR (CPU-only) ---
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# --- LANGCHAIN (FOR OPENROUTER CHATBOT) --- (NEW)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    LANGCHAIN_OPENAI_AVAILABLE = False
    # We'll handle this error in the UI

# =============================================================================
# ⚙️ CONFIGURATION & CONSTANTS
# =============================================================================

st.set_page_config(
    layout="wide",
    page_title="Medical Diagnostic AI Assistant",
    page_icon="🩺"
)

# Medical-specific constants
MEDICAL_ENTITIES = [
    "Disease/Condition", "Symptoms", "Medications", "Dosage", "Lab Results",
    "Vital Signs", "Procedures", "Allergies", "Medical History"
]

# Enhanced LLM tasks for medical diagnosis
LLM_TASKS = {
    "Comprehensive Diagnosis Summary": "Provide a detailed medical summary",
    "Extract Clinical Entities": "Extract all clinical information",
    "Risk Assessment": "Assess patient risks and red flags",
    "Treatment Recommendations": "Suggest treatment options",
    "Patient-Friendly Explanation": "Simplify for patient understanding",
    "Follow-up Actions": "Identify required follow-up actions",
    "Lab Results Interpretation": "Interpret laboratory test results",
    "Medication Analysis": "Analyze prescribed medications"
}

# Load environment variables
load_dotenv()

# Get API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # <-- NEW

# --- NEW: OpenRouter Configuration ---
# Using a free, high-quality model from OpenRouter
OPENROUTER_MODEL_NAME = "openai/gpt-4o-mini"

# =============================================================================
# 🔧 INITIALIZATION & SETUP (GROQ API)
# =============================================================================

def initialize_llm() -> Tuple[Optional[Groq], bool]:
    """Initialize the Groq API client."""
    try:
        if not GROQ_API_KEY:
            st.error("❌ GROQ_API_KEY not found in .env file")
            st.info("💡 Please add GROQ_API_KEY=... to your .env file from groq.com")
            return None, False

        client = Groq(api_key=GROQ_API_KEY)
        client.models.list()
        st.toast("✅ Connected to Groq API", icon="✅")
        return client, True
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq API: {e}")
        return None, False

# Initialize LLM (Groq for Analysis)
llm_client, llm_initialized = initialize_llm()

# =============================================================================
# 🤖 MODEL LOADING - OPTIMIZED FOR CPU
# =============================================================================

@st.cache_resource
def load_easyocr_model():
    """Loads EasyOCR model optimized for CPU."""
    if not EASYOCR_AVAILABLE:
        return None
    try:
        # gpu=False ensures CPU-only operation
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return reader
    except Exception as e:
        st.error(f"Failed to load EasyOCR: {e}")
        return None

# Lazy loading - only load when needed
easyocr_reader = None

def get_easyocr():
    global easyocr_reader
    if easyocr_reader is None:
        with st.spinner("📦 Loading EasyOCR - First time setup..."):
            easyocr_reader = load_easyocr_model()
    return easyocr_reader

# =============================================================================
# 🤖 MODEL LOADING - CHATBOT (OPENROUTER)
# =============================================================================

@st.cache_resource
def load_openrouter_model():
    """
    Loads the OpenRouter client.
    This function is cached and should NOT contain any st.* elements.
    It will raise an exception if it fails.
    """
    if not LANGCHAIN_OPENAI_AVAILABLE:
        raise ImportError("`langchain_openai` not installed. `pip install langchain_openai`")

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")

    # Configure ChatOpenAI to use OpenRouter's API
    llm = ChatOpenAI(
        model=OPENROUTER_MODEL_NAME,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
    )
    
    # Test connection
    llm.invoke("Test connection")
    return llm

# Lazy loading
chatbot_llm_global = None

def get_openrouter_llm():
    """Singleton getter for the OpenRouter model with UI elements."""
    global chatbot_llm_global
    
    if chatbot_llm_global is None:
        try:
            with st.spinner(f"📦 Connecting to OpenRouter ({OPENROUTER_MODEL_NAME})..."):
                chatbot_llm_global = load_openrouter_model()
            
            st.toast("✅ Connected to OpenRouter", icon="🤖")
        
        except ImportError as ie:
            st.error(f"❌ Chatbot disabled: {ie}")
            chatbot_llm_global = None 
        
        except ValueError as ve: # Catch the specific key error
            st.error(f"❌ OpenRouter Error: {ve}")
            st.info("💡 Please add OPENROUTER_API_KEY=... to your .env file from openrouter.ai")
            chatbot_llm_global = None

        except Exception as e:
            # Handle connection errors
            st.error(f"❌ Failed to connect to OpenRouter: {e}")
            st.info(f"💡 Check your API key and ensure the model '{OPENROUTER_MODEL_NAME}' is correct.")
            chatbot_llm_global = None 
    
    return chatbot_llm_global


# =============================================================================
# 📄 FILE PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def convert_input_to_image(uploaded_file) -> Tuple[List[Image.Image], str]:
    """Converts uploaded file (PDF or image) into PIL Images."""
    try:
        if uploaded_file.type == "application/pdf":
            if not PDF_SUPPORT:
                st.error("❌ PDF support unavailable. Install: pip install pdf2image poppler-utils")
                return [], "error"

            images_raw = convert_from_bytes(uploaded_file.read(), dpi=300)
            images = [img.convert('RGB') for img in images_raw]
            return images, "pdf"
        else:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return [image], "image"
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return [], "error"

# =============================================================================
# 🔍 IMAGE PREPROCESSING
# =============================================================================

def preprocess_for_easyocr(image: Image.Image) -> np.ndarray:
    """
    Advanced preprocessing for EasyOCR.
    This pipeline is designed for scanned text, potentially on lined paper.
    It uses morphological operations to find and remove horizontal lines.
    """
    try:
        # 1. Convert PIL to numpy array (BGR)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # 3. Binarize the image (White text/lines on Black background)
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Detect and Remove Horizontal Lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        no_lines = cv2.subtract(binary, detected_lines)
        
        # 5. Repair/Thicken Text
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        repaired_text = cv2.morphologyEx(no_lines, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # 6. Invert back to Black Text on White Background
        final_image = cv2.bitwise_not(repaired_text)
        
        # 7. Final Denoise
        final_image_blurred = cv2.medianBlur(final_image, 3)

        return final_image_blurred
    
    except Exception as e:
        st.warning(f"EasyOCR preprocessing failed, using original grayscale: {e}")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

# =============================================================================
# 🔍 OCR ENGINES - OPTIMIZED IMPLEMENTATIONS
# =============================================================================

def run_easy_ocr(image: Image.Image) -> str:
    """Runs EasyOCR with enhanced preprocessing and error handling."""
    progress_text = "🔍 Running EasyOCR on CPU..."
    progress_bar = st.progress(0, text=progress_text)
    
    reader = get_easyocr()
    
    if reader is None:
        progress_bar.empty()
        return "❌ EasyOCR not available or failed to load"
    
    try:
        progress_bar.progress(25, text=f"{progress_text} (Preprocessing...)")
        # Preprocess image
        processed_img = preprocess_for_easyocr(image)
        
        progress_bar.progress(50, text=f"{progress_text} (Running model...)")
        # Run OCR with optimized parameters
        result = reader.readtext(
            processed_img,
            detail=1,
            paragraph=False, # False is better for structured/sparse text
            low_text=0.3
        )
        
        # Extract text
        lines = []
        for detection in result:
            if len(detection) >= 2:
                text_content = detection[1]
                lines.append(text_content)
        
        progress_bar.progress(100, text="✅ OCR Complete!")
        
        if lines:
            return "\n".join(lines)
        else:
            return "⚠️ No text detected by EasyOCR"
    
    except Exception as e:
        st.error(f"❌ EasyOCR Error: {e}")
        return f"Error: {str(e)}"
    finally:
        progress_bar.empty()

# =============================================================================
# 🧹 TEXT PROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def clean_and_normalize(text: str) -> str:
    """Advanced text cleaning for medical documents."""
    try:
        # Remove excessive whitespace
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix numbers with spaces
        text = re.sub(r'(\d)\s+(\.\d+)', r'\1\2', text)
        # Remove common OCR artifacts
        text = re.sub(r'[|]', 'I', text)
        return text
    except Exception as e:
        st.error(f"❌ Text cleaning error: {e}")
        return text

def extract_medical_keywords(text: str) -> List[str]:
    """Extract potential medical keywords from text."""
    medical_patterns = [
        r'\b\d+\s*(?:mg|ml|mcg|g|kg|units?)\b',
        r'\b\d+/\d+\b',
        r'\b\d+\.?\d*\s*(?:bpm|mmHg|°[CF])\b',
        r'\b[A-Z][a-z]+(?:itis|osis|emia|pathy|algia)\b'
    ]
    
    keywords = []
    for pattern in medical_patterns:
        keywords.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(keywords))

# =============================================================================
# 🤖 LLM PROCESSING FUNCTIONS (GROQ API FOR ANALYSIS)
# =============================================================================

def get_system_prompt() -> str:
    """Returns the system prompt for medical analysis."""
    return """You are an expert medical AI assistant specializing in analyzing medical documents, reports, and clinical notes. Your responsibilities include:

1. Accurately interpreting medical terminology and lab results
2. Identifying critical findings and red flags
3. Providing evidence-based clinical insights
4. Maintaining patient safety as the top priority
5. Using clear, precise medical language

IMPORTANT DISCLAIMERS:
- Always recommend consulting with qualified healthcare professionals
- Never provide definitive diagnoses - only analytical insights
- Highlight any concerning findings that require immediate attention
- Acknowledge limitations and uncertainties in the data"""

def get_task_prompt(task: str, text: str) -> str:
    """Generate task-specific prompts for medical analysis."""
    
    prompts = {
        "Comprehensive Diagnosis Summary": f"""Analyze this medical document and provide a comprehensive summary. Structure your response as follows:

1.  **Patient Profile**: Brief overview (e.g., Name, Age, Gender, primary complaint, key history).
2.  **Primary Findings**: Main diagnoses or conditions identified.
3.  **Supporting Evidence**: Key symptoms, lab results, or observations from the text.
4.  **Potential Differential Diagnoses**: (If applicable) List other possibilities the text might suggest.
5.  **Clinical Significance**: What these findings mean clinically.

Medical Text:
{text}

Provide a structured, detailed summary:""",

        "Extract Clinical Entities": f"""From the medical text, meticulously extract and categorize ALL clinical information into the following JSON structure.

**Instructions:**
1.  Populate every field.
2.  If information for a field is not present in the text, use `null`.
3.  Do not invent or infer information that is not explicitly stated.
4.  Be as detailed as possible, especially for `patient_info`, `medications`, and `lab_results`.

**JSON Structure:**
{{
  "patient_info": {{
    "name": "Full Name",
    "age": "Age",
    "gender": "Gender",
    "dob": "Date of Birth",
    "patient_id": "Patient ID or MRN"
  }},
  "diagnoses": [
    {{"condition": "Condition Name", "status": "e.g., Active, Resolved, Suspected"}}
  ],
  "symptoms": [
    {{"symptom": "Symptom description", "duration": "e.g., 3 days"}}
  ],
  "medications": [
    {{"name": "Drug Name", "dosage": "e.g., 10mg", "frequency": "e.g., twice daily"}}
  ],
  "lab_results": [
    {{"test": "Test Name", "value": "Result Value", "unit": "Unit", "reference_range": "e.g., 0.5-1.5", "status": "e.g., High, Low, Normal"}}
  ],
  "vital_signs": {{
    "bp": "Systolic/Diastolic",
    "heart_rate": "bpm",
    "temperature": "C/F",
    "respiratory_rate": "breaths/min",
    "oxygen_saturation": "%"
  }},
  "procedures": [
    {{"name": "Procedure Name", "date": "Date of procedure"}}
  ],
  "allergies": [
    {{"allergen": "Allergen", "reaction": "Description of reaction"}}
  ],
  "medical_history": [
    {{"condition": "Past condition", "notes": "Relevant notes"}}
  ],
  "assessment_plan": "Physician's assessment and plan, as a summary."
}}

Medical Text:
{text}

JSON Output:""",

        "Risk Assessment": f"""Perform a comprehensive risk assessment of this medical document:

1. **Critical Findings**: Any life-threatening or urgent issues (RED FLAGS)
2. **High-Risk Indicators**: Factors requiring prompt medical attention
3. **Moderate Concerns**: Issues needing monitoring or follow-up
4. **Risk Factors**: Underlying conditions or lifestyle factors
5. **Recommended Actions**: Immediate steps and timeline

Medical Text:
{text}

Risk Assessment:""",

        "Treatment Recommendations": f"""Based on this medical information, provide evidence-based treatment recommendations:

1. **Pharmacological**: Medication options with rationale
2. **Non-Pharmacological**: Lifestyle modifications, therapy, etc.
3. **Monitoring**: What to track and how often
4. **Contraindications**: What to avoid
5. **Expected Outcomes**: Timeline and success indicators

Medical Text:
{text}

Treatment Recommendations:""",

        "Patient-Friendly Explanation": f"""Translate this medical information into clear, patient-friendly language. **Use a calm, empathetic, and clear tone.**

1. Explain what the findings mean in everyday terms
2. Describe why these findings matter
3. Clarify any confusing medical jargon
4. Provide context for any abnormal results (e.g., "This number is slightly high, which might mean...")
5. Suggest questions the patient should ask their doctor

Avoid technical jargon. Use analogies where helpful.

Medical Text:
{text}

Patient Explanation:""",

        "Follow-up Actions": f"""Identify all necessary follow-up actions from this medical document:

1. **Immediate Actions** (within 24-48 hours)
2. **Short-term Follow-ups** (within 1-2 weeks)
3. **Long-term Monitoring** (ongoing)
4. **Specialist Referrals** needed
5. **Tests/Procedures** to schedule
6. **Lifestyle Modifications** to implement

Medical Text:
{text}

Follow-up Plan:""",

        "Lab Results Interpretation": f"""Interpret these laboratory results comprehensively:

1. **Normal Values**: Which results are within reference ranges
2. **Abnormal Values**: Which are out of range and by how much
3. **Clinical Significance**: What abnormalities indicate
4. **Patterns**: Any concerning patterns or trends
5. **Recommendations**: Further testing or monitoring needed

Medical Text:
{text}

Lab Interpretation:""",

        "Medication Analysis": f"""Analyze the medications mentioned in this document:

1. **Medication List**: All drugs with dosages
2. **Therapeutic Purpose**: Why each is prescribed
3. **Potential Interactions**: Drug-drug or drug-condition concerns
4. **Side Effects**: Common and serious adverse effects to monitor
5. **Compliance Considerations**: Timing, food interactions, etc.

Medical Text:
{text}

Medication Analysis:"""
    }
    
    return prompts.get(task, f"Analyze this medical text:\n\n{text}")

def run_llm_task(cleaned_text: str, task: str) -> Dict[str, any]:
    """Runs LLM analysis using Groq API."""
    if not llm_initialized or llm_client is None:
        return {"error": "Groq API not initialized", "result": None}
    
    try:
        with st.spinner(f"🤖 Running {task} with Groq..."):
            system_prompt = get_system_prompt()
            user_prompt = get_task_prompt(task, cleaned_text)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            chat_completion = llm_client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-120b", # Using a known high-performance Groq model
                temperature=0.2,
                max_tokens=4096,
            )
            
            result = chat_completion.choices[0].message.content
            
            # Extract JSON if applicable
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match and task == "Extract Clinical Entities":
                try:
                    json_str = json_match.group().strip().replace("```json", "").replace("```", "")
                    json_data = json.loads(json_str)
                    return {"result": result, "json": json_data, "error": None}
                except json.JSONDecodeError:
                    pass
            
            return {"result": result, "json": None, "error": None}
            
    except Exception as e:
        st.error(f"❌ Groq API Error: {e}")
        return {"error": str(e), "result": None}

# =============================================================================
# 🎨 MAIN STREAMLIT UI
# =============================================================================

def main():
    # Header
    st.title("🩺 Medical Diagnostic AI Assistant")
    st.markdown("*Advanced AI-powered analysis of medical documents and reports*")
    
    # === SIDEBAR ===
    st.sidebar.header("📁 Document Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Medical Document",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload a medical report, prescription, lab result, or clinical note"
    )
    
    st.sidebar.header("🔬 Analysis Configuration")
    
    if not EASYOCR_AVAILABLE:
        st.sidebar.error("❌ EasyOCR engine not available!")
        st.sidebar.info("Please install it: pip install easyocr")
    
    # Performance tip
    st.sidebar.info("💡 **Performance Tip**: OCR may take a few moments to process.")
    
    # === NEW: Initialize OpenRouter LLM for Chatbot ===
    if LANGCHAIN_OPENAI_AVAILABLE:
        chatbot_llm = get_openrouter_llm() # <-- UPDATED
    else:
        chatbot_llm = None
        st.sidebar.error("❌ Chatbot disabled: `langchain_openai` not installed.")

    # Initialize session state
    if 'raw_text' not in st.session_state:
        st.session_state.raw_text = ""
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    
    image_list = []
    page_to_process = 0
    
    if uploaded_file:
        image_list, file_type = convert_input_to_image(uploaded_file)
        
        if file_type == "pdf" and len(image_list) > 0:
            st.sidebar.success(f"✅ PDF loaded: {len(image_list)} page(s)")
            page_to_process = st.sidebar.number_input(
                "Select Page to Process:",
                min_value=1, max_value=len(image_list), value=1
            ) - 1
        elif file_type == "image" and len(image_list) > 0:
            st.sidebar.success("✅ Image loaded successfully")
        elif file_type == "error":
            st.sidebar.error("❌ Failed to process file")
    
    # Process Button
    run_ocr_button = st.sidebar.button(
        "🚀 1. Run OCR",
        disabled=not uploaded_file or len(image_list) == 0 or not EASYOCR_AVAILABLE,
        type="primary",
        use_container_width=True
    )
    
    # === MAIN CONTENT AREA ===
    
    if run_ocr_button:
        st.session_state.processed_results = None
        st.session_state.raw_text = ""
        st.session_state.chat_messages = []
            
        img_to_process = image_list[page_to_process]
        
        # Stage 1: Document Preview
        st.header("📄 Stage 1: Document Preview")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(img_to_process, caption=f"Page {page_to_process + 1}", use_container_width=True)
        with col2:
            st.metric("Image Size", f"{img_to_process.size[0]}x{img_to_process.size[1]}px")
            st.metric("Mode", img_to_process.mode)
            st.metric("OCR Engine", "EasyOCR")

        # Stage 2: OCR Extraction
        st.header("🔍 Stage 2: Text Extraction (OCR)")
        
        st.info(f"⏳ Running **EasyOCR** on Page {page_to_process+1} - This may take a few moments...")
        
        raw_text = run_easy_ocr(img_to_process)
        st.session_state.raw_text = raw_text
        
        st.session_state.editable_text = raw_text 
        
        char_count = len(raw_text)
        st.success(f"✅ Extraction complete! ({char_count} characters extracted)")
    
    # Stage 3 - Editable text area
    if st.session_state.raw_text:
        st.header("🧹 Stage 3: Review & Clean Text")
        st.info("⚠️ **Critical Step:** Please review the extracted text and manually correct any OCR errors before running the AI analysis.")
        
        with st.expander("📋 View Original OCR Output"):
            st.text_area("Raw OCR Output", st.session_state.raw_text, height=200, disabled=True, key="raw_display")
        
        edited_text = st.text_area(
            "Editable Extracted Text",
            st.session_state.raw_text, 
            height=300,
            key="editable_text",
            help="Edit the text to correct any OCR mistakes before analysis"
        )
        
        keywords = extract_medical_keywords(edited_text)
        if keywords:
            with st.expander(f"🔍 Detected Medical Keywords ({len(keywords)})"):
                st.write(", ".join(keywords))
        
        cleaned_text = clean_and_normalize(edited_text)
        
        st.markdown("---")
        
        # Stage 4: AI Analysis (New Tabbed UI)
        st.header("🤖 Stage 4: AI Medical Analysis (Groq)")
        st.info("Select a tab below and click the button to run a specific analysis.")

        if not llm_initialized:
            st.error("❌ AI Analysis disabled. Groq API is not initialized.")
        elif not cleaned_text:
            st.warning("⚠️ Please provide text in Stage 3 to run analysis.")
        else:
            tab_names = list(LLM_TASKS.keys())
            tabs = st.tabs(tab_names)
            
            for i, task in enumerate(tab_names):
                with tabs[i]:
                    st.subheader(f"Analysis: {task}")
                    st.markdown(f"**Description:** {LLM_TASKS[task]}")
                    
                    button_key = f"run_{task.lower().replace(' ', '_')}"
                    
                    if st.button(f"🤖 Run {task}", key=button_key, use_container_width=True, type="primary"):
                        result = run_llm_task(cleaned_text, task)
                        
                        if result["error"]:
                            st.error(f"Error: {result['error']}")
                        elif result["result"]:
                            if result["json"]:
                                st.subheader("Structured Data")
                                st.json(result["json"])
                                st.subheader("Full Analysis")
                            st.markdown(result["result"])
                            
                            if 'processed_results' not in st.session_state or st.session_state.processed_results is None:
                                st.session_state.processed_results = {
                                    "timestamp": datetime.now().isoformat(),
                                    "ocr_engine": "EasyOCR",
                                    "raw_text": st.session_state.raw_text,
                                    "edited_text": edited_text,
                                    "cleaned_text": cleaned_text,
                                    "results": {},
                                }
                            st.session_state.processed_results["results"][task] = result
                            st.success(f"'{task}' analysis complete.")
        
        # =============================================================================
        # 💬 STAGE 5: AI CHATBOT (OPENROUTER)
        # =============================================================================
        st.markdown("---")
        st.header(f"💬 Stage 5: Medical Chatbot ({OPENROUTER_MODEL_NAME})")
        
        if not chatbot_llm: # <-- UPDATED
            st.error("❌ OpenRouter is not connected or `langchain_openai` is not installed. Chatbot is disabled.")
        elif not cleaned_text: 
            st.warning("⚠️ Text from Stage 3 is empty. Chatbot context will be limited.")
        else:
            st.info("Ask follow-up questions about the report or general medical questions.")

            # 1. Display chat history
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # 2. Get user input
            if prompt := st.chat_input("Ask a question about the report..."):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # 3. Generate response
                with st.chat_message("assistant"):
                    with st.spinner("OpenRouter is thinking..."):
                        
                        # --- Build the prompt for the LLM ---
                        
                        # System prompt with context (model-agnostic)
                        system_prompt_content = f"""You are an expert medical chatbot.
Your purpose is to answer questions based on a provided medical report and your general medical knowledge.

**Strict Rules:**
1.  **Prioritize Report Context:** If the user's question seems related to the report, use the "MEDICAL REPORT CONTEXT" provided below to answer.
2.  **General Knowledge:** If the question is general (e.g., "What is hypertension?"), answer from your trained knowledge.
3.  **Disclaimers:** ALWAYS include a disclaimer: "I am an AI assistant, not a medical professional. Consult a doctor for medical advice."
4.  **Do Not Hallucinate:** If the answer is not in the report or your knowledge, state that you don't know.

**--- MEDICAL REPORT CONTEXT ---**
{cleaned_text}
**--- END OF CONTEXT ---**
"""
                        
                        # Create the message history
                        messages_for_llm = [SystemMessage(content=system_prompt_content)]
                        
                        for msg in st.session_state.chat_messages[:-1]: 
                            if msg["role"] == "user":
                                messages_for_llm.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant":
                                messages_for_llm.append(AIMessage(content=msg["content"]))
                        
                        messages_for_llm.append(HumanMessage(content=prompt))

                        # Generate the response
                        try:
                            response = chatbot_llm.invoke(messages_for_llm) # <-- UPDATED
                            response_content = response.content
                        except Exception as e:
                            response_content = f"Sorry, an error occurred with the chatbot: {e}"
                        
                        st.markdown(response_content)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response_content})

    # Show export button if any results exist
    if st.session_state.processed_results:
        st.markdown("---")
        st.header("💾 Export Results")
        st.info("Download a JSON file containing all the analyses you have run (chatbot history is not included).")
        json_data = json.dumps(st.session_state.processed_results, indent=2)
        st.download_button(
            "📊 Download JSON Report",
            data=json_data,
            file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

if __name__ == "__main__":
    main()