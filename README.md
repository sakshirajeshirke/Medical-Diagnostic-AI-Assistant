# 🩺 Medical Diagnostic AI Assistant

This is an advanced, multi-stage Streamlit application designed to assist medical professionals or users in analyzing medical documents (reports, lab results, clinical notes) using **OCR (Optical Character Recognition)** for text extraction and powerful **Large Language Models (LLMs)** for clinical analysis and interactive Q\&A.

It leverages Groq for high-speed analysis and OpenRouter (using `gpt-4o-mini`) for an interactive, context-aware chatbot experience.

## ✨ Features

  * **Multi-Format Document Support:** Process **PDFs** and common image files (`.png`, `.jpg`, etc.).
  * **Optimized OCR:** Uses **EasyOCR** with advanced **image preprocessing** (including horizontal line removal) for better accuracy on scanned medical documents.
  * **Editable Text Review:** Allows users to manually correct OCR errors before passing the text to the AI.
  * **Groq-Powered Clinical Analysis:** Utilizes the Groq API for rapid execution of comprehensive medical analysis tasks (e.g., **Risk Assessment**, **Treatment Recommendations**, **Structured Entity Extraction**).
  * **Context-Aware Chatbot:** An integrated chatbot powered by **OpenRouter** and LangChain allows users to ask follow-up questions directly about the extracted report context.
  * **JSON Export:** Export a structured JSON file containing the extracted text and all AI analysis results.

## ⚙️ Prerequisites

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **API Keys:** You will need two API keys:
      * **Groq API Key:** For high-speed analysis. Obtainable from [groq.com](https://groq.com).
      * **OpenRouter API Key:** For the interactive chatbot. Obtainable from [openrouter.ai](https://openrouter.ai).

## 🚀 Installation & Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repo-link>
    cd <repo-directory>
    ```

2.  **Install Dependencies:**
    This project requires several libraries, including `easyocr`, `pdf2image`, `langchain-openai`, and `groq`.

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: You may need to install **Poppler** for PDF support. See `pdf2image` documentation.)*

3.  **Set Environment Variables:**
    Create a file named `.env` in the root directory and add your API keys:

    ```ini
    # .env file
    GROQ_API_KEY="your-groq-api-key"
    OPENROUTER_API_KEY="your-openrouter-api-key"
    ```

4.  **Run the Application:**

    ```bash
    streamlit run app.py
    ```

## 🖼️ Working Demo

### 1\. Document Upload & OCR

A visual of the sidebar upload and the text extraction process.
<img width="1365" height="653" alt="med1" src="https://github.com/user-attachments/assets/0e902bb7-aaa1-48cd-8cc1-c9bfbd19a5bd" />
<img width="1358" height="683" alt="med3" src="https://github.com/user-attachments/assets/951967bc-1470-4378-bdc4-a5d6463628bf" />

-----

### 2\. AI Analysis in Tabs

Showcasing the different clinical tasks executed via Groq.
<img width="1340" height="460" alt="med4" src="https://github.com/user-attachments/assets/05568b87-7252-4d07-bd27-7976c00f5f2d" />
<img width="982" height="476" alt="med5" src="https://github.com/user-attachments/assets/1543b820-3824-4d0a-af8c-c555cfe441a4" />


-----

### 3\. Interactive Chatbot

Demonstrating a user asking a question about the report and the OpenRouter model responding with context.
<img width="1027" height="460" alt="med6" src="https://github.com/user-attachments/assets/b748c06e-7e78-47fe-b48c-1d8b35daa1c3" />

-----

### 🎥 Video Demonstration

You can see a full walkthrough of the application's functionality here:
<div align="center">
  <video src="https://raw.githubusercontent.com/saksirajeshirke/Medical-Diagnostic-AI-Assistant/main/med.mp4" 
         controls 
         loop 
         muted
         playsinline
         width="800">
    Your browser does not support the video tag.
  </video>
</div>

**[Video of the Medical Diagnostic AI Assistant in action]**

## ⚠️ Important Disclaimer

This tool is an **AI assistant** and is **not a substitute for professional medical advice, diagnosis, or treatment.** Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition. The analysis provided is for informational and educational purposes only.
