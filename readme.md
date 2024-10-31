# PDF Conversational Chatbot

A web-based application that allows users to upload multiple PDF documents and engage in a conversation with the content. The system utilizes language models, vector storage, and conversational memory to retrieve and provide intelligent answers based on the uploaded documents. 

**What makes this application unique** is that it not only provides the answers to your queries but also includes the **PDF name** and **page number** from which the response was retrieved, offering full transparency and traceability.
Check out the user interface below:
![GUI Overview](__screenshorts/streamlit-gui.png?raw=true "GUI Overview")


---

## Features

- **Multiple PDF Support**: Upload and process multiple PDFs.
- **Vector Store (FAISS)**: Efficient storage of document chunks for similarity search.
- **Conversational Retrieval Chain**: Engage in a conversational AI system that recalls the context.
- **Metadata Tracking**: Keeps track of page numbers and PDF file names for reference.
- **Embeddings**: Uses Sentence Transformer models for text embedding.
- **Source Tracking**: Provides the name and page number of the PDF from which the answer is retrieved.

## Tech Stack

- **Streamlit**: For the web interface.
- **LangChain**: Conversational AI framework.
- **FAISS**: Vector storage for document search.
- **PyPDF2**: To extract text from PDFs.
- **Groq LLM**: For generating responses using the `llama3-8b-8192` model.
- **Instructor Embeddings**: For custom embeddings.

## Project Structure

```bash
📁 env/                  # Virtual environment
📁 __pycache__/          # Python cache files
📁 __screenshorts/       # Screenshots of the app
📄 .env                  # Environment variables (api keys, etc.)
📄 .gitignore            # Ignored files for git
📄 .python-version       # Python version
📄 app.py                # Main application code
📄 htmlTemplates.py      # HTML templates for the app's UI
📄 requirements.txt      # Python dependencies
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/MuhammadZeeshan2/PDF-Conversational-Chatbot.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd pdf-conversational-chatbot
   ```

3. **Create a virtual environment**:

   ```bash
   python -m venv env
   ```

4. **Activate the virtual environment**:

   ```bash
   # On Windows
   env\Scripts\activate

   # On MacOS/Linux
   source env/bin/activate
   ```

5. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

6. **Set up the environment variables**:

   Create a `.env` file in the root directory and add your API keys:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingfacehub_api_token_here
   ```

7. **Run the application**:

   ```bash
   streamlit run app.py
   ```

   The app will be available at `http://localhost:8501`.

## Usage

1. **Upload PDFs**: Use the sidebar to upload multiple PDF files.
2. **Process PDFs**: Click on the **Process** button and wait until the processing is complete.
3. **Ask Questions**: Once processing is done, type your query about the documents in the input field.
4. **Receive Answers**: The app will return an answer along with the PDF source and page number from which the answer was retrieved.

## Dependencies

- `streamlit`
- `dotenv`
- `PyPDF2`
- `langchain`
- `langchain_community`
- `instructor-embedding`
- `faiss-cpu`
- `sentence-transformers`
- `groq`

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

---

