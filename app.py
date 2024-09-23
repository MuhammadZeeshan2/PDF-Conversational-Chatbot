import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from InstructorEmbedding import INSTRUCTOR
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_pdf_text(pdf_docs):
    text = []
    filenames = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        filenames.append(pdf.name)  # Get the filename
        file_text = []
        for page in pdf_reader.pages:
            file_text.append(page.extract_text())
        text.append(file_text)
    return text, filenames


def get_text_chunks_with_metadata(text, filenames):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = []
    for pdf_index, pdf_text in enumerate(text):
        pdf_name = filenames[pdf_index]
        for page_number, page_text in enumerate(pdf_text):
            chunks = text_splitter.split_text(page_text)
            for chunk in chunks:
                text_chunks.append({
                    "chunk": chunk,
                    "metadata": {
                        "page_number": page_number + 1,
                        "pdf_name": pdf_name  # Store the PDF name
                    }
                })
    return text_chunks


def get_vectorstore(text_chunks_with_metadata):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    texts = [item["chunk"] for item in text_chunks_with_metadata]
    metadata = [item["metadata"] for item in text_chunks_with_metadata]

    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata)
    return vectorstore

def get_conversation_chain(vectorstore):
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"), 
        model_name="llama3-8b-8192"
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_chat, 
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    
    )
    print(conversation_chain)

    return conversation_chain




def handle_userinput(user_question):
    # Ensure vectorstore exists
    if "vectorstore" not in st.session_state:
        st.error("Vectorstore is not initialized. Please upload and process your documents.")
        return
    
    vectorstore = st.session_state.vectorstore
    query = user_question
    results = vectorstore.similarity_search_with_score(query, k=3) 

    # Initialize a list to store page numbers and PDF names
    page_numbers = []
    pdf_names = []

    # Extract document content, metadata, and page numbers from the search results
    for doc, score in results:
        print("Document Content:", doc.page_content)
        print("Metadata:", doc.metadata)
        print("Score:", score)
        # Append the page number and PDF name from the metadata
        page_numbers.append(doc.metadata.get("page_number", "Unknown"))
        pdf_names.append(doc.metadata.get("pdf_name", "Unknown"))
    
    print(pdf_names)

    if st.session_state.conversation:
        # Get the response from the conversation chain
        response = st.session_state.conversation({'question': user_question})
        
        # Separate the answer and source documents
        answer = response.get("answer", "")
        print("Answer", answer)
        source_documents = response.get("source_documents", [])
    
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                # User's message
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Append the page numbers and PDF names to the bot's response
                pdf_info = ', '.join([f"Page {page} from {pdf}" for page, pdf in zip(page_numbers, pdf_names)])
                response_with_metadata = f"{message.content}\n\n (Sources: \n{pdf_info})"
                st.write(bot_template.replace("{{MSG}}", response_with_metadata), unsafe_allow_html=True)
    else:
        st.error("Conversation object is not initialized. Please process your documents first.")



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                raw_text_pages, filenames = get_pdf_text(pdf_docs)  # get both text and filenames
                text_chunks_with_metadata = get_text_chunks_with_metadata(raw_text_pages, filenames)  # pass both arguments
                vectorstore = get_vectorstore(text_chunks_with_metadata)
                
                # Store the vectorstore in session_state
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()






