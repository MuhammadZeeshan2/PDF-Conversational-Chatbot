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
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text.append(page.extract_text())
    return text

def get_text_chunks_with_metadata(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    text_chunks = []
    for page_number, page_text in enumerate(text):
        chunks = text_splitter.split_text(page_text)
        for chunk in chunks:
            text_chunks.append({
                "chunk": chunk,
                "metadata": {"page_number": page_number + 1}
            })
    return text_chunks

def get_vectorstore(text_chunks_with_metadata):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    texts = [item["chunk"] for item in text_chunks_with_metadata]
    metadata = [item["metadata"] for item in text_chunks_with_metadata]
    print(metadata)

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
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    # docs=retriever.invoke(" Description of Business and Summary of Significant Accounting Policies"),
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                # Retrieve the source documents and metadata from the response
                source_documents = response.get("source_documents", [])
                print(source_documents)
                for doc in source_documents:
                   print(doc.metadata)
                page_numbers = [doc.metadata.get("page_number", "Unknown") for doc in source_documents]
                
                # Assuming a single document match for simplicity
                if page_numbers:
                    page_number = page_numbers[0]
                else:
                    page_number = "Unknown"
                response_with_metadata = f"{message.content} (Page {page_number})"
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
                raw_text_pages = get_pdf_text(pdf_docs)
                text_chunks_with_metadata = get_text_chunks_with_metadata(raw_text_pages)
                vectorstore = get_vectorstore(text_chunks_with_metadata)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()















# import streamlit as st
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# # import torch
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
# from langchain.llms import HuggingFaceHub
# from InstructorEmbedding import INSTRUCTOR
# import os
# from groq import Groq
# from langchain_groq import ChatGroq

# from langchain_community.embeddings.sentence_transformer import (
#     SentenceTransformerEmbeddings,
# )

# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )



# # def get_pdf_text(pdf_docs):
# #     text = ""
# #     for pdf in pdf_docs:
# #         pdf_reader = PdfReader(pdf)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
# #     return text

# def get_pdf_text(pdf_docs):
#     pages_text = []
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             pages_text.append(page.extract_text())
#     return pages_text


# # def get_text_chunks(text):
# #     text_splitter = CharacterTextSplitter(
# #         separator="\n",
# #         chunk_size=1000,
# #         chunk_overlap=200,
# #         length_function=len
# #     )
# #     chunks = text_splitter.split_text(text)
# #     return chunks

# def get_text_chunks_with_metadata(pages_text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len
#     )

#     text_chunks = []
#     for page_number, page_text in enumerate(pages_text):
#         chunks = text_splitter.split_text(page_text)
#         for chunk in chunks:
#             text_chunks.append({
#                 "chunk": chunk,
#                 "metadata": {"page_number": page_number + 1}
#             })
#     return text_chunks






# # def get_vectorstore(text_chunks):
# #     # embeddings = OpenAIEmbeddings()
# #     # device = "cuda" if torch.cuda.is_available() else "cpu"
# #     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# #     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
# #     return vectorstore

# def get_vectorstore(text_chunks_with_metadata):
#     embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     texts = [item["chunk"] for item in text_chunks_with_metadata]
#     metadata = [item["metadata"] for item in text_chunks_with_metadata]
#     print(metadata)

#     vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     # llm = ChatOpenAI()
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
# #     llm = client.chat.completions.create(
# #     messages=[
# #         {
# #             "role": "user",
# #             "content": "Explain tmy resume",
# #         }
# #     ],
# #     model="llama3-8b-8192",

    
# # )
    
#     groq_chat = ChatGroq(
#             groq_api_key=os.environ.get("GROQ_API_KEY"), 
#             model_name="llama3-8b-8192"
#     )

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#         # parent_retriever= ParentDocumentRetriever(vectorstore=vectorstore)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=groq_chat, 
#         retriever=vectorstore.as_retriever(),
#         # retriever=parent_retriever,
#         memory=memory
#     )
#     return conversation_chain


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # # get the text chunks
#                 # text_chunks = get_text_chunks(raw_text)

#                 # # create vector store
#                 # vectorstore = get_vectorstore(text_chunks)
#                 # get the text chunks with metadata
#                 text_chunks_with_metadata = get_text_chunks_with_metadata(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks_with_metadata)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# if __name__ == '__main__':
#     main()
