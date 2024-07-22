import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.document_loaders import UnstructuredURLLoader
import tempfile

def common_code(documents,file_type):
    # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Print the number of chunks
        st.write(f"Number of chunks: {len(docs)}")

        # Create embeddings and Chroma database
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(docs, embedding_function)

        # Input query and search button
        query = st.text_input("Enter what you want to search:")
        button = st.button("Search")
        if button:
            results = db.similarity_search(query)
            if results:
                st.header("Answer")
                st.text_area("",results[0].page_content)
            else:
                st.header("Answer")
                st.text_area("No results found.")

def RAG_text_loader():
    uploaded_txt_file = st.file_uploader("Choose a txt file", type=["txt"])
    if uploaded_txt_file is not None:
        # Read the uploaded file content
        file_content = uploaded_txt_file.read().decode("utf-8")

        # Write the content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(file_content.encode("utf-8"))
            temp_file_path = temp_file.name

        # Use TextLoader to process the temporary file
        loader = TextLoader(temp_file_path)
        documents = loader.load()
        common_code(documents,"txt")
        

def RAG_pdf_loader():
    uploaded_pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_pdf_file is not None:
        # Read the uploaded file content
        # Write the content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_pdf_file.read())
            temp_file_path = temp_file.name

        # Use PDFPlumberLoader to process the temporary file
        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()
        common_code(documents,"pdf")

def RAG_url_loader():
    urls=[]
    url = st.text_input(f"Article URL")
    if url :
        urls.append(url)
        # Read the uploaded file content
        # Write the content to a temporary file
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        common_code(documents,"url")
        del urls


st.title("Retrieval-Augmented Generation ")


# Display the RAG image
image_path = 'rag.png'  # Update with the actual path to your image
st.image(image_path, caption='RAG Image', use_column_width=True)
# Sidebar for file type selection
st.sidebar.title("File Type Selection")
file_type = st.sidebar.radio("Choose file type", ("Text", "PDF","URL"))

if file_type == "Text":
    st.title("Bot: Text Search Tool 📋")
    RAG_text_loader()
elif file_type == "PDF":
    st.title("Bot: Document Search Tool 📖")
    RAG_pdf_loader()  
elif file_type == "URL":
    st.title("Bot: Research Tool 🔗")
    RAG_url_loader()   
