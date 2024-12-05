import os, tempfile
import fitz  # PyMuPDF
import requests  # For asynchronous requests

# Placeholder server-side URL (replace with your actual URL)
server_url = "http://localhost:8501/pages"

from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain


from langchain_community.llms import Ollama


from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma



from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="TheBrightWay's python expert")
st.title("TheBrightWay's python expert")

def highlight_text_in_pdf(pdf_name, text_to_highlight):
    # Check if PDF file exists
    print("hello")
    if not os.path.exists(pdf_name):
        print("Error: PDF file does not exist.")
        return
    
    # Open the PDF file
    pdf_document = fitz.open(pdf_name)
    
    # Loop through each page in the PDF
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        
        # Search for the text and get its bounding box
        text_instances = page.search_for(text_to_highlight)
        
        # Highlight each instance of the text found on the page
        for text_rect in text_instances:
            # Highlight the text using a rectangle
            highlight = page.add_rect_annot(text_rect)
            highlight.set_colors(stroke=[1, 1, 0], fill=[1, 1, 0])  # Yellow highlight
            highlight.set_opacity(0.4)  # Set transparency
            
    # Save the modified PDF with highlights
    output_filename = os.path.splitext(pdf_name)[0] + "_highlighted.pdf"
    pdf_document.save(output_filename)
    pdf_document.close()
    os.startfile(output_filename) 
    return pdf_name
    
    
    print("Text highlighted and saved to:", output_filename)
# def load_documents():
#     loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
#     documents = loader.load()
#     return documents

# def split_documents(documents):
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(documents)
#     return texts

# def embeddings_on_local_vectordb(texts):
#     model_name = "BAAI/bge-base-en-v1.5"
#     model_kwargs = {"device": "cuda"}
#     encode_kwargs = {"normalize_embeddings": True}
#     embedding_model = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
#     )
    
#     persist_directory = 'aa/'

#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
#     retriever = vectordb.as_retriever()
#     return retriever

def embeddings_on_pinecone():
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    # pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
    # embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
    # vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    persist_directory = 'fct/'

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever()
    return retriever

def query_llm(retriever, query):
   
    llm = Ollama(model="nous-hermes2:10.7b")
    qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    return_source_documents=True,
    )
    
       
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    # result = result['answer']
    st.session_state.messages.append((query, result['answer']))
    return result

def callback():
    name_pdf = "louka.pdf"
    text_to_highlight = "Conflict of interests: ODDO BHF Corporate Markets, a division of ODDO BHF SCA, limited sharepartnership"

    st.write("Highlighting text in PDF...")
    response = requests.post(server_url, json={"name_pdf": name_pdf, "text_to_highlight": text_to_highlight})

    if response.status_code == 200:
        st.success("Highlighting completed. Download link (placeholder):")
        # ... (display download link if applicable)
    else:
        st.error("An error occurred. Please try again.")


def boot():
    #
    # input_fields()
    # name_pdf=None
    # text_to_highlight=None
    # st.button("Submit Documents", on_click=highlight_text_in_pdf)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        # response = query_llm(embeddings_on_pinecone(), query)
        # st.chat_message("ai").write(response['answer'])
        st.chat_message("ai").write("jjj")

        # name_pdf =  response["source_documents"][1].metadata['source']
        # print(name_pdf)
        # print(response["source_documents"][i].page_content)
        # highlight_text_in_pdf(name_pdf,response["source_documents"][1].page_content)
        # execute_python_function("highlight_text_in_pdf", name_pdf, response["source_documents"][1].page_content)
        

        name_pdf = "louka.pdf"
        text_to_highlight = "Conflict of interests: ODDO BHF Corporate Markets, a division of ODDO BHF SCA, limited sharepartnership"
        highlight_text_in_pdf(name_pdf, text_to_highlight) 
       


           


if __name__ == '__main__':
    
    boot()
    