import os
import tempfile
import fitz  # PyMuPDF

from streamlit.components.v1 import html

from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain


from langchain_community.llms import Ollama


from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import webbrowser

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st


TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve(
).parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="TheBrightWay's python expert")
st.title("TheBrightWay's python expert")


def highlight_text_in_pdf(pdf_name, text_to_highlight):
    # Check if PDF file exists
    
    # if not os.path.exists():
    
    #     print("Error: PDF file does not exist.")
    #     return

    # Open the PDF file
    if  os.path.exists(os.path.join(os.getcwd(), f'{pdf_name}')):

        pdf_document = fitz.open(f'{pdf_name}')

    # Loop through each page in the PDF
        for page_number in range(pdf_document.page_count):
          page = pdf_document[page_number]

        # Search for the text and get its bounding box
          text_instances = page.search_for(text_to_highlight)

        # Highlight each instance of the text found on the page
          for text_rect in text_instances:
            # Highlight the text using a rectangle
              highlight = page.add_rect_annot(text_rect)
              highlight.set_colors(stroke=[1, 1, 0], fill=[
                                 1, 1, 0])  # Yellow highlight
              highlight.set_opacity(0.4)  # Set transparency

    # Save the modified PDF with highlights
        output_filename = os.path.splitext(pdf_name)[0] + "_highlighted.pdf"
        pdf_document.save( f'deploy/mysite/static/{output_filename}')
        pdf_document.close()
    # os.startfile(output_filename)
    

    return output_filename


def embeddings_on_pinecone():
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


    persist_directory = 'pythondocumentation/'

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
    st.session_state.messages.append((query, result['answer']))
    # result = result['answer']
    
    return result

def open_page(url):
    open_script = """
        <script type="text/javascript">
            window.open('%s').focus();
        </script>
    """ % (url)
    html(open_script)
def open_pdf(file_path):
  """Opens the specified PDF file in the user's default PDF viewer.

  Args:
      file_path (str): The absolute path to the PDF file on the user's system.
  """
  import os  # Import for cross-platform compatibility
  if os.path.isfile(file_path):
    os.startfile(file_path)
  else:
    st.error(f"Error: PDF file '{file_path}' not found.")

def boot():
    #

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.tab=[]
        st.session_state.a=0
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])

      
    for tab in st.session_state.tab:
    
        
      
        st.button(tab[0], on_click=lambda: webbrowser.open('http://localhost:8000/static/' + tab[1]))
        


    #
    if query := st.chat_input():
        st.session_state.tab=[]
        

        st.chat_message("human").write(query)
        response = query_llm(embeddings_on_pinecone(), query)
        st.chat_message("ai").write(response['answer'])  
       
        

        for i in range(4):
               

             name_pdf =  response["source_documents"][i].metadata['source']
             page =  response["source_documents"][i].metadata['page_number']
       
             file_name = highlight_text_in_pdf(name_pdf, response['source_documents'][i].page_content)
             st.session_state.a+=1
             st.button(name_pdf + str(st.session_state.a)+"Page number"+page, on_click=lambda: webbrowser.open('http://localhost:8000/static/' + file_name))

               
              
             st.session_state.tab.append((name_pdf + str(st.session_state.a)+"Page number"+page,file_name))
        

      
     



if __name__ == '__main__':

    boot()
