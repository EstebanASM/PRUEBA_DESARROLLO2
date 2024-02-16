import openai
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import langchain
from bs4 import BeautifulSoup
from datetime import datetime

# Desactiva la salida detallada de la biblioteca langchain
langchain.verbose = False

# Carga las variables de entorno desde un archivo .env
load_dotenv()

# Función para procesar el texto extraído de un archivo HTML
def process_text(text):
    # Divide el texto en trozos usando langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Convierte los trozos de texto en incrustaciones para formar una base de conocimientos
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    knowledge_base = FAISS.from_texts(chunks, embeddings) if chunks else None

    return knowledge_base

# Función principal de la aplicación
def main():
    st.title("AIHTML")

    html = st.file_uploader("Sube tu archivo HTML", type="html")
    rss = st.file_uploader("Sube tu archivo RSS", type=["rss", "xml"])

    text = ""

    for file, parser in [(html, 'html.parser'), (rss, 'xml')]:
        if file is not None:
            soup = BeautifulSoup(file, parser)
            text += soup.get_text()

    if text:
        # Crea un objeto de base de conocimientos a partir del texto del HTML
        knowledge_base = process_text(text)

        # Caja de entrada de texto para que el usuario escriba su pregunta
        query = st.text_input('Escribe tu pregunta para los HTMLs...')

        # Botón para cancelar la pregunta
        cancel_button = st.button('Cancelar')

        if cancel_button:
            st.stop()  # Detiene la ejecución de la aplicación

        if query and knowledge_base:
            # Realiza una búsqueda de similitud en la base de conocimientos
            docs = knowledge_base.similarity_search(query)

            # Inicializa un modelo de lenguaje de OpenAI y ajustamos sus parámetros
            model = "gpt-3.5-turbo-instruct" # Acepta 4096 tokens
            temperature = 0  # Valores entre 0 - 1
            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name=model, temperature=temperature)

            # Carga la cadena de preguntas y respuestas
            chain = load_qa_chain(llm, chain_type="stuff")

            # Obtiene la realimentación de OpenAI para el procesamiento de la cadena
            with get_openai_callback() as cost:
                start_time = datetime.now()
                response = chain.invoke(input={"question": query, "input_documents": docs})
                end_time = datetime.now()
                print(cost)  # Imprime el costo de la operación

                st.write(response["output_text"])  # Muestra el texto de salida de la cadena de preguntas y respuestas en la aplicación

                # Muestra el tiempo y el costo
                st.write(f"El costo es:", cost)
                st.write(f"Tiempo de transacción: {end_time - start_time}")

# Punto de entrada para la ejecución del programa
if __name__ == "__main__":
    main()  # Llama a la función principal