import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from maikol_utils.print_utils import print_separator

from src.config import Configuration

import warnings

warnings.filterwarnings("ignore")

def cargar_y_indexar(CONFIG: Configuration):
    embeddings = OllamaEmbeddings(model=CONFIG.rag_embedding_model)

    if os.path.exists(CONFIG.db_path):
        print(" - Cargando base de datos existente...")
        return Chroma(persist_directory=CONFIG.db_path, embedding_function=embeddings)
    print_separator(f"Escaneando PDFs en {CONFIG.pdf_path}")
    
    # DirectoryLoader busca todos los archivos .pdf y usa PyPDFLoader para cada uno
    loader = DirectoryLoader(
        CONFIG.pdf_path,
        glob="./*.pdf",
        loader_cls=PyPDFLoader
    )
    
    docs = loader.load()
    print(f" - Documentos cargados: {len(docs)} páginas encontradas.")

    # Dividimos el texto en trozos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print(f" - Creando base de datos vectorial con {len(splits)} fragmentos...")
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CONFIG.db_path
    )
    return vectorstore

def evaluar_relevancia(pregunta, documentos, llm) -> str:
    """
    Determina si los documentos recuperados son útiles.
    """
    prompt_evaluador = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador de relevancia. Responde estrictamente con una palabra: 'SI' o 'NO'."),
        ("human", f"Pregunta: {pregunta} \n\n Contexto recuperado: {documentos} \n\n ¿Contiene este contexto la información necesaria para responder?")
    ])
    
    chain = prompt_evaluador | llm | StrOutputParser()

    respuesta = chain.invoke({}).strip().upper()
    return "SI" in respuesta

def ask_rag(transcription:str, CONFIG:Configuration) -> str:
    vectorstore = cargar_y_indexar(CONFIG)
    llm = OllamaLLM(model=CONFIG.rag_model_name)
    search_tool = TavilySearchResults(k=5, search_depth="advanced",)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs_locales = retriever.invoke(transcription)
    contexto_local = "\n".join([d.page_content for d in docs_locales])

    es_relevante = evaluar_relevancia(transcription, contexto_local, llm)

    if es_relevante:
        print(" - [OK] Información encontrada en documentos locales.")
        contexto_final = contexto_local
        fuente_info = "tus documentos internos"
    else:
        print(" - [!] Información local insuficiente o irrelevante. Consultando internet...")
        web_results = search_tool.invoke({"query": transcription})
        contexto_final = "\n".join([res["content"] for res in web_results])
        fuente_info = "fuentes externas en internet"

    system_prompt = (
        "Eres un asistente de investigación. Responde de forma natural, fluida y concisa, no des detalles innecesarios y ves directamente al grano. Si breve, no uses más de 100 palabras. "
        f"La información proviene de {fuente_info}. "
        "IMPORTANTE: No uses negritas, ni asteriscos, ni listas, ni ningún formato markdown. "
        "Escribe todo en un párrafo o párrafos de texto plano, como si fuera una carta o un mensaje de voz. Independientemente del lenguaje de la pregunta, responde SIEMPRE en Español. "
        "\n\nContexto: {context}"
    )

    prompt_final = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain_final = prompt_final | llm | StrOutputParser()
    
    respuesta = chain_final.invoke({
        "context": contexto_final, 
        "input": transcription
    })

    return respuesta