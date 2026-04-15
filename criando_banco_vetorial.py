from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv
import os

load_dotenv()
CAMINHO = "C:\\Users\\Nitro\\Desktop\\Python_IA_projetos\\Projeto_Conselheiro_Musical\\db_musics"
def capturar_letras(caminho):
    files = [caminho + "\\" + u for u in os.listdir(caminho)]
    letras = []
    i = 0
    for file in files:
        i += 1
        print(i)
        with open(file, "r", encoding="utf-8") as f:
            letra = f.read()
            letras.append(letra)

    return letras

# capturar_letras(CAMINHO)

def criar_db_vetor():
    documentos = capturar_letras(CAMINHO)
    chunks = criar_chunks(documentos)
    vetorizar_chunks(chunks)
    return

def criar_chunks(documentos):
    docs = [Document(page_content=texto) for texto in documentos]
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(docs)
    return chunks

def vetorizar_chunks(chunks):
    Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory='db')
    print("Banco de dados criado")

if __name__ == "__main__":
    criar_db_vetor()






