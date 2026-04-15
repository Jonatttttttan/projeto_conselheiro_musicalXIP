from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os

CAMINHO_DB = "db"
load_dotenv()


prompt_template = '''
Você é um agente especialista em dar conselhos com base nas músicas da Taylor Swift.
O usuário vai fazer uma pergunta ou pedir um conselho {pergunta} e você vai acessar a base de dados das letras das
músicas da Taylor Swift {musicas} e vai responder com base somente nela, não precisa escrever o nome da música em que tirou o conselho, como se você fosse a própria cantora.'''

def perguntar():
    pergunta_usuario =  input("O que você deseja perguntar hoje?")

    funcao_embedding = OpenAIEmbeddings()
    db =  Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta_usuario, k=5)
    if len(resultados) == 0:
        print("Nenhum resultado foi encontrado")
        return

    texto_resultados = []
    for resultado in resultados:
        texto_resultados.append(resultado[0].page_content)

    base_conhecimento = "\n\n----\n\n".join(texto_resultados)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"musicas" : base_conhecimento, "pergunta" : pergunta_usuario})


    modelo = ChatOpenAI(model="gpt-5.2", temperature=0)
    texto_resposta = modelo.invoke(prompt).content

    # Camada de audio
    print(texto_resposta)
    return
perguntar()

