from openai import OpenAI


from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
texto = "Olá mundo, este áudio foi gerado pela API da OPNEAI"

# testando áudio
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="nova",
    input=texto,
    response_format="mp3"
) as response:
    response.stream_to_file("teste.mp3")

