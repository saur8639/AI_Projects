import os
import re
import json
import smtplib
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from openai import AsyncOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from email.message import EmailMessage
import speech_recognition as sr
from openai.helpers import LocalAudioPlayer

path = Path(__file__).parent

load_dotenv()

def stt_processor():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1

        print(f"\n🤖: Say something ... ")
        audio = r.listen(source)

        stt = r.recognize_google(audio)
        return stt
    
async def tts_processor(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model = "gpt-4o-mini-tts",
        voice = "ash",
        instructions = "Speak in a very professional tone",
        input = speech,
        response_format = "pcm",
    ) as voice_response:
        await LocalAudioPlayer().play(voice_response)

def mailer(rcv_mail_id, msg_body):
    msg = EmailMessage()
    msg['Subject'] = "Saurabh's AI Assistant"
    msg['From'] = os.getenv("SENDER_MAIL_ID")
    msg['To'] = rcv_mail_id

    msg.set_content(msg_body, subtype = 'html')

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(os.getenv("SENDER_MAIL_ID"), os.getenv("GMAIL_APP_PASSWORD"))
            server.send_message(msg)
        print(f"\n🤖:Email sent successfully !\n")
    except Exception as e:
        print(f"\n🤖:Error: {e}\n")

chatter_ai = OpenAI()
async_client = AsyncOpenAI()

loader = PyPDFDirectoryLoader(path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 128
)

chunks = text_splitter.split_documents(documents = docs)

embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large")

vector_store_input = QdrantVectorStore.from_documents(
    documents = chunks,
    embedding = embedding_model,
    url = "http://localhost:6333/",
    collection_name = "rag_for_abc"
)

vector_store_output = QdrantVectorStore.from_existing_collection(
    embedding = embedding_model,
    url = "http://localhost:6333/",
    collection_name = "rag_for_abc"
)

message_log = []
prev_response = ""

while True:
    #user_input = input("\n\nAsk me... ")
    user_input = stt_processor()
    user_input = re.sub(r"kbc", "abc", user_input.lower())

    if (bool(re.search("stop", user_input.lower()))):
        #print(f"\nOk ! It was nice talking to you. Good Bye !\n")
        asyncio.run(tts_processor("Ok ! It was nice talking to you. Good Bye !"))
        break

    search_result = vector_store_output.similarity_search(query = user_input)
    context = "\n".join([result.page_content for result in search_result])

    SYSTEM_PROMPT_CHATTER = f'''
    You are an AI assistant who needs to provide response in json formatted string. You need to strictly follow one of the below rules based on user query:
    Rule 1: If the user is asking anything not related to ABC bank then create a key named 'text' & put your response as the value to it. Do not add statements like can I help you with anything else after your response.
    Rule 2: If the user is asking something about ABC bank then use the context {context} to respond to user queries. You also need to generate html code for your response. You final reponse should strictly be a json formatted string with first key named 'text' which should carry your text format reply as value which should also be followed by can I help you with anything else in a new line & the second key as 'code' which should carry the html code as value that you created which carries your text reply without any mention of can I help you with anything kind of statements in the message body.
    Don't generate any other output.
    Rule 3: If the user only asks to mail something then generate only the previous output as per Rule 2. Don't generate any other output.
    '''
    message_log.append({"role":"system", "content":SYSTEM_PROMPT_CHATTER})
    message_log.append({"role":"user", "content":user_input})

    response = chatter_ai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = message_log
    )

    try:
        ai_response = json.loads(response.choices[0].message.content)
    except:
        #print(f"\n🤖: Something didn't work as expected. Please try again !")
        asyncio.run(tts_processor("Something didn't work as expected. Please try again !"))
        continue

    if (bool(re.search("mail", user_input.lower()))):
        mailer(os.getenv("SENDER_MAIL_ID"), ai_response.get("code"))
    
    if ((ai_response.get("text") != prev_response) | (bool(re.search("repeat|again", user_input.lower())))):
        #print(f"\n🤖: {ai_response.get("text")}")
        asyncio.run(tts_processor(ai_response.get("text")))

    prev_response = ai_response.get("text")
    #message_log.append({"role":"assistant", "content":prev_response})

    os.system('cls')