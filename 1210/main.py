# <main.py>
# uvicorn main:app --reload

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import ChatCompletion, OpenAI
from dotenv import load_dotenv
import logging
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# OpenAI API key setup
load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)

# FastAPI app setup
app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FastAPI models
class UserRequest(BaseModel):
    utterance: str

class BotResponse(BaseModel):
    bot_message: str

# Queue object creation
response_queue = []

# Template directory setup
templates = Jinja2Templates(directory="templates")

# Root endpoint for rendering HTML
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chatbot endpoint
@app.post('/chatbot/', response_model=BotResponse)
async def chat(user_request: UserRequest):
    user_question = user_request.utterance.strip()

    logger.debug(f'Received POST request: {user_question}')

    if user_question:
        gpt_answer = get_qa_by_gpt(user_question)
        response_queue.append(gpt_answer)
        logger.debug(f'Response queue: {response_queue}')
        return JSONResponse(content={'bot_message': gpt_answer})
    else:
        return JSONResponse(content={'bot_message': 'No input received'})

# Additional endpoint to get the response
@app.get('/response/', response_class=JSONResponse)
async def get_response():
    if response_queue:
        return JSONResponse(content={'bot_message': response_queue.pop(0)})
    else:
        return JSONResponse(content={'bot_message': 'No response'})

# Function to get Q&A from GPT
def get_qa_by_gpt(prompt):
# 아래와 같이 큰 따옴표 3개로 여러 줄의 문자열을 정의하세요.
    prompt_template = [
    {"role": "system", "content": """You are a capable and friendly medical assistant with a 'catty' personality.
Your responses should be concise, no more than 50 words, and in Korean.
When the user describes their symptoms, along with their gender and age, provide three possible medical conditions.
If the user mentions less than two symptoms, ask "다른 증상은 없냥?" (Are there any other symptoms?) once more.
If a user mentions three or more of their symptoms during a conversation, they will also respond in the following format.
If the user doesn't describe a symptom, respond with the format: "당신의 증상과 건강상태를 고려하면 유력한 질병은 (질환명1), (질환명2), (질환명3) 일 가능성이 높습니다냥. 이에 따라 당신이 방문해야 할 진료과를 추천드리면 <질환명1과 관련한 진료과 목록>, <질환명2와 관련한 진료과 목록>, <질환명3와 관련한 진료과 목록>입니다냥." (Considering your symptoms and health condition, the likely diseases are (Disease1), (Disease2), (Disease3). Accordingly, the departments you should visit are <List of departments related to Disease1>, <List of departments related to Disease2>, <List of departments related to Disease3>.)
Keep the dialogue tone 'catty'.
Any conversation not related to the broader healthcare field should be immediately ended with "저희는 의료와 관련한 정보만 제공한다냥" (We only provide information related to healthcare)."""},
        {"role": "user", "content": prompt}
    ]


    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo-1106',
            messages=prompt_template
        )

        message = response.choices[0].message.content
        logger.debug(f'GPT response: {message}')
        return message
    except Exception as e:
        logger.error(f'Error during OpenAI API call: {e}')
        return "Sorry, I am unable to process your request at the moment."
