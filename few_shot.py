import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경 변수 로드
# 1. 내 API 키 입력 (식별 과정)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 사용할 AI 모델 선택 (가장 빠르고 가벼운 flash 모델)
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction="너는 유치원 학생이야. 유치원생처럼 답변해줘.",
    generation_config={"temperature": 0.9} 
)

response = model.generate_content([
    {"role": "user",  "parts": ["참새"]},
    {"role": "model", "parts": ["짹짹"]},
    {"role": "user",  "parts": ["말"]},
    {"role": "model", "parts": ["히이잉"]},
    {"role": "user",  "parts": ["개구리"]},
    {"role": "model", "parts": ["개굴개굴"]},
    {"role": "user",  "parts": ["코끼리"]},
])


print(response.text) 