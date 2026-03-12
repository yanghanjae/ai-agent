import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경 변수 로드
# 1. 내 API 키 입력 (식별 과정)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 2. 사용할 AI 모델 선택 (가장 빠르고 가벼운 flash 모델)
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction="너는 배트맨에 나오는 조커야. 조커의 악당 캐릭터에 맞게 답변해줘",
    generation_config={"temperature": 2.0} # 거울이니까 적당한 창의성 부여
)

# 3. AI에게 질문 던지기
# GPT의 {"role": "user", "content": "..."} 부분이 여기에 들어갑니다.
print("마법 거울에게 질문하는 중...\n")
response = model.generate_content("세상에서 누가 제일 아름답니?")

# 4. 답변 출력
print(response.text)