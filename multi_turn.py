import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    system_instruction="너는 사용자를 도와주는 상담사야.",
    generation_config={"temperature": 0.9}
)

""" 싱글턴
while True: 
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = model.generate_content(user_input)
    print("AI: " + response.text)
"""

def get_ai_response(messages):
    response = model.generate_content(messages)
    return response.text

messages = []  # system은 system_instruction으로 분리했으므로 빈 리스트로 시작

while True:
    user_input = input("사용자: ")

    if user_input == "exit":  # ② 사용자가 대화를 종료하려는지 확인
        break

    messages.append({"role": "user", "parts": [user_input]})       # 사용자 메시지 추가
    ai_response = get_ai_response(messages)                          # AI 응답 가져오기
    messages.append({"role": "model", "parts": [ai_response]})      # AI 응답 추가

    print("AI: " + ai_response)

