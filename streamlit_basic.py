import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# (0) 사이드바
with st.sidebar:
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    "[Get a Gemini API key](https://aistudio.google.com/app/apikey)"

st.title("💬 Chatbot")

# (1) session_state에 "messages"가 없으면 초기값 설정
#     assistant → model로 변경
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "model", "parts": ["How can I help you?"]}]

# (2) 대화 기록 출력
#     Gemini는 role이 "model"이지만 UI에는 "assistant"로 표시
for msg in st.session_state.messages:
    display_role = "assistant" if msg["role"] == "model" else "user"
    st.chat_message(display_role).write(msg["parts"][0])

# (3) 사용자 입력 처리
if prompt := st.chat_input():
    if not gemini_api_key:
        st.info("Please add your Gemini API key to continue.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name='gemini-2.5-flash')

    st.session_state.messages.append({"role": "user", "parts": [prompt]})
    st.chat_message("user").write(prompt)

    response = model.generate_content(st.session_state.messages)
    msg = response.text

    st.session_state.messages.append({"role": "model", "parts": [msg]})
    st.chat_message("assistant").write(msg)