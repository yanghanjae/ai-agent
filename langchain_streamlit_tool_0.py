import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
import os

load_dotenv()

# 모델 초기화
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GEMINI_API_FREE_KEY") or os.getenv("GEMINI_API_KEY")
)

def get_ai_response(messages):
    response = llm.stream(messages)
    for chunk in response:
        content = chunk.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "")
        if content:
            yield content  # 텍스트만 yield

# Streamlit 앱
st.title("💬 Gemini Langchain Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다."),
        AIMessage("How can I help you?")
    ]

for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        pass
    elif isinstance(msg, AIMessage):
        content = msg.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "")
        if content:
            st.chat_message("assistant").write(content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    response = get_ai_response(st.session_state["messages"])

    result = st.chat_message("assistant").write_stream(response)
    st.session_state["messages"].append(AIMessage(result))