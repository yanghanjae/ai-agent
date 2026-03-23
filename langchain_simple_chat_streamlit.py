import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 사용자의 질문에 친절이 답하는 AI챗봇이다.")
    ]

if "store" not in st.session_state:
    st.session_state["store"] = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = InMemoryChatMessageHistory()
    return st.session_state["store"][session_id]

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GEMINI_API_FREE_KEY") or os.getenv("GEMINI_API_KEY")
)
with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config = {"configurable": {"session_id": "abc2"}}

# 기존 메시지 출력
for msg in st.session_state.messages:
    if msg:
        if isinstance(msg, SystemMessage):
            pass
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            st.chat_message("assistant").write(content)
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    print('user:', prompt)
    st.session_state.messages.append(HumanMessage(prompt))
    st.chat_message("user").write(prompt)

    response = with_message_history.stream([HumanMessage(prompt)], config=config)

    ai_response_bucket = None
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for r in response:
            content = r.content
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            if not content:
                continue

            if ai_response_bucket is None:
                ai_response_bucket = r
            else:
                ai_response_bucket += r

            # 누적 텍스트 추출
            accumulated = ai_response_bucket.content
            if isinstance(accumulated, list) and len(accumulated) > 0:
                accumulated = accumulated[0].get("text", "")

            print(content, end='')
            placeholder.markdown(accumulated)

    if ai_response_bucket:
        st.session_state.messages.append(ai_response_bucket)
        print('assistant:', accumulated)