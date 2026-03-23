import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import retriever
from dotenv import load_dotenv
import os

load_dotenv()

# 모델 초기화
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GEMINI_API_FREE_KEY") or os.getenv("GEMINI_API_KEY")
)

# 사용자의 메시지 처리하기 위한 함수
def get_ai_response(messages, docs):
    response = retriever.document_chain.stream({
        "messages": messages,
        "context": docs
    })
    for chunk in response:
        if isinstance(chunk, str) and chunk:
            yield chunk

# Streamlit 앱
st.title("💬 Gemini Langchain Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage("너는 문서에 기반해 답변하는 도시 정책 전문가야"),
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

    augmented_query = str(retriever.query_augmentation_chain.invoke({
        "messages": st.session_state["messages"],
        "query": prompt,
    }))
    print("augmented_query\t", augmented_query)

    # 관련 문서 검색
    print("관련 문서 검색")
    docs = retriever.retriever.invoke(f"{prompt}\n{augmented_query}")

    for doc in docs:
        print('---------------')
        print(doc)
        with st.expander(f"**문서:** {doc.metadata.get('source', '알 수 없음')}"):
            st.write(f"**page:**{doc.metadata.get('page', '')}")
            st.write(doc.page_content)
    print("===============")

    with st.spinner(f"AI가 답변을 준비 중입니다... '{augmented_query}'"):
        response = get_ai_response(st.session_state["messages"], docs)
        result = st.chat_message("assistant").write_stream(response)
    st.session_state["messages"].append(AIMessage(result))