import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from datetime import datetime
import pytz
from dotenv import load_dotenv
import os

load_dotenv()

# 모델 초기화
llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GEMINI_API_FREE_KEY") or os.getenv("GEMINI_API_KEY")
)

# 도구 함수 정의
@tool
def get_current_time(timezone: str, location: str) -> str:
    """현재 시각을 반환하는 함수."""
    try:
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        result = f'{timezone} ({location}) 현재시각 {now}'
        print(result)
        return result
    except pytz.UnknownTimeZoneError:
        return f"알 수 없는 타임존: {timezone}"

tools = [get_current_time]
tool_dict = {"get_current_time": get_current_time}
llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
    # Gemini는 tool_calls 스트리밍 시 thought_signature 문제로
    # 첫 호출은 stream=False로 처리
    response = llm_with_tools.invoke(messages)

    tool_calls = response.tool_calls

    if tool_calls:
        # 도구 호출 필요한 경우
        st.session_state.messages.append(response)

        for tool_call in tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg)
            st.session_state.messages.append(tool_msg)

        # 최종 답변은 스트리밍으로
        final_response = llm_with_tools.stream(st.session_state.messages)
        for chunk in final_response:
            content = chunk.content
            if isinstance(content, list) and len(content) > 0:
                content = content[0].get("text", "")
            if content:
                yield content
    else:
        # 도구 없으면 바로 텍스트 반환
        content = response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "")
        if content:
            yield content


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
    elif isinstance(msg, ToolMessage):
        st.chat_message("tool").write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append(HumanMessage(prompt))

    response = get_ai_response(st.session_state["messages"])

    result = st.chat_message("assistant").write_stream(response)
    st.session_state["messages"].append(AIMessage(result))