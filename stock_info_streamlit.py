from gemini_functions import get_current_time, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations
import google.genai as genai
import google.genai.types as types
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

get_current_time_func = types.FunctionDeclaration(
    name="get_current_time",
    description="해당 타임존의 날짜와 시간을 반환합니다.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "timezone": types.Schema(
                type=types.Type.STRING,
                description="현재 날짜와 시간을 반환할 타임존을 입력하세요. (예: Asia/Seoul)"
            )
        },
        required=["timezone"]
    )
)

get_yf_stock_info_func = types.FunctionDeclaration(
    name="get_yf_stock_info",
    description="해당 종목의 Yahoo Finance 정보를 반환합니다.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "ticker": types.Schema(
                type=types.Type.STRING,
                description="Yahoo Finance 정보를 반환할 종목의 티커를 입력하세요. (예: AAPL)"
            )
        },
        required=["ticker"]
    )
)

get_yf_stock_history_func = types.FunctionDeclaration(
    name="get_yf_stock_history",
    description="해당 종목의 Yahoo Finance 주가 정보를 반환합니다.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "ticker": types.Schema(
                type=types.Type.STRING,
                description="Yahoo Finance 주가 정보를 반환할 종목의 티커를 입력하세요. (예: AAPL)"
            ),
            "period": types.Schema(
                type=types.Type.STRING,
                description="주가 정보를 조회할 기간을 입력하세요. (예: 1d, 5d, 1mo, 1y, 5y)"
            )
        },
        required=["ticker", "period"]
    )
)

get_yf_stock_recommendations_func = types.FunctionDeclaration(
    name="get_yf_stock_recommendations",
    description="해당 종목의 Yahoo Finance 추천 정보를 반환합니다.",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "ticker": types.Schema(
                type=types.Type.STRING,
                description="Yahoo Finance 추천 정보를 반환할 종목의 티커를 입력하세요. (예: AAPL)"
            )
        },
        required=["ticker"]
    )
)

gemini_tools = types.Tool(function_declarations=[
    get_current_time_func,
    get_yf_stock_info_func,
    get_yf_stock_history_func,
    get_yf_stock_recommendations_func
])


def get_ai_response_stream(messages, tools=None):
    """최종 답변 스트리밍용"""
    response = client.models.generate_content_stream(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[tools] if tools else None,
            system_instruction="너는 사용자를 도와주는 상담사야. 주식 정보가 필요하면 반드시 함수를 사용해."
        )
    )
    for chunk in response:
        yield chunk


def get_ai_response(messages, tools=None):
    """함수 호출 확인용 (스트리밍 없이)"""
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[tools] if tools else None,
            system_instruction="너는 사용자를 도와주는 상담사야. 주식 정보가 필요하면 반드시 함수를 사용해."
        )
    )
    return response


st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    if msg.role in ("user", "model"):
        display_role = "assistant" if msg.role == "model" else "user"
        st.chat_message(display_role).write(msg.parts[0].text)


if user_input := st.chat_input():
    st.session_state.messages.append(
        types.Content(role="user", parts=[types.Part(text=user_input)])
    )
    st.chat_message("user").write(user_input)

    # 첫 호출은 stream=False로 function_call 확인
    ai_response = get_ai_response(st.session_state.messages, tools=gemini_tools)
    ai_content = ai_response.candidates[0].content
    function_calls = [p for p in ai_content.parts if p.function_call]

    if function_calls:
        # AI 응답 저장 (thought_signature 포함된 원본 content 그대로 저장)
        st.session_state.messages.append(ai_content)

        for part in function_calls:
            func_call = part.function_call
            tool_name = func_call.name
            arguments = dict(func_call.args)

            if tool_name == "get_current_time":
                func_result = get_current_time(timezone=arguments['timezone'])
            elif tool_name == "get_yf_stock_info":
                func_result = get_yf_stock_info(ticker=arguments['ticker'])
            elif tool_name == "get_yf_stock_history":
                func_result = get_yf_stock_history(
                    ticker=arguments['ticker'],
                    period=arguments['period']
                )
            elif tool_name == "get_yf_stock_recommendations":
                func_result = get_yf_stock_recommendations(ticker=arguments['ticker'])

            if func_result is None:
                func_result = "데이터를 가져올 수 없습니다."

            st.session_state.messages.append(types.Content(
                role="tool",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"result": func_result}
                    )
                )]
            ))

        # 최종 답변 스트리밍으로 출력하는 부분 수정
        content = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()  # ← empty()를 안에서 따로 선언
            for chunk in get_ai_response_stream(st.session_state.messages, tools=gemini_tools):
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        content += part.text
                        placeholder.markdown(content)

    else:
      content = ""
      with st.chat_message("assistant"):
          placeholder = st.empty()  # ← 동일하게 수정
          for part in ai_content.parts:
              if part.text:
                  content += part.text
                  placeholder.markdown(content)


    st.session_state.messages.append(
        types.Content(role="model", parts=[types.Part(text=content)])
    )