import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from datetime import datetime
import pytz
from langchain_tavily import TavilySearch
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
from requests import Session
from typing import List
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

@tool
def get_web_search(query: str) -> str:
    """
    웹 검색을 수행하는 함수.

    Args:
        query (str): 검색어
    """
    print('-------- WEB SEARCH --------')
    print(query)
    search = TavilySearch(max_results=5)
    result = search.invoke({"query": query})
    return str(result)

@tool
def get_youtube_search(query: str) -> List:
    """
    유튜브 검색을 한 뒤, 영상들의 내용을 반환하는 함수.

    Args:
        query (str): 검색어
    """
    print('-------- YOUTUBE SEARCH --------')
    print(query)

    http_client = Session()
    http_client.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.youtube.com/",
    })
    ytt_api = YouTubeTranscriptApi(http_client=http_client)

    videos = YoutubeSearch(query, max_results=5).to_dict()
    videos = [video for video in videos if len(str(video['duration'])) <= 5]

    for video in videos:
        video_url = 'http://youtube.com' + video['url_suffix']
        video_id = video['id']
        video['video_url'] = video_url

        try:
            transcript_list = ytt_api.list(video_id)
            available_langs = [t.language_code for t in transcript_list]
            print(f"사용 가능한 자막: {available_langs}")

            if 'ko' in available_langs:
                transcript = ytt_api.fetch(video_id, languages=['ko'])
            elif 'en' in available_langs:
                transcript = ytt_api.fetch(video_id, languages=['en'])
            else:
                print(f"자막 없음: {video_url}")
                video['content'] = ""
                continue

            content = " ".join([t.text for t in transcript])
            video['content'] = content
        except Exception as e:
            print(f"자막 로드 실패: {video_url} - {e}")
            video['content'] = ""

    videos = [v for v in videos if v['content']]
    if not videos:
        return "유튜브에서 자막이 있는 영상을 찾지 못했습니다. 웹 검색을 이용해주세요."
    return videos


# 도구 바인딩
tools = [get_current_time, get_web_search, get_youtube_search]
tool_dict = {
    "get_current_time": get_current_time,
    "get_web_search": get_web_search,
    "get_youtube_search": get_youtube_search
}

llm_with_tools = llm.bind_tools(tools)


def get_ai_response(messages):
    response = llm_with_tools.invoke(messages)
    tool_calls = response.tool_calls

    if tool_calls:
        st.session_state.messages.append(response)

        for tool_call in tool_calls:
            selected_tool = tool_dict[tool_call['name']]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg)
            st.session_state.messages.append(tool_msg)

        final_response = llm_with_tools.invoke(st.session_state.messages)
        content = final_response.content
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "")
        yield content
    else:
        content = response.content
        print("content type:", type(content))  # ← 추가
        print("content:", content)     
        if isinstance(content, list) and len(content) > 0:
            content = content[0].get("text", "")
        if content:
            yield content


# Streamlit 앱
st.title("💬 Gemini Langchain Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        SystemMessage(f"""너는 사용자를 돕기 위해 최선을 다하는 인공지능 봇이다.
최신 정보가 필요하거나 모르는 내용은 반드시 get_web_search 함수를 사용해서 검색해라.
오늘 날짜는 {datetime.now().strftime('%Y년 %m월 %d일')} 이다."""),
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
    st.session_state["messages"].append(AIMessage(str(result) if result else ""))