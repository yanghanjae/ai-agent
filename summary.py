import google.genai as genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # env 없으면 기본값 사용

def summarize_txt(file_path: str):  # ①
    # ② 텍스트 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # ③ 요약 프롬프트 생성
    prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개

    =============== 이하 텍스트 ===============

    { txt }
    '''

    print(prompt)
    print('=========================================')

    # ④ Gemini API로 요약 생성
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text

if __name__ == '__main__':
    file_path = './output/프롬프트 명령어_한글_요약.txt'

    summary = summarize_txt(file_path)
    print(summary)

    # ⑤ 요약 결과 파일로 저장
    os.makedirs('./output', exist_ok=True)
    with open('./output/crop_model_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)