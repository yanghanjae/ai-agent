import pymupdf
import google.genai as genai  # ← 변경
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))  # ← 변경

# ① PDF 파일 열고 텍스트 추출
pdf_file_path = "data/프롬프트 명령어_한글.pdf"
doc = pymupdf.open(pdf_file_path)

full_text = ''

# ② 페이지마다 텍스트 추출해서 합치기
for page in doc:
    text = page.get_text()
    full_text += text

# ③ Gemini에게 PDF 내용 전달해서 요약 요청
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=f"다음 내용을 한국어로 요약해줘:\n\n{full_text}"
)

# ④ 파일명 추출 (확장자 제거)
pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_name = os.path.splitext(pdf_file_name)[0]

# ⑤ output 폴더 없으면 자동 생성 후 저장
os.makedirs("output", exist_ok=True)  # ← 추가
txt_file_path = f"output/{pdf_file_name}_요약.txt"
with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write(response.text)

print("요약 완료!")
print(response.text)