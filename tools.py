from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

# RAG를 위한 설정
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Gemini Embedding 설정
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_FREE_KEY") or os.getenv("GEMINI_API_KEY")
)

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        dict: 검색 결과
    """
    client = TavilyClient()

    content = client.search(
        query,
        search_depth="advanced",
        include_raw_content=True,
    )

    results = content["results"]

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    resources_json_path = f'{current_path}/data/resources_{datetime.now().strftime("%Y_%m%d_%H%M%S")}.json'
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results, resources_json_path


def web_page_to_document(web_page):
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']

    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url']
        }
    )
    return document


def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        resources = json.load(f)

    documents = []
    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)
    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits


def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    urls = [document.metadata['source'] for document in documents]
    stored_metadatas = vectorstore._collection.get()['metadatas']
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas]
    new_urls = set(urls) - set(stored_web_urls)

    new_documents = []
    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if splits:
        # 20개씩 나눠서 저장 + 대기
        batch_size = 10
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i+batch_size]
            vectorstore.add_documents(batch)
            print(f"{i+len(batch)}/{len(splits)} 완료")
            time.sleep(15)  # 15초 대기
    else:
        print("No new urls to process")


def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    raw_content = content[0].page_content.strip()

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')

    return raw_content


@tool
def retrieve(query: str, top_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    # 1. 웹 검색으로 데이터 수집
    queries = [
        "HYBE JYP 경영 전략 비교 2026",
        "하이브 멀티레이블 JYP 본부제 비교",
        "위버스 버블 팬덤 플랫폼 비교",
        "HYBE JYP 글로벌 전략 비교",
    ]

    for query in queries:
        print(f"\n검색 중: {query}")
        results, json_path = web_search.invoke(query)
        print(f"저장된 파일: {json_path}")
        add_web_pages_json_to_chroma(json_path)

    # 2. 검색 테스트
    print("\n=== 검색 테스트 ===")
    retrieved_docs = retrieve.invoke({"query": "HYBE JYP 비교"})
    for doc in retrieved_docs:
        print(doc.page_content[:200])
        print('---')