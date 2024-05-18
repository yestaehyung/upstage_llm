import gradio as gr


from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

from dotenv import load_dotenv
import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings
from langchain.docstore.document import Document

load_dotenv()

chat_key = os.getenv('UPSTAGE_API_KEY')

llm = ChatUpstage(openai_api_key= chat_key, streaming=True)


df = pd.read_excel('./Cloth_3.xlsx')

# Initialize an empty list to store the sentences
sample_text_list = []

# Loop through each row in the dataframe
for index, row in df.iterrows():
    single_sentence = f"카테고리:{row['카테고리']} 상품명:{row['상품명']} 상품평:{row['상품평']}"
    # single_sentence = f"{row['카테고리']} {row['상품명']} {row['상품평']}"
    sample_text_list.append(single_sentence)


sample_docs = [Document(page_content=text) for text in sample_text_list]

vectorstore = Chroma.from_documents(
    documents=sample_docs,
    embedding=UpstageEmbeddings(model="solar-embedding-1-large"),
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever()

# More general chat
chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "너는 사용자의 정보를 받아 옷을 추천해 주는 에이전트야"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

chain = chat_with_history_prompt | llm | StrOutputParser()

def chat(message, history):
    history_langchain_format = [] # 사용자가 입력한 것과 ai 가 답한 것을 저장함
    for human, ai in history:
        print(human, ai)
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    generator = chain.stream({"message": message, "history": history_langchain_format})

    assistant = ""
    for gen in generator:
        print("temp", gen)
        assistant += gen
        yield assistant

html_content = """
<div class="header">
    <img class='logo' src="https://images.squarespace-cdn.com/content/v1/659384103b38c97cdaf368bd/3b855d2d-bbf9-47f3-a67a-bad0b4e15717/Logo_Black.png?format=1500w">
    <div class='text_container'>
        <div class='text'>Welcome to fashion curator</div>
        <div class='text'>TPO에 맞게 옷을 추천해 주는 챗봇입니다.</div>
    </div>    
</div>
<div class="header-2">
    <div class='text'>
    본 프로젝트에 대해 궁금하시면 <a href="https://docs.google.com/presentation/d/14MZq9M53qYI1Vc4mLsbibIABqcITS-XUJo7LeyHwC5g/edit?usp=sharing">링크</a> 를 눌러주세요
    </div>
</div>
"""

css_style = """
<style>
    h1{
        margin: 0; !important;
    }
    p{
        margin: 0; !important;
    }
    .header {
        display: flex;
        justify-content: space-around;
    }
    .header-2{
        margin-top: 10px;
    }
    .logo {
        width: 30%;
        height: auto;
        margin-right: 5%;
    }
    .text_container {
        width: 65%;
        height: 100%;
        text-align: center;
        color: black !important;
    }
    .text {
        font-size: 1.2rem;
        color: black !important;
        margin: 0;
    }
    .gradio-container {
        background-color: #B6CEFA;
    }
    
</style>"""# System prompt

system_prompt = """너는 상황/분위기/옷을 추천해 주는 에이전트야, 우선 성별/장소/스타일을 물어봐줘

답변을 할 때 검색되어 나온 결과를 참고해줘
검색 결과: {content}"""

def generate_response(gender: str, message: str, history: list) -> str:
    if gender:
        prompt = system_prompt + f"\n성별: {gender}"
    else:
        prompt = system_prompt
    
    result_docs = retriever.invoke(message)
    
    chat_with_history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt.format(content="result_docs")),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{message}"),
        ]
    )
    
    chain = chat_with_history_prompt | llm | StrOutputParser()

    # Format history for LangChain
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    if gender:
        solar_response = chain.stream({"message": message, "history": history_langchain_format, "gender": gender})
    else:
        solar_response = chain.stream({"message": message, "history": history_langchain_format})

    assistant = ""
    for gen in solar_response:
        assistant += gen
    
    return assistant

def chat(message: str, history: list) -> str:
    if '남자' in message:
        return generate_response('남자', message, history)
    elif '여자' in message:
        return generate_response('여자', message, history)
    else:
        return generate_response(None, message, history)
