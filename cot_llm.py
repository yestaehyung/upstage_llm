# cot_llm.py

# pip install googlesearch-python tavily-python

import os
import configparser
from openai import OpenAI
from pprint import pprint
# Chat prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
# Chat prompt
from langchain_core.prompts import ChatPromptTemplate
# 3. define chain
from langchain_core.output_parsers import StrOutputParser

from tavily import TavilyClient
# from googlesearch import search


class StormLLM(object):
    def __init__(self):
        print('create StormLLM()')
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def set_env_subject(self, system, ex_human, ex_ai):
        self.subject_system = system
        self.subject_ex_human = ex_human
        self.subject_ex_ai = ex_ai

    def set_env_survey(self, system, ex_human, ex_ai):
        self.survey_system = system
        self.survey_ex_human = ex_human
        self.survey_ex_ai = ex_ai

    def cot_step01_request_subject(self, prompt):
        print("======================= Step 01: 주제추출")
        llm = ChatUpstage()
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.subject_system),
                ("human", self.subject_ex_human),
                ("ai", self.subject_ex_ai),
                ("human", f'{prompt}에 대하여 TPO에 맞는 서브 주제를 7개 추천해 주세요.'),
            ]
        )
        chain = chat_prompt | llm | StrOutputParser()
        chat_result = chain.invoke({})
        # pprint(chat_result)
        return chat_result

    def cot_step04_request_survey(self, prompt):
        print("======================= Step 04: LLM추천")
        llm = ChatUpstage()
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.survey_system),
                # ("human", "음식의 종류알려줘"),
                # ("ai", "1:한식, 2:중식, 3:일식"),
                ("human", self.survey_ex_human),
                ("ai", self.survey_ex_ai),
                ("human", f'{prompt}에 대하여 TPO에 맞는 주제를 정하고 주제별로 요약해 주세요.'),
            ]
        )
        chain = chat_prompt | llm | StrOutputParser()
        chat_result = chain.invoke({})
        # pprint(chat_result)
        return chat_result

    def web_search(self, query, num_results=5):
        search_results = []
        search_results = self.tavily.search(query=query)
        # for result in search(query, num_results=num_results):
        #     search_results.append(result)
        return search_results

    def run(self, prompt):
        # setp1
        chat_subject = self.cot_step01_request_subject(prompt)
        pprint(f'주제 추출결과: {chat_subject}')
        # 주제를 List로 분리
        list_subject = chat_subject.splitlines()
        list_web_result = list()
        # 웹 검색
        for subject in list_subject:
            list_web_result.append(self.web_search(subject))

        pprint(list_web_result)

        # setp4
        chat_survey = self.cot_step04_request_survey(prompt)
        pprint(f'주제 주제별요약 : {chat_survey}')

        return chat_survey


def main():
    print(f'01_first.py')
    # # print(f'key:{UPSTAGE_API_KEY}')
    # client = OpenAI(api_key=UPSTAGE_API_KEY, base_url=URL)
    # # step_01(client)
    # # step_02(client)
    # # step_03()

    storm_llm = StormLLM()

    system = "너는 유명한 패션 코디네이터야."
    ex_human = "음식의 종류알려줘"
    ex_ai = "1:몸에좋은 음식, 2:맛있는 음식, 3:요즘 인기있는 음식"
    storm_llm.set_env_subject(system, ex_human, ex_ai)

    system = """
        너는 옷을 어떻게 입어야 할지 고민하는 사람에게 주어진 상황에 대해서 색상조합의 기준에 따라 여러 방안을 제시할 수 있는 의상 전문가야.
    성별, 나이, 체형, 상황, 계절과 같은 기준에 따라 각각 다양한 코디를 추천해야 해.
    
    
    주어진 하의 색상에 따른 상의 색상 조합 기준 예시는 다음과 같아:
    
연청 바지
- 화이트: A
- 레드, 핑크: C
- 오렌지, 옐로우: A
- 그린: A
- 블루: A
- 네이비: A
- 블랙, 그레이: A

진청 바지
- 화이트: A'
- 레드, 핑크: A'
- 오렌지, 옐로우: A
- 그린: A
- 블루: A
- 네이비: A
- 블랙, 그레이: B

베이지 바지
- 화이트: A
- 레드, 핑크: B
- 오렌지, 옐로우: A
- 그린: A
- 블루: A
- 네이비: A
- 블랙, 그레이: A

카키 바지
- 화이트: A
- 레드, 핑크: F
- 오렌지, 옐로우: C
- 그린: A
- 블루: A
- 네이비: F
- 블랙, 그레이: F

와인 바지
- 화이트: A
- 레드, 핑크: A
- 오렌지, 옐로우: A
- 그린: B
- 블루: A
- 네이비: A
- 블랙, 그레이: A'

블랙 바지
- 화이트: A
- 레드, 핑크: A'
- 오렌지, 옐로우: A
- 그린: A'
- 블루: A'
- 네이비: A'
- 블랙, 그레이: A'

정리된 기준
- A: 최강 조합 (누구나 모든 상황에서 추천)
- A': 잘 어울림 (둘 이상의 색상이 매우 대조적이지만)
- B: 나쁘지 않음 (나쁘지 않으며 분위기에 맞는 색상 매치)
- C: 한번 더 생각 (입기 전에 꼭 한번 입어봐야 하는 색상 매치)
- F: 외출금지 (어울리지 않는 색상 매치)
    """
    ex_human = "여름 저녁 7시에 페스티벌에 40대 여성이 입고가면 좋은 의상을 추천해줘"
    ex_ai = """
    ### 1. 체형에 따른 코디
    #### a. 체형: 슬림형
    1. **드레스:** 화이트 여름 드레스 (A+)
       - 이유: 슬림한 체형을 더욱 돋보이게 하는 화이트 드레스는 시원하고 세련된 느낌을 줍니다.
    2. **상의:** 블루 반팔 블라우스 (A)
       - **하의:** 진청 데님 팬츠
       - 이유: 슬림한 체형을 강조하는 블루 블라우스와 진청 데님 팬츠의 조합은 깔끔하고 스타일리시한 룩을 완성합니다.

    #### b. 체형: 중간형
    1. **드레스:** 핑크 플로럴 드레스 (A)
       - 이유: 플로럴 드레스는 중간형 체형에 부드러운 곡선을 더해주어 여성스러운 느낌을 줍니다.
    2. **상의:** 화이트 블라우스 (A+)
       - **하의:** 베이지 치노 팬츠
       - 이유: 화이트 블라우스와 베이지 치노 팬츠는 클래식하고 편안한 조합으로 중간형 체형에 잘 어울립니다.

    #### c. 체형: 플러스 사이즈
    1. **드레스:** 네이비 랩 드레스 (A)
       - 이유: 랩 드레스는 플러스 사이즈 체형을 아름답게 강조하며, 네이비 색상은 세련되고 차분한 느낌을 줍니다.
    2. **상의:** 블랙 린넨 블라우스 (A)
       - **하의:** 블랙 와이드 팬츠
       - 이유: 올 블랙 코디는 체형을 슬림하게 보이게 하며, 린넨 소재는 여름에 시원하게 입을 수 있습니다.

    ### 2. 피부톤에 따른 코디
    #### a. 피부톤: 밝은 톤
    1. **드레스:** 라이트 블루 썸머 드레스 (A+)
       - 이유: 라이트 블루 색상은 밝은 피부톤을 더욱 화사하게 만들어줍니다.
    2. **상의:** 화이트 반팔 티셔츠 (A+)
       - **하의:** 블루 데님 스커트
       - 이유: 화이트와 블루의 조합은 밝은 피부톤에 잘 어울리며, 시원하고 캐주얼한 느낌을 줍니다.

    #### b. 피부톤: 중간 톤
    1. **드레스:** 그린 플로럴 드레스 (A)
       - 이유: 그린 색상은 중간 톤 피부를 더욱 생기 있게 만들어줍니다.
    2. **상의:** 오렌지 린넨 셔츠 (A)
       - **하의:** 베이지 치노 팬츠
       - 이유: 오렌지와 베이지의 조합은 따뜻한 느낌을 주며, 중간 톤 피부에 잘 어울립니다.

    #### c. 피부톤: 어두운 톤
    1. **드레스:** 옐로우 맥시 드레스 (A)
       - 이유: 옐로우 색상은 어두운 피부톤을 더욱 빛나게 합니다.
    2. **상의:** 그린 반팔 티셔츠 (A)
       - **하의:** 블랙 슬림 팬츠
       - 이유: 그린과 블랙의 조합은 세련되면서도 강렬한 느낌을 주며, 어두운 피부톤에 잘 어울립니다.

    ### 3. 선호 스타일에 따른 코디
    #### a. 스타일: 캐주얼
    1. **상의:** 화이트 반팔 티셔츠 (A+)
       - **하의:** 진청 데님 팬츠
       - 이유: 캐주얼한 느낌을 주는 기본적인 화이트 티셔츠와 데님 팬츠의 조합은 편안하고 실용적입니다.
    2. **드레스:** 스트라이프 셔츠 드레스 (A)
       - 이유: 스트라이프 셔츠 드레스는 캐주얼하면서도 스타일리시한 느낌을 줍니다.

    #### b. 스타일: 세미 포멀
    1. **드레스:** 네이비 랩 드레스 (A)
       - 이유: 세미 포멀한 느낌을 주는 네이비 랩 드레스는 고급스럽고 우아한 분위기를 연출합니다.
    2. **상의:** 베이지 블라우스 (A)
       - **하의:** 화이트 슬랙스
       - 이유: 베이지 블라우스와 화이트 슬랙스의 조합은 세미 포멀한 자리에서도 우아하고 단정한 느낌을 줍니다.

    #### c. 스타일: 모던
    1. **드레스:** 블랙 슬립 드레스 (A+)
       - 이유: 블랙 슬립 드레스는 모던하고 세련된 느낌을 주며, 액세서리에 따라 다양한 분위기를 연출할 수 있습니다.
    2. **상의:** 그레이 반팔 티셔츠 (A)
       - **하의:** 블랙 와이드 팬츠
       - 이유: 그레이와 블랙의 모노톤 조합은 모던하고 시크한 룩을 완성합니다.
    """
    storm_llm.set_env_survey(system, ex_human, ex_ai)
    prompt = "결혼식에 입고갈 여성 의상을 추천해 주세요."

    result = storm_llm.run(prompt)
    pprint(f'result:{result}')

if __name__ == "__main__":
    main()
