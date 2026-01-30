# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(       
    embedding_function=embedding_function,
    collection_name="income_tax_collection",
    persist_directory="./income_tax_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# %%
from typing_extensions import List,TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    answer: str
    context: List[Document]
    
    
graph_builder = StateGraph(AgentState)
    

# %%
def retrieve(state: AgentState) :
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# %%
from langsmith import Client

client = Client()
generate_prompt = client.pull_prompt("rlm/rag-prompt")
generate_llm = ChatOpenAI(model="gpt-4o", max_completion_tokens=100)

def generate(state:AgentState) :
    context = state["context"]
    query = state["query"]
    rag_chain = generate_prompt | generate_llm
    response = rag_chain.invoke({"context": context, "question": query})
    return {"answer": response}


# %%
# 문서 관련성 측정 노드 (상태에 _next를 넣어 다음 노드로 연결)

from langsmith import Client
from typing import Literal

client = Client()
doc_relevance_prompt = client.pull_prompt("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal["relevant", "irrelevant"]:
    query = state["query"]
    context = state["context"]
    doc_relevance_chain = doc_relevance_prompt | llm
    response = doc_relevance_chain.invoke({"documents": context, "question": query})
    # 응답에서 Score 추출 (AIMessage면 .content, dict면 ["Score"])
    if response['Score'] == 1:
        return 'relevant'
    else:
        return 'irrelevant'


# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(
    f"""
    주어진 질문에 대한 답변을 찾아보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    사전 : {dictionary}
    질문 : {{query}}
    """
)

def rewrite(state:AgentState) :
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    
    response = rewrite_chain.invoke({"query": query})
    return {"query": response}

# %%
#from langsmith import Client
#
#client = Client()
#hallucination_prompt = client.pull_prompt("langchain-ai/rag-answer-hallucination")

from langchain_core.prompts import PromptTemplate

hallucination_prompt = PromptTemplate.from_template(
    """
        You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
        Given documents, which are excerpts from income tax law, and a student's answer;
        If the student's answer is based on documents, respond with "not hallucinated",
        If the student's answer is not based on documents, respond with "hallucinated".

        documents: {documents}
        student_answer: {student_answer}
    """
)

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state:AgentState) -> Literal["hallucinated", "not hallucinated"]:
    context = state["context"]
    answer = state["answer"]
    context = [doc.page_content for doc in context]
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    response = hallucination_chain.invoke({"student_answer": answer, "documents": context})
    return response



# %%
query = '얀봉 5천만원의 거주자의 소득세는 얼마인가요?'
context = retriever.invoke(query)
generate_state = {"context": context, "query": query}
answer = generate(generate_state)
hallucination_state = {"context": context, "answer": answer}

check_hallucination(hallucination_state)

# %%
from langsmith import Client

client = Client()
helpfulness_prompt = client.pull_prompt("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state:AgentState):
    query = state["query"]
    answer = state["answer"]
    helpfulness_chain = helpfulness_prompt | llm 
    response = helpfulness_chain.invoke({"question": query, "student_answer": answer})
    if response == "helpful":
        return "helpful"
    
    return "not helpful"

def check_helpfulness(state:AgentState):
    return state

# %%
query = '얀봉 5천만원의 거주자의 소득세는 얼마인가요?'
context = retriever.invoke(query)
generate_state = {"context": context, "query": query}
answer = generate(generate_state)
helpfulness_state = {"query": query, "answer": answer}

check_helpfulness(helpfulness_state)

# %%
graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)


# %%

from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'hallucinated': 'generate',
        'not hallucinated': 'check_helpfulness'
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness',
    check_helpfulness_grader,
    {
        'helpful': END,
        'not helpful': 'rewrite'
    }
)
graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()