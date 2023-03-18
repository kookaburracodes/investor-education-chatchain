import os

from langchain.chains import ChatVectorDBChain
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import FAISS

model_options = {'all-mpnet-base-v2': "sentence-transformers/all-mpnet-base-v2",
                'instructor-base': "hkunlp/instructor-base"}

model_options_list = list(model_options.keys())

def load_vectorstore(model):
    '''load embeddings and vectorstore'''

    if 'mpnet' in model:
        
        emb = HuggingFaceEmbeddings(model_name=model)
        return FAISS.load_local('vanguard_embeddings', emb)

    elif 'instructor'in model:
        
        emb = HuggingFaceInstructEmbeddings(model_name=model,
                                               query_instruction='Represent the Financial question for retrieving supporting paragraphs: ',
                                               embed_instruction='Represent the Financial paragraph for retrieval: ')
        return FAISS.load_local('vanguard_embeddings_inst', emb)

#default embeddings
vectorstore = load_vectorstore(model_options['all-mpnet-base-v2'])

def on_value_change(change):
    '''When radio changes, change the embeddings'''
    global vectorstore
    vectorstore = load_vectorstore(model_options[change])
    
# vectorstore = load_vectorstore('vanguard-embeddings',sbert_emb)
    
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about investing and the investment management industry.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about investing and the investment management industry.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about investing, politely inform them that you are tuned to only answer questions about investing and the investment management industry.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])


def get_chain(vectorstore):
    llm = OpenAI(temperature=0)
    qa_chain = ChatVectorDBChain.from_llm(
        llm,
        vectorstore,
        qa_prompt=QA_PROMPT,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    return qa_chain

def get_llm():
    return get_chain(vectorstore)