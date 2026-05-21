# %%
# Fetching Google API Key
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)




# %%
# Data Ingestion
from langchain_community.document_loaders import TextLoader
import requests

url = "https://raw.githubusercontent.com/hwchase17/chroma-langchain/refs/heads/master/state_of_the_union.txt"
response = requests.get(url)
raw_data = response.text

with open("state_of_union.txt", "w") as f:
    f.write(raw_data)

loader = TextLoader('D:\\Pratik\\AI\\Gen_AI\\RAG_App_using_Langchain_Gemini_FAISS\\script\\state_of_union.txt')
document = loader.load()
print(document[0].page_content)




# %%
# Data  Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunk = text_splitter.split_documents(document)

print(text_chunk[0].page_content)
print(text_chunk[1].page_content)
print(text_chunk[2].page_content)




# %%
# Data Embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="gemini-embedding-001")
vectore_store = FAISS.from_documents(text_chunk, embeddings)




# %%
# Data Generation
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

retriver = vectore_store.as_retriever()
output_parser = StrOutputParser()

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
llm_model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash")

rag_chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | llm_model
    | output_parser
)


# %%
rag_chain.invoke("How is the United States supporting Ukraine economically and militarily?")

# %%
rag_chain.invoke("What action is the U.S. taking to address rising gas prices?")
