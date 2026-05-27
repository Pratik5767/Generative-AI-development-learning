# %%
# Fetching APIS from the env 
import os
from dotenv import load_dotenv
load_dotenv()

WEAVIATE_API_KEY=os.environ['WEAVIATE_API_KEY']
WEAVIATE_CLUSTER=os.environ['WEAVIATE_CLUSTER']



# %%
import weaviate
from weaviate.classes.init import Auth

WEAVIATE_URL = WEAVIATE_CLUSTER
WEAVIATE_API_KEY = WEAVIATE_API_KEY

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)
print(f'connected to weaviate: {client.is_ready()}')



# %%
# fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"



# %%
# specify embedding model (using huggingface sentence transformer)
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)



# %%
# ## You can load multiple types of pdf using the langchain just check with the document
# # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/



# %%
# extract data and load
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "D:\\Pratik\\AI\\Gen_AI\\RAG_app_using_Langchain_MistralAI_Weviatedb\\data\\rag_research_paper.pdf", 
    extract_images=True
)

pages = loader.load()
pages



# %%
# Split the text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

docs = text_splitter.split_documents(pages)
docs



# %%
# metadata cleaning
import re

def clean_metadata(metadata: dict) -> dict:
    cleaned = {}
    for key, value in metadata.items():
        # Replace dots and invalid chars with underscore
        new_key = re.sub(r'[^_A-Za-z0-9]', '_', key)
        # If starts with digit, prefix with underscore
        if new_key and new_key[0].isdigit():
            new_key = '_' + new_key
        if new_key:
            cleaned[new_key] = value
    return cleaned

# ✅ Clean BEFORE inserting
for doc in docs:
    doc.metadata = clean_metadata(doc.metadata)



# %%
# embedding and storing in db
from langchain_weaviate import WeaviateVectorStore

vector_db = WeaviateVectorStore.from_documents(
    docs, embeddings, client=client, by_text=False
)



# %%
print(
    vector_db.similarity_search(
        "What is RAG?",
        k=3
    )[0].page_content
)

# %%
print(
    vector_db.similarity_search(
        "What is RAG?",
        k=3
    )[1].page_content
)

# %%
print(
    vector_db.similarity_search(
        "What is RAG?",
        k=3
    )[2].page_content
)

# %%
print(
    vector_db.similarity_search(
        "What is attention?", 
        k=3
    )
)



# %%
from langchain_core.prompts import ChatPromptTemplate

template="""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use ten sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

prompt=ChatPromptTemplate.from_template(template)
prompt



# %%
from langchain_huggingface import HuggingFaceEndpoint
load_dotenv()

HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

model = HuggingFaceEndpoint(
    huggingfacehub_api_token=HUGGING_FACE_API_TOKEN,
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_new_tokens=512
)



# %%
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
retriver = vector_db.as_retriever()

rag_chain = (
    {"context": retriver, "question": RunnablePassthrough()}
    | prompt
    | model
    | output_parser
)

# %%
print(rag_chain.invoke("What is rag application?"))

# %%
print(rag_chain.invoke("How does the RAG model differ from traditional language generation models?"))