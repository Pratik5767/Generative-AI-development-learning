from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import os
from dotenv import load_dotenv

load_dotenv()


def main(url: str) -> None:
    document = SimpleDirectoryReader(url).load_data()
    index = VectorStoreIndex.from_documents(documents=document)
    query_engine = index.as_query_engine()
    response = query_engine.query("What is the article saying please summarize it?")
    print(response)


if __name__=='__main__':
    main(url="D:\Documents\Others\pdfandtxts")