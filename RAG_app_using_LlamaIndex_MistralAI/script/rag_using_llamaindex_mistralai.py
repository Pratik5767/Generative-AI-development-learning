# %%
# imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch


# %%
# extract and load
document = SimpleDirectoryReader("../data/").load_data()
print(document)

# %%
# setup prompt specific to stable LLM
system_prompt = """<|SYSTEM|># You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

# %%
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-huggingface/llama_index/llms/huggingface/base.py

# loading the model in the memory / hardware
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="mistralai/Mistral-7B-Instruct-v0.1",
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

# %%
# embedding model from huggingface
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# %%
# servcie context - simple data class, it is a utility class for initializing the variables.
service_context = ServiceContext.from_defaults(
    chunk_size=1042,
    llm=llm,
    embed_model=embed_model
)

# %%
# embedding and storing it into simple vector db
index = VectorStoreIndex.from_documents(document, service_context=service_context)

# %%
query_engine = index.as_query_engine()

# %%
query_engine.query("What is a attention?")

# %%
query_engine.query("How attention is different from rnn and lstm?")