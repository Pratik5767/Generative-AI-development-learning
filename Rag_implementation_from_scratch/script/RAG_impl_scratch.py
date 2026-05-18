# %%
from collections import Counter
import math



# %%
corpus_of_documents = [
    "Take a leisurely walk in the park and enjoy the fresh air.",
    "Visit a local museum and discover something new.",
    "Attend a live music concert and feel the rhythm.",
    "Go for a hike and admire the natural scenery.",
    "Have a picnic with friends and share some laughs.",
    "Explore a new cuisine by dining at an ethnic restaurant.",
    "Take a yoga class and stretch your body and mind.",
    "Join a local sports league and enjoy some friendly competition.",
    "Attend a workshop or lecture on a topic you're interested in.",
    "Visit an amusement park and ride the roller coasters."
]

corpus_of_documents



# %%
user_query = "is yoga good for health"
document = "Yoga is very good for living healthy life style"

# %%
def cosine_similarity(query, document):
    # Tokenize and convert to lower case
    query_tokens = query.lower().split(' ')
    document_tokens = document.lower().split(' ')

    # Create counters for query and document
    query_counter = Counter(query_tokens)
    document_counter = Counter(document_tokens)

    # Calculate the dot product
    dot_product = sum(query_counter[tokens] * document_counter[tokens] for tokens in query_counter.keys() & document_counter.keys())

    # Calculate magnitude
    query_magnitude = math.sqrt(sum(query_counter[tokens] ** 2 for tokens in query_counter))
    document_magnitude = math.sqrt(sum(document_counter[tokens] ** 2 for tokens in document_counter))

    # Calculate cosine similarity
    similarity = dot_product / (query_magnitude * document_magnitude) if query_magnitude * document_magnitude != 0 else 0
    return similarity

# %%
cosine_similarity(user_query, document)



# %%
def return_response(query, corpus):
    similarities = []
    for doc in corpus:
        similarity = cosine_similarity(query, doc)
        similarities.append(similarity)
    return corpus_of_documents[similarities.index(max(similarities))]

# %%
corpus_of_documents

# %%
query = "i like fresh air."
return_response(query, corpus_of_documents)

# %%
query2 = "i like to do yoga"
return_response(query2, corpus_of_documents)



# %%
# How can you configure LLM in your local systems
# LLAMA2
# Option 1: hugging face (we are not going to use this one)

# %%
# Augement the response by using llama2 model

# %%
user_input = "i like fresh air."
relevant_document = return_response(user_input, corpus_of_documents)

# %%
import json
import requests
full_response = []

# %%
full_response = []

prompt = """
You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information.
This is the recommended activity: {relevant_document}
The user input is: {user_input}
Compile a recommendation to the user based on the recommended activity and the user input.
"""

url = "http://localhost:11434/api/generate"

data = {
    "model":"llama3.2:1b",
    "prompt": prompt.format(user_input=user_input, relevant_document=relevant_document)
}

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)

try:
    for line in response.iter_lines():
        # filter out keep-alive new lines
        if line:
            decoded_line = json.loads(line.decode('utf-8'))
            #print(decoded_line['response']) 
            full_response.append(decoded_line['response'])
finally:
    response.close()

print(''.join(full_response))