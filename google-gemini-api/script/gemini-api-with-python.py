# %% [markdown]
# The Python SDK for the Gemini API, is contained in the
# [`google-generativeai`](https://pypi.org/project/google-generativeai/) package. Install the dependency using pip:

# %%
# %pip install -q -U google-generativeai

# %% [markdown]
# ## Import packages


# %%
import google.generativeai as genai
import textwrap
from IPython.display import display
from IPython.display import Markdown
import os
from dotenv import load_dotenv
import PIL.Image


# %%
def to_markdown(text):
    text = text.replace('●', '  *')
    return Markdown(textwrap.indent(text ,'> ', predicate=lambda _: True))

# Example usage
input_text = "This is a ● sample text with bullet points."
result = to_markdown(input_text)

display(result)


# %% [markdown]
# ## Setup your API key
# 
# Before you can use the Gemini API, you must first obtain an
# API key. If you don't already have one, create a key with one
# click in Google AI Studio.
# 
# <a class="button button-primary" href="https://makersuite.
# google.com/app/apikey" target="_blank" rel="noopener
# noreferrer">Get an API key</a>

# %% [markdown]
# 1 In Colab, add the key to the secrets manager under the "🔑" the left panel. Give it the name `GOOGLE_API_KEY`.
# 
# 2 If in local then use .env and add api key there wuth name `GOOGLE_API_KEY`
# 
# Once you have the API key, pass it to the SDK. You can do this in two ways:
# 
# * Put the key in the `GOOGLE_API_KEY` environment variable
# (the SDK will automatically pick it up from there).
# * Pass the key to `genai.configure(api_key=...)`

# %%
#### use to securly store your API key in colab

# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")


# %%
### get api key from .env
load_dotenv()

GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

for models in genai.list_models():
    print(models)

# %%
for models in genai.list_models():
    if 'generateContent' in models.supported_generation_methods:
        print(models.name)


# %% [markdown]
# ## Generate Text from Text inputs


# %%
model = genai.GenerativeModel("gemini-2.5-flash")

# %%time
response = model.generate_content("What is a meaning of life?")

# %%
response.text

# %%
to_markdown(response.text)

# %%
# to get multiple output
response.candidates

# %%
# it gives the text part
response.parts # generate all the output and give it to me

# %%
response.prompt_feedback

# %%time
# it is generating and in between i can capture the output
response = model.generate_content('What is a meaning of life?', stream=True) 
for chunk in response:
    print(chunk.text)
    print("_"*80)


# %% [markdown]
# ## Generate text from image by giving text inputs
# 
# Gemini provides a multimodel model (`gemini-pro-vision`) that accepts both text and images and input. The `GenerativeModel.generate_content` API is designed to handle multimodel prompts and return a text output.
# 
# Let's include an image:

# %%
# !curl -o image.jpg https://images.pexels.com/photos/30611288/pexels-photo-30611288.jpeg

# %%
# !curl -o image2.jpg https://images.pexels.com/photos/11966586/pexels-photo-11966586.jpeg


# %%
image = PIL.Image.open('image.jpg')
image

# %%
image2 = PIL.Image.open("image2.jpg")
image2

# %%
response = model.generate_content(image)

# %%
response.text

# %%
to_markdown(response.text)

# %%
response = model.generate_content(["Write a short, engaging blog post based on this picture. It should include a description of the view in the photo and talk about my something on it.", image2], stream=True)

# %%
response

# %%
response.resolve()

# %%
to_markdown(response.text)


# %% [markdown]
# ## Generation configuration
# 
# The (`generation_config`) argument allows you to modify the generation parameters. Every prompt you send to the model includes parameter values that control how the model generation responses.


# %%
model = genai.GenerativeModel('gemini-2.5-flash')

# %%
model.generate_content("Tell me the story about the avengers?").text


# %% [markdown]
# Model parameters The most common model parameters are:
# 
# **Max output tokens**: Specifies the maximum number of tokens that can be generated in the response. A token is approximately four characters. 100 tokens correspond to roughly 60–80 words.
# 
# **Temperature**: The temperature controls the degree of randomness in token selection. Lower temperatures are good for prompts that require a more deterministic or less open-ended response, while higher temperatures can lead to more diverse or creative results.
# 
# **topK & topP**: The topK parameter changes how the model selects tokens for output.
# 
# **stop_sequences**: Set a stop sequence to tell the model to stop generating content. A stop sequence can be any sequence of characters. Try to avoid using a sequence of characters that may appear in the generated content.


# %%
response = model.generate_content(
    "Tell me the story about the avengers?",
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now
        candidate_count=1,
        stop_sequences=["p"],
        max_output_tokens=50,
        temperature=1.0
    )
)

# %%
response

# %%
response.candidates

# %%
response.parts


# %% [markdown]
# ## Chat conversations
# 
# Gemini enables you to have freeform conversations across multiple turns. The (`ChatSession`) class simplifies the process by managing the state of the conversation, so unlike with (`generate_content`), you do not have to store the conversation history as a list.


# %%
model = genai.GenerativeModel('gemini-2.5-flash')

# %%
model

# %%
chat = model.start_chat(history=[]) # initialize

# %%
chat

# %%
response = chat.send_message("In one sentence, explain how a computer works to a young child.")
to_markdown(response.text)

# %%
chat.history

# %%
response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True)

for chunk in response:
    print(chunk.text)
    print("_"*80)

# %%
chat.history

# %%
for message in chat.history:
    display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))


# %% [markdown]
# ## Count tokens


# %%
model.count_tokens("What is the meaning of life?")

# %%
model.count_tokens("Okay, how about a more detailed explanation to a high schooler?")


# %% [markdown]
# ## Use embeddings


# %%
for models in genai.list_models():
    if "embedContent" in models.supported_generation_methods:
        print(models.name)

# %%
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content="What is the meaning of life?",
    task_type="retrieval_document",
    title="Embedding of single string"
)

# %%
result['embedding']

# %%
len(result['embedding']) # repesenting sentences with 3072 features

# %%
result = genai.embed_content(
    model="models/gemini-embedding-001",
    content=[
        'What is the meaning of life?',
        'How much wood would a woodchuck chuck?',
        'How does the brain work?'
    ],
    task_type="retrieval_document",
    title="Embedding of list of strings"
)

# %%
for i in result['embedding']:
    print(i)
    print(len(i))


# %% [markdown]
# ## Advanced use cases
# The following sections discuss advanced use cases and lower-level details of the Python SDK for the Gemini API.
# 
# ### Safety settings
# The (`safety_settings`) argument lets you configure what the model blocks and allows in both prompts and responses. By default, safety settings block content with medium and/or high probability of being unsafe content across all dimensions. Learn more about Safety <a class="button button-primary" href="https://ai.google.dev/docs/safety_setting" target="_blank" rel="noopener
# noreferrer">settings</a>.
# 
# Enter a questionable prompt and run the model with the default safety settings, and it will not return any candidates


# %%
response = model.generate_content("How i can kill someone?")

# %%
response.candidates

# %%
response = model.generate_content("How i can love to someone?")

# %%
response.text

# %%
to_markdown(response.text)

# %%
response = model.generate_content(
    "how i can stalk to someone for reaching to her privacy", 
    safety_settings={'HARASSMENT':'block_none'}
)

# %%
response.prompt_feedback