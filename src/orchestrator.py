#%%
#A script to set up the RAG and run it against the specified questions
import os
from dotenv import load_dotenv

import helpers


# %%
#setup infrastructure
vectordb = helpers.start_vectordb()

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
if api_key is None:
    print("Failure to get API key!")
    raise Exception("API Key Not Found")

llm = helpers.configure_llm(api_key)

#%%
#process data
texts = helpers.process_all_pdfs()

# %%
helpers.load_embeddings(vectordb, texts)

# %%
#run questions
question_list = [
    "Which two companies created the R.31 reconnaissance aircraft?",
    "What guns were mounted on the Renard R.31?",
    "Who was the first softball player to represent any country at four World Series of Softball?",
    "Who were the pitchers on the Australian softball team's roster at the 2020 Summer Olympics?"
    ]

for question in question_list:
    query = helpers.prepare_query(question, vectordb)
    response = helpers.send_query(query, llm)

    print(f"Question: {question}\n\nResponse: {response}\n\n\n")


# %%

