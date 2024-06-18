#%%
#A script to set up the RAG and run it against the specified questions
import helpers



# %%
texts = helpers.process_all_pdfs()

# %%
vectordb = helpers.start_vectordb()
helpers.load_embeddings(vectordb, texts)

# %%

question = "Which two companies created the R.31 reconnaissance aircraft?"
query = helpers.prepare_query(question, vectordb)
# %%
