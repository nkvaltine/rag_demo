#A collection of functions required for the various parts of a RAG
#%%
import os
import re

from pypdf import PdfReader
import chromadb
import google.generativeai as genai

#%%
#open a pdf and mine it for text.  
#also break the text into segments, to help with embedding specificity
def _process_one_pdf(filename):
    reader = PdfReader(filename)
    pdf_texts = [page.extract_text().strip() for page in reader.pages]

    #segment the text into sentences
    expression = '(?<=[.!?])\s+'
    text_segments = []
    for text in pdf_texts:
        #ahh, there's some weird pdf stuff about newlines
        sentences = re.sub('\\n', ' ', text)
        sentences = re.split(expression, sentences)
        text_segments.extend(sentences)

    # Filter the empty strings
    text_segments = [text for text in text_segments if text]

    return text_segments


#process all of the pdfs...location hardcoded for convenience
def process_all_pdfs():
    data_path = "../data"
    pdfs = os.listdir(data_path)

    #filter out non-pdfs (.DS_Store 🙄)
    pdfs = [f for f in pdfs if f.endswith('.pdf')]

    texts = []
    for file in pdfs:
        texts.extend( _process_one_pdf(os.path.join(data_path, file)) )
    
    return texts


#initialize the vector database
def start_vectordb():
    chroma_client = chromadb.Client()
    
    #a choice made here, to make one collection of both docs
    collection = chroma_client.create_collection(name="all_docs")

    return collection

#take the text strings, embed them, and load the embeddings into the database
def load_embeddings(vectordb, text_segments):
    #chromadb does the embeddings automatically, apparently
    vectordb.add(
        documents=text_segments,
        #these ids could be more useful...
        ids = [str(i) for i in range(len(text_segments))]
    )
    return True


#prepare the context by embedding the question,
#  selecting the best matches from the database,
#  and compiling the actual text in a sensible way for the llm
def _prepare_context(question, vectordb):
    results = vectordb.query(
        query_texts=[question], # Chroma will embed this for you
        n_results=10
    )
    #unusual list of lists format in the documents output
    context = "Use the following as context:\n" + '\n'.join(results["documents"][0])
    return context

#prepare the query to send to llm, including the context and preprompts
def prepare_query(question, vectordb):
    context = _prepare_context(question, vectordb)
    preamble = "---\nAnswer the following question to the best of your abilities, using the above provided context.  If the answer doesn't appear in the context, politely decline to answer.  Question:\n"
    query = "\n".join([context, preamble, question])
    return query


#initialize the llm
def configure_llm(API_KEY):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    return model

#send the prepared query to the LLM
def send_query(query, model):
    response = model.generate_content(query)

    return response.text
# %%
