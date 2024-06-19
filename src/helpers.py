#A collection of functions required for the various parts of a RAG
#%%
import os
import re

import pdfplumber
import chromadb
import google.generativeai as genai

#%%
DATA_PATH = "../data"
#a few constants to tune the results
NUM_RESULTS = 10
NUM_SENTENCES = 4


#open a pdf and mine it for text.  
#also break the text into segments, to help with embedding specificity
def _process_one_pdf(filename):
    print(f"Processing filename: {filename}")
    text_segments = []
    with pdfplumber.open(filename) as reader:

        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            
            pdf_text = page.extract_text().strip()
            #segment the text into sentences based on punctuation
            expression = '(?<=[.!?])\s+'
            sentences = re.split(expression, pdf_text)
            # filter out the empty strings
            sentences = [text for text in sentences if len(text) != 0]

            #but mash some back together to have bigger context chunks
            chunks = []
            for i in range(0, len(sentences), NUM_SENTENCES):
                group = '\n'.join(sentences[i:i + NUM_SENTENCES])
                chunks.append(group)
            text_segments.extend(chunks)
            #grab some tables too for good measure
            pdf_tables = page.extract_tables()
            table_segment = ""
            for table in pdf_tables:
                table_segment += "Table: \n"
                for row in table:
                    #do all rows come as lists of length 1? 
                    table_segment += str(row[0]) + "\n"
                text_segments.append(table_segment)
    
    return text_segments


#process all of the pdfs...location hardcoded for convenience
def process_all_pdfs():
    pdfs = os.listdir(DATA_PATH)

    #filter out non-pdfs (.DS_Store 🙄)
    pdfs = [f for f in pdfs if f.endswith('.pdf')]

    texts = []
    for file in pdfs:
        texts.extend( _process_one_pdf(os.path.join(DATA_PATH, file)) )
    
    return texts



#initialize the vector database
def start_vectordb():
    chroma_client = chromadb.Client()
    #a choice made here, to make one collection of both docs
    collection = chroma_client.create_collection(name="all_docs")

    return collection

#take the text strings, embed them, and load the embeddings into the database
def load_embeddings(vectordb, text_segments):
    #chromadb does the embeddings automatically
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
        n_results=NUM_RESULTS
    )
    #unusual list of lists format in the documents output
    context = "Use the following as context:\n" + '\n'.join(results["documents"][0])
    return context

#prepare the query to send to llm, including the context and preprompts
def prepare_query(question, vectordb):
    if len(question) == 0:
        print("No question provided")
        raise Exception("Empty Question")
    
    context = _prepare_context(question, vectordb)
    preamble = "\n---\n" + "Answer the following question to the best of your abilities, using the above provided context, without mentioning that you got the answer from the context.  If the answer doesn't appear in the context, politely decline to answer.  \nQuestion:\n"
    query = "\n".join([context, preamble, question])
    return query



#initialize the llm
def configure_llm(API_KEY):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    return model

#send the prepared query to the LLM
def send_query(query, model):
    #won't answer the question about the guns with default safety setting
    safety = [{
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
        }]
    response = model.generate_content(query, safety_settings=safety)

    return response.text
# %%
