# rag_demo
A demonstration of a RAG system
---

This is a simple demonstration of a RAG system that will answer questions based on the content of some pdf's.

The process has been broken out into multiple functions which are located in helpers.py.  There are functions for ingesting the pdf content, intializing a vector store and an llm interface, ingesting embeddings from the pdfs, preparing a query for the llm, and sending the query to the llm.

There is also an orchestrator.py file which runs each function and submits the specified list of questions.

Included in the main directory is a sample of the Questions and Answers resulting from executing the orchestrator.py file.  To reproduce these answers, a Google Gemini API key is required - place a .env file containing the line 'GOOGLE_API_KEY=your_api_key' in the main directory.