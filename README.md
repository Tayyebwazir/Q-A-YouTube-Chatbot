# Q-A-YouTube-Chatbot
🧠 Purpose of the Project:
To analyze the content of a YouTube video by:

Extracting its transcript,

Splitting and embedding the text,

Storing it in a vector store,

Using a language model (LLM) to answer questions based on the video content.

🔧 Main Components and Steps:
✅ Step 1: Transcript Extraction
Extracts transcript using YouTubeTranscriptApi.

Tries English first, then falls back to other available languages.

If all fails, uses a sample hardcoded text as a fallback.

✅ Step 2: Text Splitting
Splits the transcript into manageable chunks using RecursiveCharacterTextSplitter.

This allows the embedding model to better process long text.

✅ Step 3: Embeddings + Vector Store
Converts text chunks into embeddings using HuggingFace’s all-MiniLM-L6-v2 model.

Stores those embeddings into FAISS, a vector store.

This makes it possible to search for relevant chunks based on a user's query.

✅ Step 4: Retrieval
Uses vector similarity to retrieve chunks relevant to the user’s question (e.g., “What is C++?”).

✅ Step 5: Prompt Construction & LLM Response
Constructs a prompt using the retrieved transcript data.

Sends the prompt to a Groq LLM (like llama-3.1-8b-instant) to get a generated answer.

The LLM answers only based on the context provided (RAG principle).

✅ Step 6: Chain Building
Builds a reusable chain using LangChain’s RunnableParallel and RunnablePassthrough tools.

