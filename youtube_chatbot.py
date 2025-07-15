import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re

# Set page config
st.set_page_config(
    page_title="YouTube Q&A",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    /* Dark background for entire app */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 40px;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #FFFFFF;
        margin-bottom: 20px;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1A1A1A;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px;
        font-size: 14px;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4A90E2;
        box-shadow: 0 0 0 1px #4A90E2;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4A90E2, #357ABD);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #357ABD, #4A90E2);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    }
    
    /* Video container */
    .video-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    
    /* Answer container */
    .answer-container {
        background: linear-gradient(135deg, #1A1A1A 0%, #2D2D2D 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        border-left: 4px solid #4A90E2;
    }
    
    .answer-text {
        color: #FFFFFF;
        line-height: 1.6;
        font-size: 15px;
    }
    
    /* Status messages */
    .success-msg {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid #4CAF50;
        color: #4CAF50;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .error-msg {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid #F44336;
        color: #F44336;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .info-msg {
        background: rgba(74, 144, 226, 0.1);
        border: 1px solid #4A90E2;
        color: #4A90E2;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    /* Quick video buttons */
    .quick-btn {
        background: linear-gradient(45deg, #333, #555);
        color: white;
        border: 1px solid #666;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 13px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .quick-btn:hover {
        background: linear-gradient(45deg, #555, #777);
        border-color: #4A90E2;
    }
    
    /* Spinner customization */
    .stSpinner {
        color: #4A90E2;
    }
</style>
""", unsafe_allow_html=True)

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'(?:youtube\.com\/v\/)([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url

def get_youtube_transcript(video_id):
    """Get YouTube transcript with error handling"""
    try:
        transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except NoTranscriptFound:
            try:
                transcript_list = transcript_list_obj.find_transcript(['en-US', 'en-GB', 'en-CA']).fetch()
            except:
                transcript_list = list(transcript_list_obj)[0].fetch()

        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
        
    except Exception as e:
        return None, str(e)

@st.cache_resource
def setup_rag_chain(transcript, groq_api_key):
    """Set up the RAG chain with caching"""
    try:
        if not groq_api_key or not groq_api_key.startswith('gsk_'):
            raise ValueError("Invalid API key format")
        
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        try:
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
            test_response = llm.invoke("Hello")
        except Exception as api_error:
            if "invalid_api_key" in str(api_error):
                raise ValueError("Invalid API key")
            else:
                raise ValueError(f"API error: {str(api_error)}")
        
        prompt = PromptTemplate(
            template="""
            Answer ONLY from the provided transcript context.
            If the context is insufficient, say you don't know.
            Keep answers clear and concise.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )
        
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)
        
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser
        
        return main_chain, len(chunks)
        
    except Exception as e:
        return None, 0

def main():
    # Header
    st.markdown('<div class="main-title">🎥 YouTube Q&A Chatbot</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = ""
    if 'video_loaded' not in st.session_state:
        st.session_state.video_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Video Settings</div>', unsafe_allow_html=True)
        
        video_url = st.text_input(
            "YouTube Video URL:",
            placeholder="Enter URL of video...",
            key="video_input"
        )
        
        if st.button("Load Video", key="load_video"):
            if video_url:
                st.session_state.video_url = video_url
                st.session_state.video_loaded = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # API Key
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">API Key</div>', unsafe_allow_html=True)
        
        groq_api_key = st.text_input(
            "Groq API Key:",
            type="password",
            placeholder="gsk_...",
        )
        
        if groq_api_key:
            if groq_api_key.startswith('gsk_'):
                st.markdown('<div class="success-msg">✅ Valid key format</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-msg">❌ Invalid key format</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick options
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">Quick Start</div>', unsafe_allow_html=True)
        
        videos = [
            ("C++ Tutorial", "https://www.youtube.com/watch?v=ZzaPdXTrSb8"),
            ("Neural Networks", "https://www.youtube.com/watch?v=aircAruvnKk"),
            ("Python Basics", "https://www.youtube.com/watch?v=kqtD5dpn9C8")
        ]
        
        for title, url in videos:
            if st.button(title, key=f"quick_{title}"):
                st.session_state.video_url = url
                st.session_state.video_loaded = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Video Player
    with col1:
        st.markdown('<div class="section-header">Video Player</div>', unsafe_allow_html=True)
        
        if st.session_state.get('video_loaded') and st.session_state.get('video_url'):
            video_id = extract_video_id(st.session_state.video_url)
            
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            st.video(f"https://www.youtube.com/watch?v={video_id}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            if groq_api_key and groq_api_key.startswith('gsk_'):
                if ('transcript_processed' not in st.session_state or 
                    st.session_state.get('current_video_id') != video_id):
                    
                    with st.spinner("Processing transcript..."):
                        transcript, error = get_youtube_transcript(video_id)
                        
                        if transcript:
                            chain, num_chunks = setup_rag_chain(transcript, groq_api_key)
                            if chain:
                                st.session_state.rag_chain = chain
                                st.session_state.transcript_processed = True
                                st.session_state.current_video_id = video_id
                                st.markdown(f'<div class="success-msg">✅ Ready for questions ({num_chunks} chunks)</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="error-msg">❌ Processing failed</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-msg">❌ Transcript error: {error}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-msg">ℹ️ Add API key to enable Q&A</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-msg">👈 Load a video from sidebar</div>', unsafe_allow_html=True)
    
    # Questions
    with col2:
        st.markdown('<div class="section-header">Ask Questions</div>', unsafe_allow_html=True)
        
        if (st.session_state.get('transcript_processed') and 
            hasattr(st.session_state, 'rag_chain') and 
            st.session_state.rag_chain):
            
            question = st.text_input(
                "Your question:",
                placeholder="What is this video about?",
                key="question"
            )
            
            if st.button("Get Answer", key="get_answer", type="primary"):
                if question.strip():
                    with st.spinner("Generating answer..."):
                        try:
                            answer = st.session_state.rag_chain.invoke(question)
                            st.session_state.current_answer = answer
                        except Exception as e:
                            st.session_state.current_answer = f"Error: {str(e)}"
            
            if st.session_state.current_answer:
                st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-text">{st.session_state.current_answer}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.markdown('<div class="info-msg">📝 Load video and add API key to start</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()