#!/usr/bin/env python
# coding: utf-8
"""Medical LLM ROUTER - Health Query System"""

import os
import asyncio
import tempfile
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from llama_index.core.workflow import (
    step,
    Event,
    Context,
    StartEvent,
    StopEvent,
    Workflow,
)

# Pick a cache folder inside your app that you control
cache_dir = Path(os.getcwd()) / "tiktoken_cache"
cache_dir.mkdir(exist_ok=True)

# Tell tiktoken to use it (must come before any tiktoken import)
os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("MedicalAssistant")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool
from typing import Union, List, Dict

# Configure page
st.set_page_config(
    page_title="Medical LLM ROUTER",
    page_icon="🩺",
    layout="wide"
)

# Initialize session state
if "progress_updates" not in st.session_state:
    st.session_state.progress_updates = []
if "response" not in st.session_state:
    st.session_state.response = ""
if "processing" not in st.session_state:
    st.session_state.processing = False
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = None
if "storage_dir" not in st.session_state:
    st.session_state.storage_dir = None
if "query_classification" not in st.session_state:
    st.session_state.query_classification = None

# Configure models via .env
def configure_models():
    # Try to use NVIDIA model if available, otherwise fallback to OpenAI
    try:
        # Settings.llm = NVIDIA(model=os.getenv("LLAMA_MODEL", "meta/llama-3.3-70b-instruct"))
        Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    except:
        logger.info("Falling back to OpenAI model")
        Settings.llm = OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    
    # Set embedding model
    Settings.embed_model = NVIDIAEmbedding(
        model=os.getenv("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2"),
        truncate="END"
    )

# Generate a unique hash for a file to track vector store
def get_file_hash(file_path):
    """Generate a unique hash for a file based on its path and modification time"""
    file_stat = os.stat(file_path)
    file_info = f"{file_path}_{file_stat.st_size}_{file_stat.st_mtime}"
    return hashlib.md5(file_info.encode()).hexdigest()

# Vector store registry to track what documents have been indexed
class VectorStoreRegistry:
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.join(tempfile.gettempdir(), "vector_store_registry")
        os.makedirs(self.base_dir, exist_ok=True)
        self.registry_file = os.path.join(self.base_dir, "registry.json")
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Creating new one.")
                return {}
        return {}
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_document(self, file_path, file_hash, storage_dir):
        """Register a document with its vector store location"""
        self.registry[file_hash] = {
            "file_path": file_path,
            "storage_dir": storage_dir,
            "created_at": datetime.now().isoformat()
        }
        self._save_registry()
    
    def get_storage_dir(self, file_hash):
        """Get the storage directory for a document hash if it exists"""
        if file_hash in self.registry:
            storage_dir = self.registry[file_hash]["storage_dir"]
            if os.path.exists(storage_dir):
                return storage_dir
        return None

# Initialize vector store registry
vector_store_registry = VectorStoreRegistry()

# Load or build index
def get_index(file_path, storage_dir):
    """Get index for document - reuses existing vector store if available"""
    # Generate a hash for this file
    file_hash = get_file_hash(file_path)
    
    # Check if we already have a vector store for this document
    existing_storage_dir = vector_store_registry.get_storage_dir(file_hash)
    
    if existing_storage_dir:
        logger.info(f"✅ Using existing vector store for document {os.path.basename(file_path)}")
        logger.info(f"🔍 RAG: Loading existing vector index from {existing_storage_dir}")
        # Make it more prominent that we're reusing an existing vector store
        st.success(f"✅ Reusing existing vector index for {os.path.basename(file_path)} - No need to rebuild!")
        st.session_state.progress_updates.append({
            "progress": f"Reusing existing vector store for {os.path.basename(file_path)} - Faster response!",
            "step": "Vector Store"
        })
        
        try:
            ctx = StorageContext.from_defaults(persist_dir=existing_storage_dir)
            index = load_index_from_storage(ctx)
            logger.info(f"🔍 RAG: Vector index loaded successfully with {len(index.docstore.docs)} documents")
            return index
        except Exception as e:
            logger.warning(f"Failed to load existing index, creating new one: {e}")
    
    # Check if the storage directory exists AND contains necessary index files
    docstore_path = os.path.join(storage_dir, "docstore.json")
    
    if os.path.exists(storage_dir) and os.path.exists(docstore_path):
        try:
            logger.info(f"Loading index from {storage_dir}")
            logger.info(f"🔍 RAG: Loading vector index from {storage_dir}")
            # Inform that we're using a previously created index
            st.info(f"Using previously created vector index for {os.path.basename(file_path)}")
            st.session_state.progress_updates.append({
                "progress": f"Using existing vector index from {storage_dir}",
                "step": "Vector Store"
            })
            
            ctx = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(ctx)
            
            # Register this storage location for future use
            vector_store_registry.register_document(file_path, file_hash, storage_dir)
            logger.info(f"🔍 RAG: Vector index loaded successfully with {len(index.docstore.docs)} documents")
            
            return index
        except (FileNotFoundError, ValueError, Exception) as e:
            logger.warning(f"Could not load existing index, creating new one: {str(e)}")
            # Fall through to create new index
    
    # Create the storage directory if it doesn't exist
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create new index
    try:
        logger.info(f"📄 Creating new vector index for {os.path.basename(file_path)}")
        logger.info(f"🔍 RAG: Initializing new vector store (this is where embeddings are created)")
        st.info(f"Building new vector index for {os.path.basename(file_path)}. This may take a moment...")
        st.session_state.progress_updates.append({
            "progress": f"Creating new vector index for {os.path.basename(file_path)} - First time processing",
            "step": "Vector Store"
        })
        
        start_time = datetime.now()
        logger.info(f"🔍 RAG: Parsing document content with LlamaParse")
        docs = LlamaParse(result_type="markdown").load_data(file_path)
        
        loading_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Document parsed in {loading_time:.2f} seconds")
        logger.info(f"🔍 RAG: Document parsed into {len(docs)} chunks")
        
        # Log chunk details for transparency
        for i, doc in enumerate(docs[:3]):  # Show first 3 chunks only
            content_preview = doc.text[:100] + "..." if len(doc.text) > 100 else doc.text
            logger.info(f"🔍 RAG: Chunk {i+1} preview: {content_preview}")
        
        if len(docs) > 3:
            logger.info(f"🔍 RAG: ... and {len(docs)-3} more chunks")
        
        logger.info(f"🔍 RAG: Creating embeddings and vector store from {len(docs)} chunks")
        idx = VectorStoreIndex.from_documents(docs)
        
        nodes_count = len(idx.docstore.docs)
        logger.info(f"🔍 RAG: Vector store created with {nodes_count} nodes")
        
        logger.info(f"🔍 RAG: Persisting vector store to {storage_dir}")
        idx.storage_context.persist(persist_dir=storage_dir)
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Vector index created in {total_time:.2f} seconds")
        logger.info(f"🔍 RAG: Vector store ready for retrieval with {nodes_count} embeddings")
        
        # Register this storage location for future use
        vector_store_registry.register_document(file_path, file_hash, storage_dir)
        
        return idx
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise Exception(f"Error processing document: {str(e)}")

# Wrap query engine as tool
def make_document_tool(index, name, description):
    engine = index.as_query_engine(similarity_top_k=10)
    return QueryEngineTool.from_defaults(
        engine,
        name=name,
        description=description,
    )

# Query classifier to determine if a question is general or domain-specific
async def classify_query(query):
    """Classify a health query as general or domain-specific (with specialist area)"""
    logger.info("Classifying query as general health or domain-specific")
    st.session_state.progress_updates.append({"progress": "Classifying your query...", "step": "Classification"})
    
    classify_prompt = f"""
    Classify the following health query as either GENERAL or one of these SPECIALIST domains:
    - NEUROLOGY (brain, nervous system, headaches, strokes, seizures, etc.)
    - CARDIOLOGY (heart, blood pressure, circulation, chest pain, etc.)
    - ORTHOPEDICS (bones, joints, muscles, arthritis, fractures, etc.)
    
    Query: {query}
    
    Return only one of these answers with no explanation:
    - GENERAL
    - NEUROLOGY
    - CARDIOLOGY
    - ORTHOPEDICS
    """
    
    response = await Settings.llm.acomplete(classify_prompt)
    classification = str(response).strip().upper()
    
    valid_classes = ["GENERAL", "NEUROLOGY", "CARDIOLOGY", "ORTHOPEDICS"]
    if classification not in valid_classes:
        # If classification is unclear, default to GENERAL
        logger.warning(f"Unclear classification '{classification}', defaulting to GENERAL")
        classification = "GENERAL"
    
    logger.info(f"Query classified as: {classification}")
    st.session_state.progress_updates.append({"progress": f"Your query is classified as: {classification}", "step": "Classification"})
    st.session_state.query_classification = classification
    
    return classification

# Replace the workflow classes with specialized agent functions

# Base Medical Agent class for specialized medical domains
class MedicalAgent:
    """Base class for medical agents with specialized domain knowledge"""
    
    def __init__(self, classification="GENERAL"):
        self.classification = classification
        self.name = classification.title()
        self.description = "Medical specialist"
        
    async def retrieve_context(self, query, medical_history_tool):
        """Retrieve relevant medical history context"""
        logger.info(f"Retrieving medical context for {self.classification} query")
        st.session_state.progress_updates.append({
            "progress": "Searching medical history for relevant information...", 
            "step": "Retrieval"
        })
        
        try:
            # Query the medical history for relevant information
            history_query = f"Retrieve any information related to {query} or {self.classification.lower()} conditions"
            history_response = await medical_history_tool.query_engine.aquery(history_query)
            history_context = history_response.response
            
            # Log how many source nodes were used
            source_count = len(history_response.source_nodes) if hasattr(history_response, 'source_nodes') else 0
            logger.info(f"Retrieved {source_count} relevant chunks from medical history")
            
            has_relevant_history = source_count > 0 and len(history_context) > 20
            
            if has_relevant_history:
                st.session_state.progress_updates.append({
                    "progress": f"Found {source_count} relevant information points in your medical history", 
                    "step": "Context"
                })
                logger.info("Relevant medical history found")
            else:
                st.session_state.progress_updates.append({
                    "progress": "No specific relevant information found in your medical history", 
                    "step": "Context"
                })
                logger.info("No relevant medical history found")
                history_context = "No specific relevant information found in patient's medical history."
        except Exception as e:
            logger.warning(f"Error searching medical history: {str(e)}")
            history_context = "Could not access medical history information."
            st.session_state.progress_updates.append({
                "progress": "Unable to search medical history", 
                "step": "Error"
            })
        
        logger.info("Context retrieval complete")
        return history_context
    
    async def generate_response(self, query, medical_context):
        """Generate a response (to be implemented by specialized agents)"""
        raise NotImplementedError("This method should be implemented by subclasses")
    
    async def format_response(self, response):
        """Format the final response with consistent styling and disclaimer"""
        # Add a disclaimer
        disclaimer = """

**Medical Disclaimer**: The information provided is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
"""
        
        formatted_response = f"{response}\n{disclaimer}"
        
        logger.info("Response formatting complete")
        st.session_state.progress_updates.append({
            "progress": "Response generated successfully", 
            "step": "Complete"
        })
        
        return formatted_response
    
    async def process_query(self, query, medical_history_tool):
        """Process a query using this agent"""
        logger.info(f"Processing query with {self.name} agent: {query}")
        st.session_state.progress_updates.append({
            "progress": f"Processing with {self.name} specialist...", 
            "step": "Processing"
        })
        
        # Step 1: Retrieve relevant context from medical history
        medical_context = await self.retrieve_context(query, medical_history_tool)
        
        # Step 2: Generate specialized response
        response = await self.generate_response(query, medical_context)
        
        # Step 3: Format the response with disclaimer
        formatted_response = await self.format_response(response)
        
        logger.info(f"{self.name} agent processing complete")
        return formatted_response

# General Health Agent
class GeneralHealthAgent(MedicalAgent):
    """Agent for general health queries"""
    
    def __init__(self):
        super().__init__("GENERAL")
        self.name = "General Health"
        self.description = "General health information and wellness"
    
    async def generate_response(self, query, medical_context):
        """Generate response for general health query"""
        logger.info("Generating general health response")
        st.session_state.progress_updates.append({
            "progress": "Generating response from general health knowledge...", 
            "step": "Generation"
        })
        
        system_prompt = """
        You are a helpful general health assistant providing information about common health concerns.
        Provide clear, accurate health information. If the query requires specialist knowledge, 
        acknowledge that limitation and suggest consulting a healthcare professional.
        When medical history information is available, use it to provide personalized advice.
        Always include important disclaimers about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        Provide a clear, helpful response addressing the patient's query.
        When referencing the medical history, be specific about how it relates to the current question.
        Format your response clearly with markdown headings and bullet points as appropriate.
        """
        
        response = await Settings.llm.acomplete(final_prompt)
        
        logger.info("General health response generated")
        return str(response)

# Neurology Agent
class NeurologyAgent(MedicalAgent):
    """Agent for neurology specialist queries"""
    
    def __init__(self):
        super().__init__("NEUROLOGY")
        self.name = "Neurology"
        self.description = "Brain, nervous system, headaches, seizures"
    
    async def generate_response(self, query, medical_context):
        """Generate response for neurology query"""
        logger.info("Generating neurology specialist response")
        st.session_state.progress_updates.append({
            "progress": "Generating response from neurology knowledge...", 
            "step": "Generation"
        })
        
        system_prompt = """
        You are a specialized Neurology assistant with expertise in neurological conditions.
        Provide detailed information about brain, nervous system, headaches, strokes, seizures, 
        and other neurological topics, drawing on your specialized knowledge.
        When medical history information is available, use it to provide personalized advice 
        relevant to neurological conditions.
        Always include important disclaimers about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        Provide a clear, helpful response addressing the patient's query.
        When referencing the medical history, be specific about how it relates to the current question.
        Format your response clearly with markdown headings and bullet points as appropriate.
        Use your neurology expertise to provide specialized insights.
        """
        
        response = await Settings.llm.acomplete(final_prompt)
        
        logger.info("Neurology specialist response generated")
        return str(response)

# Cardiology Agent
class CardiologyAgent(MedicalAgent):
    """Agent for cardiology specialist queries"""
    
    def __init__(self):
        super().__init__("CARDIOLOGY")
        self.name = "Cardiology"
        self.description = "Heart health, blood pressure, circulation"
    
    async def generate_response(self, query, medical_context):
        """Generate response for cardiology query"""
        logger.info("Generating cardiology specialist response")
        st.session_state.progress_updates.append({
            "progress": "Generating response from cardiology knowledge...", 
            "step": "Generation"
        })
        
        system_prompt = """
        You are a specialized Cardiology assistant with expertise in cardiovascular conditions.
        Provide detailed information about heart health, blood pressure, circulation, chest pain, 
        and other cardiovascular topics, drawing on your specialized knowledge.
        When medical history information is available, use it to provide personalized advice 
        relevant to cardiovascular conditions.
        Always include important disclaimers about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        Provide a clear, helpful response addressing the patient's query.
        When referencing the medical history, be specific about how it relates to the current question.
        Format your response clearly with markdown headings and bullet points as appropriate.
        Use your cardiology expertise to provide specialized insights.
        """
        
        response = await Settings.llm.acomplete(final_prompt)
        
        logger.info("Cardiology specialist response generated")
        return str(response)

# Orthopedics Agent
class OrthopedicsAgent(MedicalAgent):
    """Agent for orthopedics specialist queries"""
    
    def __init__(self):
        super().__init__("ORTHOPEDICS")
        self.name = "Orthopedics"
        self.description = "Bones, joints, muscles, arthritis"
    
    async def generate_response(self, query, medical_context):
        """Generate response for orthopedics query"""
        logger.info("Generating orthopedics specialist response")
        st.session_state.progress_updates.append({
            "progress": "Generating response from orthopedics knowledge...", 
            "step": "Generation"
        })
        
        system_prompt = """
        You are a specialized Orthopedics assistant with expertise in musculoskeletal conditions.
        Provide detailed information about bones, joints, muscles, arthritis, fractures, 
        and other orthopedic topics, drawing on your specialized knowledge.
        When medical history information is available, use it to provide personalized advice 
        relevant to orthopedic conditions.
        Always include important disclaimers about seeking professional medical advice.
        """
        
        final_prompt = f"""
        {system_prompt}
        
        PATIENT QUERY: {query}
        
        RELEVANT MEDICAL HISTORY:
        {medical_context}
        
        Provide a clear, helpful response addressing the patient's query.
        When referencing the medical history, be specific about how it relates to the current question.
        Format your response clearly with markdown headings and bullet points as appropriate.
        Use your orthopedics expertise to provide specialized insights.
        """
        
        response = await Settings.llm.acomplete(final_prompt)
        
        logger.info("Orthopedics specialist response generated")
        return str(response)

# Process query using the appropriate specialized agent
async def process_with_agent(query, medical_history_tool, classification):
    """Process a medical query using the appropriate specialist agent"""
    logger.info(f"Processing {classification} query using agent: {query}")
    st.session_state.progress_updates.append({"progress": f"Processing your {classification.lower()} health query...", "step": "Processing"})
    
    # Select the appropriate agent based on classification
    agent_map = {
        "GENERAL": GeneralHealthAgent(),
        "NEUROLOGY": NeurologyAgent(),
        "CARDIOLOGY": CardiologyAgent(),
        "ORTHOPEDICS": OrthopedicsAgent()
    }
    
    agent = agent_map.get(classification, GeneralHealthAgent())
    
    # Process the query with the selected agent
    result = await agent.process_query(query, medical_history_tool)
    
    logger.info(f"{classification} agent processing completed")
    return result

async def process_document(file_path, query, storage_dir):
    """Process the medical history document and answer health query"""
    # Reset progress
    st.session_state.progress_updates = []
    st.session_state.response = ""
    st.session_state.query_classification = None
    
    # Add a log handler that writes to the UI
    log_container = st.empty()
    logs = []
    
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            logs.append(log_entry)
            # Only show the last 10 logs to keep it concise
            log_container.code("\n".join(logs[-10:]), language="bash")
    
    # Add the custom handler
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
    
    try:
        # Log start of processing
        logger.info(f"🚀 Starting medical query processing: {query}")
        logger.info(f"📄 Medical history document: {os.path.basename(file_path)}")
        
        # Configure the models
        configure_models()
        
        # Get index and create tool
        with st.status("Creating medical history index..."):
            try:
                index = get_index(file_path, storage_dir)
            except Exception as e:
                error_msg = f"Error creating document index: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.progress_updates.append({"progress": error_msg, "step": "Error"})
                st.session_state.processing = False
                return None
        
        logger.info("Creating medical history tool from document index")
        medical_history_tool = make_document_tool(
            index, 
            name="medical_history",
            description=f"Patient's medical history information"
        )
        
        # Step 1: Classify the query
        logger.info("Step 1: Classifying medical query")
        st.session_state.progress_updates.append({"progress": "Step 1: Classifying your medical query", "step": "Classification"})
        classification = await classify_query(query)
        
        # Step 2: Process query using specialized agent
        logger.info(f"Step 2: Processing {classification} query with specialized agent")
        st.session_state.progress_updates.append({"progress": f"Step 2: Processing your {classification.lower()} query with specialized agent", "step": "Processing"})
        
        # Use the specialized agent to process the query
        response = await process_with_agent(query, medical_history_tool, classification)
        
        logger.info("Medical query processing complete")
        st.session_state.response = response
        st.session_state.progress_updates.append({"progress": "Medical query answered successfully", "step": "Complete"})
        return response
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.exception("Error during medical query processing")
        st.error(error_msg)
        st.session_state.progress_updates.append({"progress": error_msg, "step": "Error"})
        return None
    finally:
        # Remove the handler
        logger.removeHandler(handler)
        st.session_state.processing = False

def handle_file_upload():
    """Handle file upload and save to temporary location"""
    if st.session_state.uploaded_file:
        # Create a temporary directory for the uploaded file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, st.session_state.uploaded_file.name)
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(st.session_state.uploaded_file.getbuffer())
        
        # Create a storage directory for the index
        storage_dir = os.path.join(temp_dir, "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Store the paths in session state
        st.session_state.uploaded_file_path = file_path
        st.session_state.storage_dir = storage_dir
        return True
    return False

def main():
    st.title("🩺 Medical LLM ROUTER")
    
    # Show a status indicator if processing
    if st.session_state.processing:
        st.warning("⏳ Processing your request... Please wait.", icon="⏳")
    
    st.markdown("""
    Upload your medical history document and ask a health-related question.
    The assistant will analyze your medical history and provide relevant information.
    """)
    
    # Add explanation of main steps
    with st.expander("How the system works"):
        st.markdown("""
        ### Main Steps of the Medical Query Process:
        
        1. 📋 **Upload Medical History**: Upload your medical history document (PDF)
        2. 🔍 **Query Classification**: Your health question is classified as general or specialist
        3. 📊 **Context Retrieval**: Relevant information is retrieved from your medical history
        4. 🩺 **Specialized Response**: A response is generated by the appropriate health agent
        
        ### Specialist Areas:
        - **General Health**: Common health concerns and general wellness
        - **Neurology**: Brain, nervous system, headaches, seizures, etc.
        - **Cardiology**: Heart, blood pressure, circulation, etc.
        - **Orthopedics**: Bones, joints, muscles, arthritis, etc.
        
        > **Note**: This system provides information only, not medical diagnosis or treatment. 
        > Always consult healthcare professionals for medical advice.
        """)
    
    # File uploader with better UI
    with st.container():
        st.subheader("1. Upload Medical History")
        uploaded_file = st.file_uploader(
            "Upload your medical history document", 
            type=["pdf"], 
            key="uploaded_file", 
            on_change=handle_file_upload,
            help="Upload a PDF document containing your medical history",
            disabled=st.session_state.processing
        )
    
    # Only show the rest if a file is uploaded
    if st.session_state.uploaded_file_path:
        st.success(f"✅ Medical history uploaded: {os.path.basename(st.session_state.uploaded_file_path)}")
        
        # Query input
        st.subheader("2. Ask Health Question")
        query = st.text_input(
            "What would you like to know about your health?",
            placeholder="E.g., 'What does my blood pressure history show?', 'Should I be concerned about my headaches?'",
            help="Enter a health-related question",
            disabled=st.session_state.processing
        )
        
        # Process button
        if st.button(
            "🩺 Get Medical Information", 
            disabled=st.session_state.processing,
            type="primary",
            use_container_width=True
        ):
            if query:
                st.session_state.processing = True
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Progress", "System Logs", "Medical Context"])
                
                with tab1:
                    # Create placeholder for progress
                    st.info("Processing your medical query...")
                    progress_placeholder = st.empty()
                
                with tab2:
                    # Placeholder for logs
                    st.info("System logs will appear here")
                    log_placeholder = st.empty()
                
                with tab3:
                    # Placeholder for medical context
                    st.info("Medical context details will appear here")
                    context_placeholder = st.empty()
                    context_placeholder.markdown("""
                    ### Medical Context Process
                    1. **Query Classification**: Determining if your question is general or specialist
                    2. **Medical History Analysis**: Searching your medical history for relevant information
                    3. **Context Retrieval**: Finding specific health information relevant to your query
                    4. **Response Generation**: Creating a personalized response based on your history and query
                    """)
                
                # Run the process
                loop = asyncio.get_event_loop()
                loop.run_until_complete(process_document(
                    st.session_state.uploaded_file_path,
                    query,
                    st.session_state.storage_dir
                ))
                
                # Force a rerun after processing is complete to update UI
                st.rerun()
            else:
                st.error("Please enter a health question")
        
        # Display progress
        if st.session_state.progress_updates:
            tab1, tab2, tab3 = st.tabs(["Progress", "System Logs", "Medical Context"])
            
            with tab1:
                st.subheader("3. Query Processing Progress")
                
                # Show main workflow steps
                st.markdown("""
                ### Medical Query Processing Steps:
                
                1. 🔍 **Classification** - Identify query type
                2. 📊 **Retrieval** - Find relevant medical history
                3. 🩺 **Generation** - Create specialized response
                4. ✅ **Completion** - Deliver medical information
                """)
                
                # Show visual progress tracker
                progress_steps = ["Classification", "Retrieval", "Generation", "Complete"]
                completed_steps = []
                for update in st.session_state.progress_updates:
                    if isinstance(update, dict) and update.get("step") in progress_steps and update.get("step") not in completed_steps:
                        completed_steps.append(update.get("step"))
                
                # Display progress bar
                if completed_steps:
                    progress_value = len(completed_steps) / len(progress_steps)
                    st.progress(progress_value)
                    
                    # Show current step or completion
                    if len(completed_steps) < len(progress_steps):
                        current_step_index = len(completed_steps)
                        if current_step_index < len(progress_steps):
                            current_step = progress_steps[current_step_index]
                            st.info(f"Current step: {current_step}")
                    else:
                        st.success("✅ Query processing complete")
                
                # Display detailed progress updates in an expander
                with st.expander("View detailed progress", expanded=True):
                    for update in st.session_state.progress_updates:
                        if isinstance(update, dict):
                            step = update.get("step", "")
                            progress = update.get("progress", "")
                            
                            if "Error" in step:
                                st.error(f"{progress}")
                            elif "Classification" in step:
                                st.info(f"🔍 {progress}")
                            elif "Retrieval" in step or "Context" in step:
                                st.info(f"📊 {progress}")
                            elif "Generation" in step:
                                st.info(f"🩺 {progress}")
                            elif "Complete" in step:
                                st.success(f"✅ {progress}")
                            else:
                                st.info(f"{progress}")
            
            with tab2:
                st.subheader("System Logs")
                st.info("These logs show the technical details of the process")
                
                # Create a container for the logs
                with st.container():
                    # Filter out log entries that contain sensitive data or are too verbose
                    log_entries = []
                    for update in st.session_state.progress_updates:
                        if isinstance(update, dict):
                            log_entries.append(f"{update.get('step', 'Info')}: {update.get('progress', '')}")
                    
                    # Display the logs in a code block
                    if log_entries:
                        st.code("\n".join(log_entries), language="bash")
            
            with tab3:
                st.subheader("Medical Context Information")
                
                # Display specialist area based on classification
                classification = st.session_state.query_classification or "General Health"
                specialist_descriptions = {
                    "GENERAL": "General health concerns and wellness topics",
                    "NEUROLOGY": "Brain, nervous system, headaches, seizures, etc.",
                    "CARDIOLOGY": "Heart health, blood pressure, circulation, etc.",
                    "ORTHOPEDICS": "Bones, joints, muscles, arthritis, etc."
                }
                
                specialist_icons = {
                    "GENERAL": "🏥",
                    "NEUROLOGY": "🧠",
                    "CARDIOLOGY": "❤️",
                    "ORTHOPEDICS": "🦴"
                }
                
                icon = specialist_icons.get(classification, "🩺")
                description = specialist_descriptions.get(classification, "Health information")
                
                st.markdown(f"""
                ### {icon} Specialist Area: {classification.title()}
                Your query was classified as a **{classification.title()}** question.
                
                **Specialization Focus**: {description}
                """)
                
                # Extract context-related updates
                context_updates = [update for update in st.session_state.progress_updates 
                                  if isinstance(update, dict) and 
                                  ("Context" in update.get("step", "") or 
                                   "Retrieval" in update.get("step", ""))]
                
                # Display context operations
                if context_updates:
                    st.markdown("### Medical Context Retrieved")
                    for update in context_updates:
                        step = update.get("step", "")
                        progress = update.get("progress", "")
                        
                        if "Context" in step:
                            st.info(f"📊 {progress}")
                        elif "Retrieval" in step:
                            st.info(f"🔍 {progress}")
                
                # Show visual explanation of medical context
                st.markdown("### How Medical Context Works")
                st.markdown(f"""
                1. **Question Classification**:
                   - Your health question was analyzed and classified as: **{classification}**
                   - Based on this classification, it was routed to the appropriate specialist agent
                
                2. **Medical History Retrieval**:
                   - Relevant sections of your medical history were searched
                   - Information specific to your {classification.lower()} query was extracted
                
                3. **Personalized Response**:
                   - The system combined your medical context with specialized {classification.lower()} knowledge
                   - A personalized response was generated based on your specific situation
                """)
        
        # Display result
        if st.session_state.response:
            st.subheader("4. Medical Information")
            st.markdown("---")
            
            # Create a styled container for the response
            with st.container():
                # Display classification info at the top
                classification = st.session_state.query_classification or "General Health"
                specialist_icons = {
                    "GENERAL": "🏥",
                    "NEUROLOGY": "🧠",
                    "CARDIOLOGY": "❤️",
                    "ORTHOPEDICS": "🦴"
                }
                icon = specialist_icons.get(classification, "🩺")
                
                st.info(f"{icon} This response was generated by the **{classification.title()}** specialist")
                
                # Display the actual response
                st.markdown(st.session_state.response)
            
            st.markdown("---")
            
            # Disclaimer and download button
            st.warning("""
            **Important Medical Disclaimer**: The information provided is for informational 
            purposes only and is not a substitute for professional medical advice, diagnosis, 
            or treatment. Always seek the advice of your physician or other qualified health 
            provider with any questions you may have regarding a medical condition.
            """)
            
            # Download button
            st.download_button(
                label="📄 Download Medical Information",
                data=st.session_state.response,
                file_name="medical_information.md",
                mime="text/markdown"
            )
    
    # Show instructions if no file uploaded
    else:
        st.info("👆 Please upload your medical history document to get started")

if __name__ == "__main__":
    main() 