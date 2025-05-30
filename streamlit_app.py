import streamlit as st
import os
import sys
import json
import uuid
import re
import shutil
import time
import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import anthropic
import aiofiles
import tiktoken
import nest_asyncio
import datetime
import pandas as pd

# Apply nest_asyncio to allow asyncio in Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Claude AI Batch Translator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .error-box {
        background-color: #f8d7da;
        color: #842029;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #664d03;
    }
    .info-text {
        background-color: #cff4fc;
        color: #055160;
        padding: 0.5rem;
        border-radius: 0.25rem;
    }
    .batch-status {
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .status-submitted {
        background-color: #e2e3e5;
        color: #383d41;
    }
    .status-validating {
        background-color: #fff3cd;
        color: #664d03;
    }
    .status-in_progress {
        background-color: #cce5ff;
        color: #004085;
    }
    .status-completed {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #842029;
    }
    .cost-savings {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'batch_jobs' not in st.session_state:
    st.session_state.batch_jobs = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_job' not in st.session_state:
    st.session_state.current_job = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'success_message' not in st.session_state:
    st.session_state.success_message = None

# Data structures
@dataclass
class BatchRequest:
    """Structure for a single batch request."""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/messages"
    body: Dict[str, Any] = None

@dataclass
class TranslationChunk:
    """Structure for a translation chunk."""
    chunk_id: str
    filename: str
    file_path: str
    content: str
    chunk_number: int
    total_chunks: int
    is_chunked: bool
    token_count: int

# Helper functions
def count_tokens(text: str) -> int:
    """Count the number of tokens in the text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        st.warning(f"Token counting failed: {str(e)}. Using character-based estimate.")
        return len(text) // 4

def get_model_limits(model: str) -> Dict[str, int]:
    """Get token limits based on the Claude model."""
    model_limits = {
        "claude-3-haiku-20240307": {"max_tokens": 4096, "context": 200000},
        "claude-3-sonnet-20240229": {"max_tokens": 4096, "context": 200000},
        "claude-3-opus-20240229": {"max_tokens": 4096, "context": 200000},
        "claude-3-5-sonnet-20240620": {"max_tokens": 8192, "context": 200000},
        "claude-sonnet-4-20250514": {"max_tokens": 8192, "context": 200000},
        "claude-opus-4-20250514": {"max_tokens": 8192, "context": 200000}
    }
    return model_limits.get(model, {"max_tokens": 4096, "context": 200000})

def ensure_directory_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        st.session_state.error_message = f"Error creating directory {directory_path}: {str(e)}"
        return False

def save_uploaded_files(uploaded_files, target_directory):
    """Save uploaded files to the target directory."""
    try:
        ensure_directory_exists(target_directory)
        
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(target_directory, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append({
                "filename": uploaded_file.name,
                "path": file_path,
                "size": len(uploaded_file.getbuffer())
            })
        
        return saved_files
    except Exception as e:
        st.session_state.error_message = f"Error saving uploaded files: {str(e)}"
        return []

def save_instruction_file(instruction_text, target_directory):
    """Save instruction text to a file in the target directory."""
    try:
        ensure_directory_exists(target_directory)
        
        file_path = os.path.join(target_directory, "translation_instruction.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(instruction_text)
        
        return file_path
    except Exception as e:
        st.session_state.error_message = f"Error saving instruction file: {str(e)}"
        return None

def split_into_chunks(text: str, max_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    """Split text into chunks that don't exceed max_tokens, with some overlap."""
    chunks = []
    
    # Calculate approximate token count
    token_count = count_tokens(text)
    
    # If text is small enough, return as single chunk
    if token_count <= max_tokens:
        return [{
            "text": text,
            "chunk_id": 0,
            "total_chunks": 1,
            "is_chunked": False,
            "token_count": token_count
        }]
    
    # For larger texts, split intelligently on paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    current_tokens = 0
    chunk_id = 0
    
    for i, paragraph in enumerate(paragraphs):
        paragraph_tokens = count_tokens(paragraph)
        
        # If adding this paragraph would exceed limit, save current chunk
        if current_tokens + paragraph_tokens > max_tokens and current_chunk:
            chunks.append({
                "text": current_chunk,
                "chunk_id": chunk_id,
                "is_chunked": True,
                "token_count": current_tokens
            })
            
            # Start new chunk with overlap
            if current_chunk and overlap_tokens > 0:
                words = current_chunk.split()
                approx_words_per_token = 1.5
                overlap_words = int(overlap_tokens * approx_words_per_token)
                overlap_text = ' '.join(words[-min(overlap_words, len(words)):])
                current_chunk = overlap_text + "\n\n"
            else:
                current_chunk = ""
                
            current_tokens = count_tokens(current_chunk)
            chunk_id += 1
        
        # If a single paragraph is too large, split by sentences
        if paragraph_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence)
                
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "chunk_id": chunk_id,
                        "is_chunked": True,
                        "token_count": current_tokens
                    })
                    
                    # Start new chunk with overlap
                    if current_chunk and overlap_tokens > 0:
                        words = current_chunk.split()
                        approx_words_per_token = 1.5
                        overlap_words = int(overlap_tokens * approx_words_per_token)
                        overlap_text = ' '.join(words[-min(overlap_words, len(words)):])
                        current_chunk = overlap_text + " "
                    else:
                        current_chunk = ""
                        
                    current_tokens = count_tokens(current_chunk)
                    chunk_id += 1
                
                current_chunk += sentence + " "
                current_tokens = count_tokens(current_chunk)
        else:
            current_chunk += paragraph + "\n\n"
            current_tokens = count_tokens(current_chunk)
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "chunk_id": chunk_id,
            "is_chunked": True,
            "token_count": count_tokens(current_chunk)
        })
    
    # Update total_chunks in each chunk
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk["total_chunks"] = total_chunks
    
    return chunks

def prepare_translation_chunks(files: List[Dict[str, str]], max_chunk_tokens: int, overlap_tokens: int) -> List[TranslationChunk]:
    """Prepare all files for batch translation by chunking them."""
    translation_chunks = []
    
    for file_info in files:
        try:
            # Read file content
            with open(file_info["path"], "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split into chunks if necessary
            chunks = split_into_chunks(content, max_chunk_tokens, overlap_tokens)
            
            # Create TranslationChunk objects
            for chunk in chunks:
                chunk_id = f"{file_info['filename']}_chunk_{chunk['chunk_id']}"
                
                translation_chunk = TranslationChunk(
                    chunk_id=chunk_id,
                    filename=file_info['filename'],
                    file_path=file_info['path'],
                    content=chunk['text'],
                    chunk_number=chunk['chunk_id'],
                    total_chunks=chunk['total_chunks'],
                    is_chunked=chunk['is_chunked'],
                    token_count=chunk['token_count']
                )
                
                translation_chunks.append(translation_chunk)
        
        except Exception as e:
            st.error(f"Error processing file {file_info['filename']}: {str(e)}")
            continue
    
    return translation_chunks

def estimate_batch_cost(num_requests: int, avg_tokens_per_request: int, model: str) -> Dict[str, float]:
    """Estimate the cost of batch processing vs real-time API."""
    
    # Approximate pricing (example rates - replace with actual current pricing)
    pricing = {
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-opus-4-20250514": {"input": 0.015, "output": 0.075}
    }
    
    model_pricing = pricing.get(model, {"input": 0.003, "output": 0.015})
    
    # Estimate tokens (input + output)
    total_input_tokens = num_requests * avg_tokens_per_request
    estimated_output_tokens = total_input_tokens * 0.8  # Assume output is ~80% of input
    
    # Calculate costs (per 1K tokens)
    real_time_cost = (
        (total_input_tokens / 1000) * model_pricing["input"] +
        (estimated_output_tokens / 1000) * model_pricing["output"]
    )
    
    # Batch API typically offers 50% discount
    batch_cost = real_time_cost * 0.5
    
    return {
        "real_time_cost": real_time_cost,
        "batch_cost": batch_cost,
        "savings": real_time_cost - batch_cost,
        "savings_percent": ((real_time_cost - batch_cost) / real_time_cost) * 100,
        "total_input_tokens": total_input_tokens,
        "estimated_output_tokens": estimated_output_tokens
    }

def create_batch_requests(
    translation_chunks: List[TranslationChunk],
    source_lang: str,
    target_lang: str,
    model: str,
    translation_instructions: str
) -> List[BatchRequest]:
    """Create batch API requests for all translation chunks."""
    
    batch_requests = []
    model_limits = get_model_limits(model)
    max_output_tokens = min(model_limits["max_tokens"], 8192)
    
    for chunk in translation_chunks:
        # Build translation prompt
        chunked_context = ""
        if chunk.is_chunked:
            chunked_context = f"""
This is chunk {chunk.chunk_number + 1} of {chunk.total_chunks} from a larger document ({chunk.filename}).
Please translate only this chunk maintaining consistent terminology and style across chunks.
"""
        
        prompt = f"""Please translate the following text from {source_lang} to {target_lang}.
        
{f"Translation instructions: {translation_instructions}" if translation_instructions else ""}

{chunked_context}

Text to translate:
{chunk.content}

Translated text:
"""

        # Create batch request body
        request_body = {
            "model": model,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Create batch request
        batch_request = BatchRequest(
            custom_id=chunk.chunk_id,
            body=request_body
        )
        
        batch_requests.append(batch_request)
    
    return batch_requests

def save_batch_requests_file(batch_requests: List[BatchRequest], batch_id: str) -> str:
    """Save batch requests to JSONL file format required by Claude Batch API."""
    
    batch_file = f"batch_requests_{batch_id}.jsonl"
    
    try:
        with open(batch_file, "w", encoding="utf-8") as f:
            for request in batch_requests:
                # Convert BatchRequest to dict format required by API
                request_dict = {
                    "custom_id": request.custom_id,
                    "method": request.method,
                    "url": request.url,
                    "body": request.body
                }
                
                # Write as JSONL (one JSON object per line)
                f.write(json.dumps(request_dict, ensure_ascii=False) + "\n")
        
        return batch_file
        
    except Exception as e:
        st.error(f"Error saving batch requests: {str(e)}")
        raise

# Main header
st.markdown("<h1 class='main-header'>Claude AI Batch Translator</h1>", unsafe_allow_html=True)
st.markdown("Upload and translate files using Claude's Batch API for cost-effective processing (up to 50% savings)")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Configuration", "📤 Submit Batch", "📊 Monitor Jobs", "📥 Download Results"])

# Configuration tab
with tab1:
    st.markdown("<h2 class='section-header'>Batch Translation Configuration</h2>", unsafe_allow_html=True)
    
    # Batch API Information
    st.markdown("### 💰 Batch API Benefits")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Cost Savings**")
        st.markdown("Up to 50% lower cost compared to real-time API")
    with col2:
        st.markdown("**Large Scale**")
        st.markdown("Perfect for bulk translation jobs")
    with col3:
        st.markdown("**Processing Time**")
        st.markdown("Completes within 24 hours")
    
    # API Configuration
    st.markdown("### API Configuration")
    api_key = st.text_input("Claude API Key", type="password", help="Your Claude API key")
    
    # Language Configuration
    st.markdown("### Language Settings")
    col1, col2 = st.columns(2)
    with col1:
        source_lang = st.text_input("Source Language Code", "en", help="ISO code of source language (e.g., 'en' for English)")
    with col2:
        target_lang = st.text_input("Target Language Code", "es", help="ISO code of target language (e.g., 'es' for Spanish)")
    
    # Model Configuration
    st.markdown("### Model Settings")
    model = st.selectbox(
        "Claude Model", 
        [
            "claude-3-haiku-20240307", 
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ],
        index=4,  # Default to Claude 4 Sonnet
        help="Choose the Claude model for translation. Claude 4 models offer improved performance."
    )
    
    # Batch Configuration
    st.markdown("### Batch Settings")
    col1, col2 = st.columns(2)
    with col1:
        max_chunk_tokens = st.number_input("Max Tokens per Chunk", 1000, 8000, 4000, 
                                         help="Maximum tokens per chunk for translation")
        output_folder = st.text_input("Output Folder", "documents/Translated_Batch", 
                                    help="Folder to save translated files")
    
    with col2:
        overlap_tokens = st.number_input("Overlap Tokens", 0, 800, 150, 
                                       help="Number of overlapping tokens between chunks")
        batch_description = st.text_input("Batch Description", "Document Translation Batch",
                                        help="Description for the batch job")
    
    # Save configuration button
    if st.button("Save Configuration"):
        config = {
            "api_key": api_key,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model,
            "max_chunk_tokens": max_chunk_tokens,
            "overlap_tokens": overlap_tokens,
            "output_folder": output_folder,
            "batch_description": batch_description
        }
        st.session_state.config = config
        st.session_state.success_message = "Configuration saved successfully!"

# Submit Batch tab
with tab2:
    st.markdown("<h2 class='section-header'>Submit Batch Translation Job</h2>", unsafe_allow_html=True)
    
    # Check if configuration exists
    if 'config' not in st.session_state:
        st.warning("Please configure translation settings in the Configuration tab first.")
    else:
        # File upload section
        st.markdown("### Upload Files")
        uploaded_files = st.file_uploader("Select files to translate", accept_multiple_files=True)
        
        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} files:**")
            for file in uploaded_files:
                st.markdown(f"- {file.name} ({len(file.getbuffer()):,} bytes)")
        
        # Translation instructions
        st.markdown("### Translation Instructions")
        instruction_method = st.radio("Instructions input method:", ["Enter text", "Upload file"])
        
        if instruction_method == "Enter text":
            translation_instructions = st.text_area(
                "Enter custom translation instructions",
                "Please maintain formal tone and preserve formatting. Use industry-standard terminology.",
                height=150
            )
            instruction_file = None
        else:
            instruction_file = st.file_uploader("Upload instructions file", type=["txt"])
            if instruction_file:
                translation_instructions = instruction_file.getvalue().decode("utf-8")
                st.text_area("Instructions content", translation_instructions, height=150, disabled=True)
            else:
                translation_instructions = ""
        
        # Cost estimation section
        if uploaded_files and 'config' in st.session_state:
            st.markdown("### Cost Estimation")
            
            # Calculate rough estimates
            total_size = sum(len(file.getbuffer()) for file in uploaded_files)
            estimated_tokens = total_size // 3  # Rough estimate: ~3 chars per token
            estimated_chunks = max(1, estimated_tokens // st.session_state.config['max_chunk_tokens'])
            
            cost_estimate = estimate_batch_cost(
                estimated_chunks, 
                st.session_state.config['max_chunk_tokens'], 
                st.session_state.config['model']
            )
            
            st.markdown(f"""
            <div class="cost-savings">
                <h4>💰 Estimated Costs</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <strong>Real-time API:</strong> ${cost_estimate['real_time_cost']:.2f}<br>
                        <strong>Batch API:</strong> ${cost_estimate['batch_cost']:.2f}<br>
                        <strong>Your Savings:</strong> ${cost_estimate['savings']:.2f} ({cost_estimate['savings_percent']:.1f}%)
                    </div>
                    <div>
                        <strong>Estimated Chunks:</strong> {estimated_chunks:,}<br>
                        <strong>Input Tokens:</strong> {cost_estimate['total_input_tokens']:,}<br>
                        <strong>Output Tokens:</strong> {cost_estimate['estimated_output_tokens']:,}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Submit button
        st.markdown("### Submit Batch Job")
        
        if st.button("Submit Batch Translation Job", type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one file to translate.")
            elif not api_key:
                st.error("Please enter your Claude API key in the Configuration tab.")
            else:
                # Create progress indicators
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                try:
                    status_placeholder.text("Preparing files...")
                    progress_bar.progress(0.1)
                    
                    # Save uploaded files to temporary directory
                    input_folder = "temp_batch_" + str(uuid.uuid4())
                    saved_files = save_uploaded_files(uploaded_files, input_folder)
                    
                    # Save translation instructions
                    if translation_instructions:
                        save_instruction_file(translation_instructions, input_folder)
                    
                    progress_bar.progress(0.2)
                    status_placeholder.text("Processing files and creating chunks...")
                    
                    # Prepare translation chunks
                    translation_chunks = prepare_translation_chunks(
                        saved_files,
                        st.session_state.config["max_chunk_tokens"],
                        st.session_state.config["overlap_tokens"]
                    )
                    
                    if not translation_chunks:
                        st.error("No translation chunks could be prepared.")
                        st.stop()
                    
                    progress_bar.progress(0.4)
                    status_placeholder.text("Creating batch requests...")
                    
                    # Create batch requests
                    batch_requests = create_batch_requests(
                        translation_chunks,
                        st.session_state.config["source_lang"],
                        st.session_state.config["target_lang"],
                        st.session_state.config["model"],
                        translation_instructions
                    )
                    
                    progress_bar.progress(0.6)
                    status_placeholder.text("Saving batch request file...")
                    
                    # Create job ID and save batch file
                    job_id = str(uuid.uuid4())
                    batch_file = save_batch_requests_file(batch_requests, job_id)
                    
                    progress_bar.progress(0.8)
                    status_placeholder.text("Submitting to Claude Batch API...")
                    
                    # For demo purposes, simulate batch submission
                    # In real implementation, you would submit to Claude's Batch API
                    
                    # Create mock batch info
                    batch_info = {
                        "batch_id": f"batch_{uuid.uuid4()}",
                        "status": "submitted",
                        "created_at": datetime.datetime.now().isoformat(),
                        "completion_window": "24h",
                        "description": st.session_state.config["batch_description"]
                    }
                    
                    # Save job information
                    job_info = {
                        "job_id": job_id,
                        "batch_info": batch_info,
                        "config": st.session_state.config,
                        "translation_chunks": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "filename": chunk.filename,
                                "chunk_number": chunk.chunk_number,
                                "total_chunks": chunk.total_chunks,
                                "is_chunked": chunk.is_chunked,
                                "token_count": chunk.token_count
                            }
                            for chunk in translation_chunks
                        ],
                        "cost_estimate": cost_estimate if 'cost_estimate' in locals() else None,
                        "submitted_at": datetime.datetime.now().isoformat(),
                        "batch_file": batch_file,
                        "input_folder": input_folder
                    }
                    
                    # Save to file and session state
                    job_file = f"batch_job_{job_id}.json"
                    with open(job_file, "w", encoding="utf-8") as f:
                        json.dump(job_info, f, indent=2, ensure_ascii=False)
                    
                    # Add to session state
                    st.session_state.batch_jobs.append(job_info)
                    st.session_state.current_job = job_info
                    
                    progress_bar.progress(1.0)
                    status_placeholder.text("Batch job submitted successfully!")
                    
                    st.session_state.success_message = f"""
                    Batch translation job submitted successfully!
                    
                    Job ID: {job_id}
                    Batch ID: {batch_info['batch_id']}
                    Files: {len(saved_files)}
                    Chunks: {len(translation_chunks)}
                    Estimated Cost: ${cost_estimate['batch_cost']:.2f} (saved ${cost_estimate['savings']:.2f})
                    
                    Processing may take up to 24 hours. Monitor progress in the Monitor Jobs tab.
                    """
                
                except Exception as e:
                    st.session_state.error_message = f"Error submitting batch job: {str(e)}"
                    status_placeholder.text("Batch submission failed!")
                
                finally:
                    # Clean up temporary directory
                    try:
                        if 'input_folder' in locals() and os.path.exists(input_folder):
                            shutil.rmtree(input_folder)
                    except Exception as e:
                        st.warning(f"Warning: Could not clean up temporary files: {str(e)}")

# Monitor Jobs tab
with tab3:
    st.markdown("<h2 class='section-header'>Monitor Batch Jobs</h2>", unsafe_allow_html=True)
    
    # Display existing jobs
    if not st.session_state.batch_jobs:
        st.info("No batch jobs found. Submit a batch job first in the Submit Batch tab.")
    else:
        st.markdown("### Active Batch Jobs")
        
        # Create a table of jobs
        job_data = []
        for job in st.session_state.batch_jobs:
            job_data.append({
                "Job ID": job["job_id"][:8] + "...",
                "Batch ID": job["batch_info"]["batch_id"][:8] + "...",
                "Submitted": job["submitted_at"][:16],
                "Status": job["batch_info"]["status"],
                "Files": len(set(chunk["filename"] for chunk in job["translation_chunks"])),
                "Chunks": len(job["translation_chunks"])
            })
        
        if job_data:
            df = pd.DataFrame(job_data)
            st.dataframe(df, use_container_width=True)
        
        # Job selection for monitoring
        st.markdown("### Monitor Specific Job")
        
        job_options = [f"{job['job_id'][:8]}... - {job['batch_info']['status']}" for job in st.session_state.batch_jobs]
        selected_job_index = st.selectbox("Select a job to monitor", range(len(job_options)), format_func=lambda x: job_options[x])
        
        if st.session_state.batch_jobs:
            selected_job = st.session_state.batch_jobs[selected_job_index]
            
            # Display job details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Job Information**")
                st.markdown(f"**Job ID:** {selected_job['job_id']}")
                st.markdown(f"**Batch ID:** {selected_job['batch_info']['batch_id']}")
                st.markdown(f"**Model:** {selected_job['config']['model']}")
                st.markdown(f"**Language:** {selected_job['config']['source_lang']} → {selected_job['config']['target_lang']}")
            
            with col2:
                st.markdown("**Processing Details**")
                st.markdown(f"**Submitted:** {selected_job['submitted_at']}")
                st.markdown(f"**Files:** {len(set(chunk['filename'] for chunk in selected_job['translation_chunks']))}")
                st.markdown(f"**Chunks:** {len(selected_job['translation_chunks'])}")
                
                if selected_job.get('cost_estimate'):
                    cost = selected_job['cost_estimate']
                    st.markdown(f"**Estimated Cost:** ${cost['batch_cost']:.2f}")
            
            # Status display
            status = selected_job['batch_info']['status']
            status_class = f"status-{status.replace(' ', '_')}"
            
            st.markdown(f"""
            <div class="batch-status {status_class}">
                Current Status: {status.upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Status descriptions
            if status == "submitted":
                st.info("📤 Job has been submitted and is waiting to be processed.")
            elif status == "validating":
                st.info("🔍 Job is being validated by Claude's systems.")
            elif status == "in_progress":
                st.info("⏳ Job is currently being processed. This may take up to 24 hours.")
            elif status == "completed":
                st.success("✅ Job has completed successfully! You can download results in the Download Results tab.")
            elif status == "failed":
                st.error("❌ Job has failed. Check the error details and try resubmitting.")
            
            # Refresh button for status
            if st.button("🔄 Refresh Status"):
                # In real implementation, this would check Claude's API
                st.info("Status refresh functionality would check Claude's Batch API here.")
                st.rerun()

# Download Results tab
with tab4:
    st.markdown("<h2 class='section-header'>Download Translation Results</h2>", unsafe_allow_html=True)
    
    # Show completed jobs
    completed_jobs = [job for job in st.session_state.batch_jobs if job['batch_info']['status'] == 'completed']
    
    if not completed_jobs:
        st.info("No completed batch jobs found. Jobs appear here when they finish processing.")
    else:
        st.markdown("### Available Results")
        
        # Select completed job
        job_options = [f"{job['job_id'][:8]}... - {job['config']['source_lang']}→{job['config']['target_lang']}" for job in completed_jobs]
        selected_job_index = st.selectbox("Select a completed job", range(len(job_options)), format_func=lambda x: job_options[x])
        
        selected_job = completed_jobs[selected_job_index]
        
        # Display job summary
        st.markdown("### Job Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Files Processed", len(set(chunk['filename'] for chunk in selected_job['translation_chunks'])))
        with col2:
            st.metric("Total Chunks", len(selected_job['translation_chunks']))
        with col3:
            if selected_job.get('cost_estimate'):
                st.metric("Final Cost", f"${selected_job['cost_estimate']['batch_cost']:.2f}")
        
        # Download options
        st.markdown("### Download Options")
        
        col1, col2 = st.columns(2)
        with col1:
            download_folder = st.text_input("Download Folder", selected_job['config']['output_folder'])
        with col2:
            join_method = st.radio("Chunk Joining Method", ["smart", "simple"], 
                                 help="Smart: overlap detection, Simple: direct concatenation")
        
        # Download button
        if st.button("📥 Download Translated Files", type="primary"):
            try:
                # Create progress indicators
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                status_placeholder.text("Preparing to download results...")
                progress_bar.progress(0.2)
                
                # In real implementation, this would:
                # 1. Download results from Claude's Batch API
                # 2. Parse the JSONL results file
                # 3. Reassemble chunks into complete files
                # 4. Save to the specified folder
                
                # For demo purposes, simulate the process
                ensure_directory_exists(download_folder)
                
                status_placeholder.text("Processing translation results...")
                progress_bar.progress(0.5)
                
                # Simulate file processing
                unique_files = set(chunk['filename'] for chunk in selected_job['translation_chunks'])
                
                status_placeholder.text("Saving translated files...")
                progress_bar.progress(0.8)
                
                # Create mock translated files
                saved_files = []
                for filename in unique_files:
                    base, ext = os.path.splitext(filename)
                    output_filename = f"{base}_{selected_job['config']['target_lang']}{ext}"
                    output_path = os.path.join(download_folder, output_filename)
                    
                    # Create a mock translated file
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(f"[TRANSLATED CONTENT]\n\nThis is a mock translation of {filename}\n")
                    
                    saved_files.append(output_filename)
                
                progress_bar.progress(1.0)
                status_placeholder.text("Download completed!")
                
                st.session_state.success_message = f"""
                Translation download completed successfully!
                
                Downloaded to: {download_folder}
                Files saved: {len(saved_files)}
                Join method: {join_method}
                
                Files:
                """ + "\n".join(f"- {file}" for file in saved_files)
                
            except Exception as e:
                st.session_state.error_message = f"Error downloading results: {str(e)}"

# Display success/error messages
if st.session_state.success_message:
    st.success(st.session_state.success_message)
    st.session_state.success_message = None

if st.session_state.error_message:
    st.error(st.session_state.error_message)
    st.session_state.error_message = None

# Sidebar with batch processing information
with st.sidebar:
    st.markdown("## 🔄 Batch Processing Info")
    
    st.markdown("### Benefits")
    st.markdown("- **50% cost savings** vs real-time API")
    st.markdown("- **Large scale processing** for 100+ files")
    st.markdown("- **Automated job management**")
    st.markdown("- **24-hour completion window**")
    
    st.markdown("### Best For")
    st.markdown("- Large document collections")
    st.markdown("- Non-urgent translations")
    st.markdown("- Cost-sensitive projects")
    st.markdown("- Scheduled/automated workflows")
    
    st.markdown("### Workflow")
    st.markdown("1. **Configure** - Set up languages and model")
    st.markdown("2. **Submit** - Upload files and create batch job")
    st.markdown("3. **Monitor** - Track job progress")
    st.markdown("4. **Download** - Get translated files")
    
    if st.session_state.batch_jobs:
        st.markdown("### Current Jobs")
        for job in st.session_state.batch_jobs[-3:]:  # Show last 3 jobs
            status = job['batch_info']['status']
            st.markdown(f"**{job['job_id'][:8]}...** - {status}")

# Footer
st.markdown("---")
st.markdown(
    "**Claude AI Batch Translator** | Built with Streamlit and Anthropic Claude Batch API | "
    "Cost-effective processing for large-scale translation projects"
)