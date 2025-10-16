
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import PyPDF2
import os
import shutil
from typing import List

# Initialize the FastAPI application
application = FastAPI(
    title="Document Intelligence API",
    description="Intelligent document querying system with RAG architecture",
    version="1.0.0"
)

# Configure cross-origin resource sharing
application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for vector database instance
document_store = None

# Configure embedding model for semantic search
print("Initializing semantic embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
print("Embedding model initialization complete")

# Configure language model for response generation
language_model = Ollama(model="llama3.2", base_url="http://localhost:11434")


class QuestionRequest(BaseModel):
    """Schema for user questions"""
    question: str


def read_pdf_content(pdf_stream):
    """
    Extracts textual content from PDF documents.
    
    Args:
        pdf_stream: Binary stream of PDF file
        
    Returns:
        str: Concatenated text from all pages
    """
    try:
        reader = PyPDF2.PdfReader(pdf_stream)
        extracted_text = ""
        
        for page_obj in reader.pages:
            content = page_obj.extract_text()
            if content:
                extracted_text += content + "\n"
        
        return extracted_text
    except Exception as error:
        raise Exception(f"PDF extraction failed: {str(error)}")


@application.post("/upload")
async def process_file_upload(files: List[UploadFile] = File(...)):
    """
    Endpoint to ingest and process document files.
    Supports PDF and text file formats.
    """
    global document_store
    
    try:
        extracted_contents = []
        successfully_processed = []
        
        print(f"\nBeginning processing of {len(files)} uploaded file(s)...")
        
        for uploaded_file in files:
            print(f"  Processing: {uploaded_file.filename}")
            
            # Create temporary file for processing
            temporary_file_path = f"temp_{uploaded_file.filename}"
            with open(temporary_file_path, "wb") as temp_buffer:
                shutil.copyfileobj(uploaded_file.file, temp_buffer)
            
            try:
                # Determine file type and extract content
                if uploaded_file.filename.lower().endswith('.pdf'):
                    with open(temporary_file_path, 'rb') as pdf_stream:
                        content = read_pdf_content(pdf_stream)
                else:
                    with open(temporary_file_path, 'r', encoding='utf-8', errors='ignore') as text_stream:
                        content = text_stream.read()
                
                # Validate extracted content
                if content.strip():
                    extracted_contents.append(content)
                    successfully_processed.append(uploaded_file.filename)
                    print(f"    Successfully extracted content from {uploaded_file.filename}")
                else:
                    print(f"    Warning: No content found in {uploaded_file.filename}")
                    
            except Exception as processing_error:
                print(f"    Error processing {uploaded_file.filename}: {str(processing_error)}")
            finally:
                # Remove temporary file
                if os.path.exists(temporary_file_path):
                    os.remove(temporary_file_path)
        
        # Validate that we have content to process
        if not extracted_contents:
            raise HTTPException(
                status_code=400, 
                detail="Unable to extract content from any uploaded files"
            )
        
        # Chunk the extracted text for optimal retrieval
        print("\nChunking documents for vector storage...")
        chunking_strategy = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        text_chunks = []
        for content in extracted_contents:
            chunks = chunking_strategy.split_text(content)
            text_chunks.extend(chunks)
        
        print(f"Generated {len(text_chunks)} text chunks")
        
        # Build vector database from chunks
        print("Constructing vector database...")
        document_store = Chroma.from_texts(
            texts=text_chunks,
            embedding=embedding_model,
            persist_directory="./chroma_db"
        )
        
        print("Vector database construction complete\n")
        
        return {
            "message": f"Processed {len(successfully_processed)} document(s) successfully",
            "files": successfully_processed,
            "chunks": len(text_chunks),
            "status": "complete"
        }
    
    except HTTPException:
        raise
    except Exception as error:
        print(f"Upload processing error: {str(error)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Document processing failed: {str(error)}"
        )


@application.post("/query")
async def answer_question(request: QuestionRequest):
    """
    Endpoint to query the document knowledge base.
    Returns AI-generated answers based on document content.
    """
    global document_store
    
    # Verify document store exists
    if document_store is None:
        raise HTTPException(
            status_code=400, 
            detail="Document store not initialized. Upload documents first."
        )
    
    try:
        print(f"\nProcessing question: {request.question}")
        
        # Retrieve semantically similar document chunks
        print("Searching for relevant document sections...")
        relevant_chunks = document_store.similarity_search(request.question, k=4)
        
        if not relevant_chunks:
            return {
                "question": request.question,
                "answer": "No relevant information found in the document corpus.",
                "sources": 0
            }
        
        print(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        # Construct context from retrieved chunks
        contextual_information = "\n\n".join([
            f"[Excerpt {idx+1}]\n{chunk.page_content}" 
            for idx, chunk in enumerate(relevant_chunks)
        ])
        
        # Generate response using language model
        print("Generating contextual response...")
        
        prompt_template = f"""Based on the following excerpts from documents, provide an accurate answer to the user's question.

Document Excerpts:
{contextual_information}

User Question: {request.question}

Guidelines:
- Answer must be derived solely from the provided excerpts
- If information is insufficient, clearly state this limitation
- Provide specific references to support your answer
- Maintain brevity while being comprehensive (under 200 words)

Response:"""

        generated_response = language_model.invoke(prompt_template)
        
        print("Response generation complete\n")
        
        return {
            "question": request.question,
            "answer": generated_response.strip(),
            "sources": len(relevant_chunks),
            "status": "success"
        }
    
    except Exception as error:
        print(f"Query processing error: {str(error)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Question processing failed: {str(error)}"
        )


@application.get("/")
async def api_info():
    """Root endpoint providing API information"""
    return {
        "service": "Document Intelligence API",
        "status": "operational",
        "version": "1.0.0",
        "capabilities": {
            "document_upload": "/upload endpoint",
            "question_answering": "/query endpoint",
            "system_diagnostics": "/health endpoint"
        },
        "technology": {
            "embedding": "all-MiniLM-L6-v2",
            "generation": "llama3.2"
        }
    }


@application.get("/health")
async def system_diagnostics():
    """
    System health check endpoint.
    Verifies all components are operational.
    """
    # Test language model connectivity
    try:
        language_model.invoke("ping")
        llm_status = "operational"
    except:
        llm_status = "unavailable"
    
    return {
        "api_server": "running",
        "vector_database": "initialized" if document_store else "not_initialized",
        "language_model": llm_status,
        "embedding_service": "active"
    }


# Application entry point
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("Starting Document Intelligence API Server")
    print("="*70)
    print("\nPrerequisites:")
    print("  • Ensure Ollama service is running (ollama serve)")
    print("\nAPI Endpoints:")
    print("  • POST /upload - Ingest documents")
    print("  • POST /query - Ask questions")
    print("  • GET /health - System status")
    print("\nDocumentation: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(application, host="0.0.0.0", port=8000)