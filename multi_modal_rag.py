#!/usr/bin/env python
# coding: utf-8

# # System Dependencies
# 
# To get started with Unstructured.io, we need a few system-wide dependencies: 
# 
# ## Poppler (poppler-utils)
# Handles PDF processing. It's a library that can extract text, images, and metadata from PDFs. Unstructured uses it to parse PDF documents and convert them into processable text.
# 
# ## Tesseract (tesseract-ocr) 
# Optical Character Recognition (OCR) engine. When you have scanned documents, images with text, or PDFs that are essentially pictures, Tesseract reads the text from these images and converts it to machine-readable text.
# 
# ## libmagic
# File type detection library. It identifies what type of file you're dealing with (PDF, Word doc, image, etc.) by analyzing the file's content, not just the extension. This helps Unstructured choose the right processing method for each document.

# In[ ]:


import json
from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings


# In[1]:


# for linux
# !apt-get install poppler-utils tesseract-ocr libmagic-dev

# for mac
# !brew install poppler tesseract libmagic


# In[ ]:


get_ipython().run_line_magic('pip', 'install -Uq "unstructured[all-docs]"')
get_ipython().run_line_magic('pip', 'install -Uq langchain_chroma')
get_ipython().run_line_magic('pip', 'install -Uq langchain langchain-community langchain-ollama')
get_ipython().run_line_magic('pip', 'install -Uq python_dotenv')


# In[ ]:


# Configuration for Tesseract and Poppler path manually
import os
import sys

# Define paths
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\Program Files\poppler-25.12.0\Library\bin'

# 1. Configure standard pytesseract (just in case)
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    pass

# 2. Configure unstructured_pytesseract (CRITICAL FIX)
try:
    import unstructured_pytesseract
    import unstructured_pytesseract.pytesseract
    unstructured_pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print("✅ Configured unstructured_pytesseract directly")
except ImportError:
    print("⚠️ Could not import unstructured_pytesseract (might not be installed yet)")

# 3. Add Poppler to PATH
if poppler_path not in os.environ['PATH']:
    os.environ['PATH'] += ';' + poppler_path
    print(f"✅ Added Poppler to PATH: {poppler_path}")


# In[ ]:


# --- WINDOWS COPY-PASTE FIX START ---
import os
import sys
import unstructured_pytesseract.pytesseract

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
poppler_path = r'C:\Program Files\poppler-25.12.0\Library\bin'

# Force unstructured to use this Tesseract
unstructured_pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Add Poppler to PATH
if poppler_path not in os.environ['PATH']:
    os.environ['PATH'] += ';' + poppler_path
# --- WINDOWS COPY-PASTE FIX END ---

def partition_document(file_path: str):
    """Extract elements from PDF using unstructured"""
    print(f"📄 Partitioning document: {file_path}")

    elements = partition_pdf(
        filename=file_path,  # Path to your PDF file
        strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
        infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
        extract_image_block_types=["Image"], # Grab images found in the PDF
        extract_image_block_to_payload=True,
    languages=["eng", "fra"] # Support English and French # Store images as base64 data you can actually use
    )

    print(f"✅ Extracted {len(elements)} elements")
    return elements

# Test with your PDF file
file_path = "./data/raw/normes/certification ISO 260000.pdf"  # Change this to your PDF path
elements = partition_document(file_path)


# In[ ]:


# elements
# len(elements)


elements


# In[ ]:


# All types of different atomic elements we see from unstructured
set([str(type(el)) for el in elements])


# In[ ]:


elements[36].to_dict()


# In[ ]:


# Gather all images
images = [element for element in elements if element.category == 'Image']
print(f"Found {len(images)} images")

images[0].to_dict()

# Use https://codebeautify.org/base64-to-image-converter to view the base64 text


# In[ ]:


# Gather all table
tables = [element for element in elements if element.category == 'Table']
print(f"Found {len(tables)} tables")

tables[0].to_dict()

# Use https://jsfiddle.net/ to view the table html 


# In[ ]:


def create_chunks_by_title(elements):
    """Create intelligent chunks using title-based strategy"""
    print("🔨 Creating smart chunks...")

    chunks = chunk_by_title(
        elements, # The parsed PDF elements from previous step
        max_characters=3000, # Hard limit - never exceed 3000 characters per chunk
        new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
        combine_text_under_n_chars=500 # Merge tiny chunks under 500 chars with neighbors
    )

    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# Create chunks
chunks = create_chunks_by_title(elements)


# In[ ]:


# View all chunks
# chunks

# All unique types
set([str(type(chunk)) for chunk in chunks])


# In[ ]:


# View a single chunk
# chunks[2].to_dict()

# View original elements
chunks[11].metadata.orig_elements[-1].to_dict()
# Note: 4th chunk has the first image + 11th chunk has the first table in the sample PDF


# In[ ]:


def separate_content_types(chunk):
    """Analyze what types of content are in a chunk"""
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    # Check for tables and images in original elements
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__

            # Handle tables
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)

            # Handle images
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)

    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    """Create AI-enhanced summary for mixed content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOllama(model="llava", temperature=0)

        # Build the text prompt
        prompt_text = f"""You are creating a searchable description for document content retrieval.

        CONTENT TO ANALYZE:
        TEXT CONTENT:
        {text}

        """

        # Add tables if present
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"

                prompt_text += """
                YOUR TASK:
                Generate a comprehensive, searchable description that covers:

                1. Key facts, numbers, and data points from text and tables
                2. Main topics and concepts discussed  
                3. Questions this content could answer
                4. Visual content analysis (charts, diagrams, patterns in images)
                5. Alternative search terms users might use

                Make it detailed and searchable - prioritize findability over brevity.

                SEARCHABLE DESCRIPTION:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add images to the message
        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"     ❌ AI summary failed: {e}")
        # Fallback to simple summary
        summary = f"{text[:300]}..."
        if tables:
            summary += f" [Contains {len(tables)} table(s)]"
        if images:
            summary += f" [Contains {len(images)} image(s)]"
        return summary

def summarise_chunks(chunks):
    """Process all chunks with AI Summaries"""
    print("🧠 Processing chunks with AI Summaries...")

    langchain_documents = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")

        # Analyze chunk content
        content_data = separate_content_types(chunk)

        # Debug prints
        print(f"     Types found: {content_data['types']}")
        print(f"     Tables: {len(content_data['tables'])}, Images: {len(content_data['images'])}")

        # Create AI-enhanced summary if chunk has tables/images
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
                print(f"     → AI summary created successfully")
                print(f"     → Enhanced content preview: {enhanced_content[:200]}...")
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            print(f"     → Using raw text (no tables/images)")
            enhanced_content = content_data['text']

        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )

        langchain_documents.append(doc)

    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents


# Process chunks with AI
processed_chunks = summarise_chunks(chunks)


# In[ ]:


processed_chunks


# In[ ]:


def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []

    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(export_data)} chunks to {filename}")
    return export_data

# Export your chunks
json_data = export_chunks_to_json(processed_chunks)


# In[ ]:


def create_vector_store(documents, persist_directory="dbv1/chroma_db"):
    """Create and persist ChromaDB vector store"""
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    print("🔮 Creating embeddings and storing in ChromaDB...")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

# Create the vector store
db = create_vector_store(processed_chunks)


# In[ ]:


# After your retrieval
query = "What are the two main components of the Transformer architecture? "
retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

# Export to JSON
export_chunks_to_json(chunks, "rag_results.json")


# In[ ]:


def run_complete_ingestion_pipeline(pdf_path: str):
    """Run the complete RAG ingestion pipeline"""
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Partition
    elements = partition_document(pdf_path)

    # Step 2: Chunk
    chunks = create_chunks_by_title(elements)

    # Step 3: AI Summarisation
    summarised_chunks = summarise_chunks(chunks)

    # Step 4: Vector Store
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db")

    print("🎉 Pipeline completed successfully!")
    return db

# Run the complete pipeline


# In[ ]:


db = run_complete_ingestion_pipeline("./data/raw/normes/certification ISO 260000.pdf")


# In[ ]:


# Query the vector store
query = "How many attention heads does the Transformer use, and what is the dimension of each head? "

retriever = db.as_retriever(search_kwargs={"k": 3})
chunks = retriever.invoke(query)

def generate_final_answer(chunks, query):
    """Generate final answer using multimodal content"""

    try:
        # Initialize LLM (needs vision model for images)
        llm = ChatOllama(model="llava", temperature=0)

        # Build the text prompt
        prompt_text = f"""Based on the following documents, please answer this question: {query}

CONTENT TO ANALYZE:
"""

        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"

            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])

                # Add raw text
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"

                # Add tables as HTML
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"

            prompt_text += "\n"

        prompt_text += """
Please provide a clear, comprehensive answer using the text, tables, and images above. If the documents don't contain sufficient information to answer the question, say "I don't have enough information to answer that question based on the provided documents."

ANSWER:"""

        # Build message content starting with text
        message_content = [{"type": "text", "text": prompt_text}]

        # Add all images from all chunks
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])

                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })

        # Send to AI and get response
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])

        return response.content

    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

# Usage
final_answer = generate_final_answer(chunks, query)
print(final_answer)

