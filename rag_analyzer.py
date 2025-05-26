#!/usr/bin/env python3
"""
RAG PDF Analyzer - Improved Version
A Retrieval-Augmented Generation system for PDF document analysis
"""

import os
import sys
import time
import logging
import argparse
import configparser
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from PyPDF2 import PdfReader
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from pdf2image import convert_from_path
from multiprocessing import Pool, cpu_count
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RAGPDFAnalyzer:
    def __init__(self, config_path: str = "config.ini"):
        """Initialize the RAG PDF Analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.client = self._initialize_openai()
        self.index = None
        self.metadata = []
        self._setup_paths()
        self._validate_prerequisites()
        
    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        """Load configuration from file."""
        config = configparser.ConfigParser()
        
        # Set defaults
        config['DEFAULT'] = {
            'pdf_path': 'input.pdf',
            'output_dir': 'output',
            'tesseract_cmd': '',
            'poppler_path': '',
            'max_chunk_size': '2000',
            'top_k_results': '3',
            'dpi': '300',
            'embedding_model': 'text-embedding-ada-002',
            'chat_model': 'gpt-4'
        }
        
        if os.path.exists(config_path):
            config.read(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            
        return config
    
    def _initialize_openai(self) -> OpenAI:
        """Initialize OpenAI client."""
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        return OpenAI(api_key=api_key)
    
    def _setup_paths(self):
        """Setup file paths from configuration."""
        self.pdf_path = self.config.get('DEFAULT', 'pdf_path')
        self.output_dir = Path(self.config.get('DEFAULT', 'output_dir'))
        self.output_dir.mkdir(exist_ok=True)
        
        self.output_text = self.output_dir / "extracted_text.txt"
        self.output_markdown = self.output_dir / "extracted_text.md"
        
        # Set system paths if provided
        tesseract_cmd = self.config.get('DEFAULT', 'tesseract_cmd')
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        self.poppler_path = self.config.get('DEFAULT', 'poppler_path') or None
        
    def _validate_prerequisites(self):
        """Check if all required tools are installed."""
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR found")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise Exception("Tesseract OCR is required. Please install it and update config.ini")
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
        # Initialize FAISS index - dimension depends on embedding model
        embedding_model = self.config.get('DEFAULT', 'embedding_model')
        if "3-large" in embedding_model:
            embedding_dim = 3072
        elif "3-small" in embedding_model:
            embedding_dim = 1536
        else:
            embedding_dim = 1536  # Default for older models
        
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"FAISS index initialized with dimension {embedding_dim} for model {embedding_model}")
        logger.info("FAISS index initialized")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess an image to enhance OCR accuracy."""
        image = ImageOps.grayscale(image)
        image = image.filter(ImageFilter.SHARPEN)
        return image

    def clean_text(self, text: str) -> str:
        """Remove unnecessary metadata or repeated lines from extracted text."""
        lines = text.split("\n")
        filtered_lines = [
            line for line in lines 
            if not any(keyword in line.lower() for keyword in ["copyright", "confidential"])
        ]
        return "\n".join(filtered_lines)

    def extract_page_text(self, page_num: int, pdf_path: str) -> str:
        """Extract text from a single PDF page."""
        try:
            reader = PdfReader(pdf_path)
            page = reader.pages[page_num]
            text = page.extract_text()
            
            if text and text.strip():
                logger.debug(f"Extracted text from page {page_num + 1}")
                return f"# Page {page_num + 1}\n\n{self.clean_text(text)}"
            
            # Fallback to OCR
            logger.info(f"Using OCR for page {page_num + 1}")
            dpi = int(self.config.get('DEFAULT', 'dpi'))
            
            images = convert_from_path(
                pdf_path, 
                first_page=page_num + 1, 
                last_page=page_num + 1, 
                poppler_path=self.poppler_path, 
                dpi=dpi
            )
            
            ocr_texts = []
            for img in images:
                processed_img = self.preprocess_image(img)
                ocr_text = pytesseract.image_to_string(processed_img)
                ocr_texts.append(self.clean_text(ocr_text))
                
            return f"# Page {page_num + 1} (OCR)\n\n" + "\n".join(ocr_texts)
            
        except Exception as e:
            logger.error(f"Error processing page {page_num + 1}: {e}")
            return f"# Page {page_num + 1} (Error)\n\nError: {e}"

    def process_pdf(self, use_multiprocessing: bool = True) -> None:
        """Process the entire PDF and extract text."""
        logger.info(f"Processing PDF: {self.pdf_path}")
        
        reader = PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"Total pages: {total_pages}")
        
        if use_multiprocessing and total_pages > 1:
            logger.info("Using multiprocessing for PDF extraction")
            with Pool(processes=cpu_count()) as pool:
                results = pool.starmap(
                    self.extract_page_text, 
                    [(page_num, self.pdf_path) for page_num in range(total_pages)]
                )
        else:
            logger.info("Using single-process extraction")
            results = [
                self.extract_page_text(page_num, self.pdf_path) 
                for page_num in range(total_pages)
            ]
        
        # Save results
        content = "\n\n".join(results)
        
        with open(self.output_text, "w", encoding="utf-8") as f:
            f.write(content)
        with open(self.output_markdown, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Text extracted and saved to: {self.output_text}")

    def create_embeddings(self) -> List[List[float]]:
        """Create embeddings for text chunks."""
        logger.info("Creating embeddings...")
        
        max_chunk_size = int(self.config.get('DEFAULT', 'max_chunk_size'))
        
        try:
            with open(self.output_text, "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split into chunks
            chunks = self._split_text_into_chunks(text, max_chunk_size)
            logger.info(f"Created {len(chunks)} chunks")
            
            embeddings = []
            for i, chunk in enumerate(chunks, 1):
                if not chunk.strip():
                    continue
                    
                try:
                    logger.debug(f"Processing chunk {i}/{len(chunks)}")
                    response = self.client.embeddings.create(
                        input=chunk,
                        model=self.config.get('DEFAULT', 'embedding_model')
                    )
                    
                    embedding = response.data[0].embedding
                    embeddings.append(embedding)
                    
                    self.metadata.append({
                        "text": chunk,
                        "chunk_id": i,
                        "preview": chunk[:100].replace("\n", " ")
                    })
                    
                    # Rate limiting
                    if i < len(chunks):
                        time.sleep(0.1)
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {e}")
                    continue
            
            if embeddings:
                embeddings_array = np.array(embeddings).astype("float32")
                self.index.add(embeddings_array)
                logger.info(f"Added {len(embeddings)} embeddings to FAISS index")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in create_embeddings: {e}")
            raise

    def _split_text_into_chunks(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks intelligently."""
        raw_chunks = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for chunk in raw_chunks:
            if len(chunk.strip()) < 20:
                continue
                
            if len(current_chunk) + len(chunk) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = chunk + "\n\n"
            else:
                current_chunk += chunk + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def retrieve_relevant_chunks(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks for a query."""
        top_k = int(self.config.get('DEFAULT', 'top_k_results'))
        
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")
        
        response = self.client.embeddings.create(
            input=query,
            model=self.config.get('DEFAULT', 'embedding_model')
        )
        
        query_embedding = response.data[0].embedding
        query_embedding_np = np.array(query_embedding).astype("float32")
        
        distances, indices = self.index.search(np.array([query_embedding_np]), top_k)
        
        return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]

    def answer_question(self, query: str) -> str:
        """Answer a query using retrieved context."""
        logger.info(f"Processing query: {query}")
        
        if not self.metadata:
            return "No document has been processed yet. Please process a PDF first."
        
        relevant_chunks = self.retrieve_relevant_chunks(query)
        
        if not relevant_chunks:
            return "No relevant content found. Please try a different question."
        
        context = "\n\n".join([
            f"Chunk {chunk['chunk_id']}:\n{chunk['text']}" 
            for chunk in relevant_chunks
        ])
        
        try:
            chat_model = self.config.get('DEFAULT', 'chat_model')
            
            # Reasoning models (o-series) don't support temperature parameter
            if chat_model.startswith(('o1', 'o3', 'o4')):
                response = self.client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert document analyst. Answer questions based only on the provided context. If the context doesn't contain enough information, say so."
                        },
                        {
                            "role": "user", 
                            "content": f"Question: {query}\n\nContext:\n{context}"
                        }
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert document analyst. Answer questions based only on the provided context. If the context doesn't contain enough information, say so."
                        },
                        {
                            "role": "user", 
                            "content": f"Question: {query}\n\nContext:\n{context}"
                        }
                    ],
                    temperature=0.1
                )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return f"Error generating response: {e}"

    def summarize_document(self) -> str:
        """Generate a summary of the entire document."""
        logger.info("Generating document summary")
        
        if not self.metadata:
            return "No document has been processed yet."
        
        # Use first few chunks for summary to avoid token limits
        context = "\n\n".join([
            chunk['text'] for chunk in self.metadata[:5]
        ])
        
        try:
            chat_model = self.config.get('DEFAULT', 'chat_model')
            
            # Reasoning models (o-series) don't support temperature parameter
            if chat_model.startswith(('o1', 'o3', 'o4')):
                response = self.client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert document analyst. Provide a comprehensive summary of the document."
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize this document:\n\n{context}"
                        }
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert document analyst. Provide a comprehensive summary of the document."
                        },
                        {
                            "role": "user",
                            "content": f"Please summarize this document:\n\n{context}"
                        }
                    ],
                    temperature=0.1
                )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in summarize_document: {e}")
            return f"Error generating summary: {e}"

    def get_debug_info(self) -> str:
        """Get debug information about the current state."""
        info = f"""
=== Debug Information ===
PDF Path: {self.pdf_path}
Output Directory: {self.output_dir}
Number of chunks: {len(self.metadata)}
FAISS index size: {self.index.ntotal if self.index else 0}
Configuration loaded: {bool(self.config)}
========================
        """
        return info.strip()

    def interactive_session(self):
        """Start an interactive Q&A session."""
        print("RAG PDF Analyzer - Interactive Session")
        print("Commands: 'summarize', 'debug', 'help', 'exit'")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nEnter your question: ").strip()
                
                if not query:
                    continue
                    
                if query.lower() == "exit":
                    print("Goodbye!")
                    break
                elif query.lower() == "help":
                    print("Available commands:")
                    print("- Ask any question about the document")
                    print("- 'summarize' - Get document summary")
                    print("- 'debug' - Show debug information")
                    print("- 'exit' - Quit the program")
                elif query.lower() == "debug":
                    print(self.get_debug_info())
                elif query.lower() == "summarize":
                    print("\nGenerating summary...")
                    summary = self.summarize_document()
                    print(f"\nSummary:\n{summary}")
                else:
                    print("\nThinking...")
                    answer = self.answer_question(query)
                    print(f"\nAnswer:\n{answer}")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print(f"An error occurred: {e}")


def main():
    parser = argparse.ArgumentParser(description='RAG PDF Analyzer')
    parser.add_argument('--pdf', help='Path to PDF file')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--no-multiprocessing', action='store_true', help='Disable multiprocessing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = RAGPDFAnalyzer(args.config)
        
        # Override config with command line arguments
        if args.pdf:
            analyzer.pdf_path = args.pdf
        if args.output_dir:
            analyzer.output_dir = Path(args.output_dir)
            analyzer.output_dir.mkdir(exist_ok=True)
            analyzer.output_text = analyzer.output_dir / "extracted_text.txt"
            analyzer.output_markdown = analyzer.output_dir / "extracted_text.md"
        
        # Process PDF
        print("Processing PDF...")
        use_mp = analyzer.config.getboolean('DEFAULT', 'use_multiprocessing', fallback=True)
        analyzer.process_pdf(use_multiprocessing=use_mp and not args.no_multiprocessing)
        
        # Create embeddings
        print("Creating embeddings...")
        analyzer.create_embeddings()
        
        # Start interactive session
        analyzer.interactive_session()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()