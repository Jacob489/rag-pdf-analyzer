# RAG PDF Analyzer

A Retrieval-Augmented Generation (RAG) system that extracts knowledge from PDF documents and enables intelligent question-answering using OpenAI's language models and FAISS vector search.
## 🚀 Key Features

- **Advanced Document Processing**: Native PDF text extraction with OCR fallback for scanned documents
- **Flexible AI Model Support**: Compatible with OpenAI's full model range (o4-mini, gpt-4.1, gpt-4o, etc.)
- **Text Chunking**: Semantic segmentation optimized for embedding models
- **Vector Similarity Search**: FAISS-powered efficient retrieval across document embeddings
- **Interactive Q&A Interface**: Natural language queries with context-aware responses
- **Document Summarization**: AI-powered comprehensive document summaries
- **Production-Ready**: Comprehensive logging, error handling, and configuration management

## 🛠️ Technical Stack

- **Document Processing**: PyPDF2 + Tesseract OCR with intelligent fallback
- **Embeddings**: OpenAI text-embedding models (configurable dimensions)
- **Vector Database**: FAISS IndexFlatL2 for similarity search
- **Language Models**: OpenAI GPT models with reasoning model support
- **Architecture**: Object-oriented design with comprehensive error handling

## 📋 Prerequisites

- **Python 3.8+**
- **OpenAI API Key** with access to embedding and chat completion models
- **Tesseract OCR** for scanned document processing
- **Poppler utilities** for PDF to image conversion

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-pdf-analyzer.git
cd rag-pdf-analyzer
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install System Dependencies

**Windows:**
- Download and install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Download and install [Poppler](https://github.com/oschwartz10612/poppler-windows)

**macOS:**
```bash
brew install tesseract poppler
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

### 4. Configure Environment

**Set up your API key:**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

**Configure system paths:**
```bash
cp config.ini.example config.ini
# Edit config.ini with your system-specific paths
```

### 5. Validate Installation
```bash
python test_setup.py
```

## ⚙️ Configuration

### Environment Variables (.env)
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### System Configuration (config.ini)
```ini
[DEFAULT]
# Document Processing
pdf_path = your_document.pdf
output_dir = output
max_chunk_size = 2000
top_k_results = 3
dpi = 300
use_multiprocessing = true

# System Paths (update for your system)
tesseract_cmd = /usr/local/bin/tesseract
poppler_path = /usr/local/bin

# AI Models (latest supported models)
embedding_model = text-embedding-3-large
chat_model = o4-mini
```

### Model Compatibility

The system automatically adapts to your OpenAI API access level:

**Chat Models** (use best available):
- `o4-mini` - Latest reasoning model  
- `gpt-4.1` - Advanced reasoning and coding
- `gpt-4o` - Multimodal capabilities  
- `gpt-4-turbo` - High performance
- `gpt-3.5-turbo` - Widely available fallback

**Embedding Models:**
- `text-embedding-3-large` - Highest quality (3072 dimensions)
- `text-embedding-3-small` - Balanced performance (1536 dimensions)
- `text-embedding-ada-002` - Legacy compatibility

*Check your available models:*
```bash
python check_available_models.py
```

## 🚀 Usage

### Basic Usage
```bash
# Process a PDF with default settings
python rag_analyzer.py --pdf "your_document.pdf"
```

### Advanced Usage
```bash
# Custom output directory and verbose logging
python rag_analyzer.py \
  --pdf "research_paper.pdf" \
  --output-dir "analysis_results" \
  --verbose

# Disable multiprocessing for compatibility
python rag_analyzer.py \
  --pdf "document.pdf" \
  --no-multiprocessing

# Use custom configuration
python rag_analyzer.py \
  --config "custom_config.ini" \
  --pdf "document.pdf"
```

### Interactive Commands

Once the system loads, you can use these commands:

- **Ask Questions**: Type any natural language question about your document
- **`summarize`**: Generate a comprehensive document summary
- **`debug`**: Display system information and processing statistics
- **`help`**: Show available commands
- **`exit`**: Close the application

### Example Session
```
RAG PDF Analyzer - Interactive Session
Commands: 'summarize', 'debug', 'help', 'exit'
--------------------------------------------------

Enter your question: What benchmarks were used in this study?

Answer: Based on the document, the authors compared their model against:
1. GAN implementation from Margalef-Bentabol et al. (2020)
2. DDPM from Smith et al. (2022)
They evaluated performance using FID scores, ellipticity, semi-major axis...

Enter your question: summarize

Summary: This research paper presents a novel approach to...
[Comprehensive AI-generated summary]

Enter your question: exit
Goodbye!
```

## 🧪 Testing and Validation

### System Validation
```bash
# Test all components and dependencies
python test_setup.py

# Check available OpenAI models
python check_available_models.py
```

### Command Line Testing
```bash
# Test help functionality
python rag_analyzer.py --help

# Test error handling
python rag_analyzer.py --pdf "nonexistent.pdf"
```

## 📊 Performance Notes

- Processing speed depends on document size, content complexity, and OCR requirements
- Memory usage scales with document size and number of chunks generated
- Supports standard PDF files (tested with documents up to 20 pages)

## 🔍 Technical Deep Dive

### Document Processing Workflow

1. **PDF Ingestion**: PyPDF2 attempts native text extraction
2. **OCR Fallback**: If text extraction fails, converts to images and uses Tesseract
3. **Text Cleaning**: Removes metadata, headers, and formatting artifacts
4. **Intelligent Chunking**: Segments text while preserving semantic boundaries
5. **Embedding Generation**: Creates vector representations using OpenAI's embedding models
6. **Index Creation**: Stores embeddings in FAISS for fast similarity search

### Query Processing Pipeline

1. **Query Embedding**: Convert user question to vector representation
2. **Similarity Search**: Find top-k most relevant document chunks
3. **Context Assembly**: Combine retrieved chunks with metadata
4. **LLM Processing**: Generate response using retrieved context
5. **Response Formatting**: Clean and present the final answer

### Vector Search Implementation

```python
# FAISS index configuration
embedding_dim = 3072  # for text-embedding-3-large
index = faiss.IndexFlatL2(embedding_dim)

# Retrieval with similarity scoring
distances, indices = index.search(query_embedding, top_k=3)
relevant_chunks = [metadata[i] for i in indices[0]]
```

## 🛡️ Error Handling and Logging

- **Comprehensive Logging**: Detailed logs for debugging and monitoring
- **Graceful Degradation**: Fallback mechanisms for OCR and API failures
- **Input Validation**: Robust checking of file paths, API keys, and configurations
- **Rate Limiting**: Automatic handling of API quota limits
- **Memory Management**: Efficient processing of large documents
 
## 🚦 Troubleshooting

### Common Issues

**API Connection Errors:**
```bash
# Verify API key and model access
python check_available_models.py
```

**PDF Processing Failures:**
- Ensure Tesseract is properly installed and configured
- Check PDF file permissions and corruption
- Verify Poppler installation for image conversion

**Memory Issues:**
- Reduce `max_chunk_size` in configuration
- Disable multiprocessing for large files
- Process documents in smaller segments

**Model Access Errors:**
- Verify OpenAI account has access to requested models
- Check billing and usage limits
- Try fallback models (gpt-3.5-turbo, text-embedding-ada-002)

## 📈 Future Enhancements

- **Multi-format Support**: DOCX, TXT, HTML document processing
- **Cloud Integration**: AWS S3, Google Drive, SharePoint connectors
- **Advanced Chunking**: Semantic segmentation using sentence transformers
- **Conversation Memory**: Multi-turn dialogue with context retention
- **Batch Processing**: Automated processing of document collections
- **Web Interface**: Flask/FastAPI REST API for web applications
- **Performance Monitoring**: Metrics collection and analysis dashboard

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
 
