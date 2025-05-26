#!/usr/bin/env python3
"""
Test script to verify all components are working
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pytesseract

def test_environment():
    """Test if environment is properly set up."""
    print("Testing Environment Setup...")
    
    # Test 1: Environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[FAIL] OPENAI_API_KEY not found in .env file")
        return False
    else:
        print("[PASS] OPENAI_API_KEY found")
    
    # Test 2: OpenAI connection
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        print(f"[PASS] OpenAI connection successful ({len(models.data)} models available)")
    except Exception as e:
        print(f"[FAIL] OpenAI connection failed: {e}")
        return False
    
    # Test 3: Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[PASS] Tesseract found (version: {version})")
    except Exception as e:
        print(f"[FAIL] Tesseract not found: {e}")
        return False
    
    # Test 4: Required packages
    required_packages = [
        'numpy', 'faiss', 'PyPDF2', 'PIL', 'pdf2image'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[PASS] {package} imported successfully")
        except ImportError:
            print(f"[FAIL] {package} not found")
            return False
    
    return True

def test_sample_embedding():
    """Test creating a simple embedding."""
    print("\nTesting Embedding Creation...")
    
    load_dotenv()
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input="This is a test sentence for embedding.",
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        print(f"[PASS] Embedding created successfully (dimension: {len(embedding)})")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding creation failed: {e}")
        return False

def test_sample_chat():
    """Test OpenAI chat completion."""
    print("\nTesting Chat Completion...")
    
    load_dotenv()
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using cheaper model for testing
            messages=[
                {"role": "user", "content": "Say 'Hello, RAG system test successful!'"}
            ],
            max_tokens=50
        )
        answer = response.choices[0].message.content
        print(f"[PASS] Chat completion successful: {answer}")
        return True
    except Exception as e:
        print(f"[FAIL] Chat completion failed: {e}")
        return False

def check_sample_pdf():
    """Check if sample PDF exists."""
    print("\nChecking for Sample PDF...")
    
    sample_paths = [
        "input.pdf",
        "test.pdf",
        "sample.pdf"
    ]
    
    for path in sample_paths:
        if os.path.exists(path):
            print(f"[PASS] Found sample PDF: {path}")
            return path
    
    print("[INFO] No sample PDF found. You can:")
    print("   1. Add a PDF file named 'input.pdf' to test with")
    print("   2. Use --pdf flag to specify a different PDF")
    return None

def main():
    print("RAG PDF Analyzer - System Test")
    print("=" * 40)
    
    all_tests_passed = True
    
    # Run tests
    all_tests_passed &= test_environment()
    all_tests_passed &= test_sample_embedding()
    all_tests_passed &= test_sample_chat()
    
    sample_pdf = check_sample_pdf()
    
    print("\n" + "=" * 40)
    if all_tests_passed:
        print("[SUCCESS] All tests passed! Your system is ready.")
        if sample_pdf:
            print(f"You can now run: python rag_analyzer.py --pdf {sample_pdf}")
        else:
            print("Add a PDF file and run: python rag_analyzer.py --pdf your_file.pdf")
    else:
        print("[FAILED] Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()