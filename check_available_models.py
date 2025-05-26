#!/usr/bin/env python3
"""
Check which OpenAI models are available to your account
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get all models
models = client.models.list()

# Filter for chat models
chat_models = [model for model in models.data if 'gpt' in model.id.lower() or 'o1' in model.id.lower() or 'o3' in model.id.lower() or 'o4' in model.id.lower()]

# Filter for embedding models  
embedding_models = [model for model in models.data if 'embedding' in model.id.lower()]

print("AVAILABLE CHAT MODELS:")
print("=" * 40)
for model in sorted(chat_models, key=lambda x: x.id):
    print(f"[PASS] {model.id}")

print("\nAVAILABLE EMBEDDING MODELS:")
print("=" * 40)
for model in sorted(embedding_models, key=lambda x: x.id):
    print(f"[PASS] {model.id}")

print(f"\nTOTAL MODELS AVAILABLE: {len(models.data)}")

# Test specific models we're interested in
test_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo', 'o3', 'o4-mini', 'text-embedding-3-large']

print("\nTESTING SPECIFIC MODELS:")
print("=" * 40)
available_model_ids = [model.id for model in models.data]

for model_name in test_models:
    if model_name in available_model_ids:
        print(f"[PASS] {model_name} - AVAILABLE")
    else:
        print(f"[FAIL] {model_name} - NOT AVAILABLE")