"""
FastText Language Detection - Complete Working Example
From: Simple Language Detection with FastText: A Beginner's Guide

This file contains the complete workflow from model setup to real data processing.
"""

import fasttext
import pandas as pd
from datasets import load_dataset
import urllib.request
import os

# Language mapping and helper functions
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'tr': 'Turkish',
    'nl': 'Dutch'
}

def download_language_model():
    """Download FastText language detection model"""
    model_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
    model_path = 'lid.176.bin'
    
    if not os.path.exists(model_path):
        print("Downloading FastText language detection model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    
    return model_path

def clean_language_code(pred_lang):
    """FastText returns '__label__en' format, we want just 'en'"""
    return pred_lang.replace('__label__', '')

def get_language_name(code):
    return LANGUAGE_NAMES.get(code, f'Unknown ({code})')

def detect_language(text, min_confidence=0.7):
    """Detect language using FastText"""
    if not text or len(text.strip()) < 3:
        return {
            'text': text,
            'language': 'unknown',
            'language_name': 'Unknown',
            'confidence': 0.0,
            'status': 'too_short'
        }
    
    clean_text = text.replace('\n', ' ').strip()
    predictions = language_model.predict(clean_text, k=1)
    
    pred_lang = clean_language_code(predictions[0][0])
    confidence = float(predictions[1][0])
    
    status = 'success' if confidence >= min_confidence else 'low_confidence'
    
    return {
        'text': text,
        'language': pred_lang,
        'language_name': get_language_name(pred_lang),
        'confidence': confidence,
        'status': status
    }

def load_multilingual_sample():
    """Load sample data from PAWS-X dataset"""
    try:
        dataset = load_dataset("paws-x", "en", split="train[:100]")
        english_texts = [item['sentence1'] for item in dataset]
        
        dataset_es = load_dataset("paws-x", "es", split="train[:100]")
        spanish_texts = [item['sentence1'] for item in dataset_es]
        
        dataset_fr = load_dataset("paws-x", "fr", split="train[:100]")
        french_texts = [item['sentence1'] for item in dataset_fr]
        
        sample_data = []
        
        for text in english_texts[:20]:
            sample_data.append({"text": text, "true_language": "en"})
            
        for text in spanish_texts[:20]:
            sample_data.append({"text": text, "true_language": "es"})
            
        for text in french_texts[:20]:
            sample_data.append({"text": text, "true_language": "fr"})
        
        return sample_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [
            {"text": "The quick brown fox jumps over the lazy dog.", "true_language": "en"},
            {"text": "El zorro marrón rápido salta sobre el perro perezoso.", "true_language": "es"},
            {"text": "Le renard brun rapide saute par-dessus le chien paresseux.", "true_language": "fr"}
        ]

def main():
    """Main execution function"""
    global language_model
    
    # Setup
    print("Setting up FastText Language Detection")
    print("=" * 50)
    
    model_path = download_language_model()
    language_model = fasttext.load_model(model_path)
    
    # Test with sample texts
    sample_texts = [
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es Ihnen?",
        "こんにちは、元気ですか？"
    ]
    
    print("\nBasic Language Detection Results:")
    print("-" * 50)
    
    for text in sample_texts:
        result = detect_language(text)
        print(f"Text: '{text}'")
        print(f"Language: {result['language_name']} ({result['confidence']:.3f})")
        print(f"Status: {result['status']}\n")
    
    # Test with real data
    print("Loading and testing with real multilingual data...")
    sample_data = load_multilingual_sample()
    results = []
    correct_predictions = 0
    
    for item in sample_data:
        result = detect_language(item['text'])
        result['true_language'] = item['true_language']
        result['correct'] = result['language'] == item['true_language']
        
        if result['correct']:
            correct_predictions += 1
        
        results.append(result)
    
    accuracy = correct_predictions / len(results)
    print(f"\nAccuracy on real data: {accuracy:.2%} ({correct_predictions}/{len(results)})")

if __name__ == "__main__":
    main()
