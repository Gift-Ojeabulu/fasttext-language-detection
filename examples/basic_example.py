"""
Basic Language Detection Example
From: Step 2 - Basic Language Detection

Simple examples showing basic FastText language detection usage.
"""

from language_detector import SimpleLanguageDetector

def main():
    """Basic language detection examples"""
    print("Basic Language Detection Examples")
    print("=" * 40)
    
    # Initialize detector
    detector = SimpleLanguageDetector(confidence_threshold=0.6)
    
    # Sample texts in different languages
    sample_texts = [
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous?", 
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es Ihnen?",
        "こんにちは、元気ですか？",
        "OK",  # Short text
        "This is a longer English sentence that should be detected easily."
    ]
    
    print("\nDetection Results:")
    print("-" * 50)
    
    for text in sample_texts:
        result = detector.detect(text)
        print(f"Text: '{text}'")
        print(f"Language: {result['language_name']} ({result['confidence']:.3f})")
        print(f"Status: {result['status']}\n")

if __name__ == "__main__":
    main()
