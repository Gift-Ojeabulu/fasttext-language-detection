"""
Batch Processing Example
From: Step 5 - Using Language Detector Class (Batch Processing)

Demonstrates processing multiple texts efficiently.
"""

from language_detector import SimpleLanguageDetector

def main():
    """Batch processing example"""
    print("Batch Processing Example")
    print("=" * 40)
    
    # Initialize detector
    detector = SimpleLanguageDetector(confidence_threshold=0.6)
    
    # Sample batch data
    sample_texts = [
        "Hello world, this is English text.",
        "Bonjour le monde, ceci est du français.",
        "Hola mundo, esto es español.",
        "Hallo Welt, das ist deutscher Text.", 
        "Ciao mondo, questo è italiano.",
        "Hi there!",
        "Short",
        "This is a longer English text that should be detected with high confidence.",
        "Je suis un texte français plus long qui devrait être détecté avec une grande confiance.",
        "Soy un texto español más largo que debería detectarse con alta confianza."
    ]
    
    print(f"Processing {len(sample_texts)} texts...")
    print("-" * 40)
    
    # Process batch
    batch_results = detector.detect_batch(sample_texts, show_progress=True)
    
    # Get summary
    summary = detector.get_summary(batch_results)
    
    print(f"\nBatch Processing Summary:")
    print(f"- Total texts: {summary['total_texts']}")
    print(f"- Successful detections: {summary['successful_detections']}")
    print(f"- Success rate: {summary['success_rate']:.2%}")
    print(f"- Languages found: {list(summary['language_distribution'].keys())}")
    
    print(f"\nLanguage Distribution:")
    for lang, count in summary['language_distribution'].items():
        print(f"  {lang}: {count}")

if __name__ == "__main__":
    main()
