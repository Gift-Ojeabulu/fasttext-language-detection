"""
Real Data Example with Hugging Face Dataset
From: Step 4 - Working with Real Data from Hugging Face

Demonstrates language detection on real PAWS-X dataset.
"""

from datasets import load_dataset
from language_detector import SimpleLanguageDetector

def load_multilingual_sample():
    """Load sample data from PAWS-X dataset"""
    try:
        # Load PAWS-X dataset samples
        dataset = load_dataset("paws-x", "en", split="train[:50]")
        english_texts = [item['sentence1'] for item in dataset]
        
        dataset_es = load_dataset("paws-x", "es", split="train[:50]")
        spanish_texts = [item['sentence1'] for item in dataset_es]
        
        dataset_fr = load_dataset("paws-x", "fr", split="train[:50]")
        french_texts = [item['sentence1'] for item in dataset_fr]
        
        # Combine samples
        sample_data = []
        
        # Take first 10 from each language
        for text in english_texts[:10]:
            sample_data.append({"text": text, "true_language": "en"})
            
        for text in spanish_texts[:10]:
            sample_data.append({"text": text, "true_language": "es"})
            
        for text in french_texts[:10]:
            sample_data.append({"text": text, "true_language": "fr"})
        
        return sample_data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to manual examples
        return [
            {"text": "The quick brown fox jumps over the lazy dog.", "true_language": "en"},
            {"text": "El zorro marrón rápido salta sobre el perro perezoso.", "true_language": "es"},
            {"text": "Le renard brun rapide saute par-dessus le chien paresseux.", "true_language": "fr"}
        ]

def main():
    """Real data processing example"""
    print("Real Data Example - PAWS-X Dataset")
    print("=" * 40)
    
    # Initialize detector
    detector = SimpleLanguageDetector(confidence_threshold=0.7)
    
    # Load real data
    print("Loading multilingual data from Hugging Face...")
    sample_data = load_multilingual_sample()
    print(f"Loaded {len(sample_data)} text samples")
    
    # Process the real data
    results = []
    correct_predictions = 0
    
    print("\nProcessing real multilingual data...")
    print("-" * 50)
    
    for i, item in enumerate(sample_data):
        result = detector.detect(item['text'])
        result['true_language'] = item['true_language']
        result['correct'] = result['language'] == item['true_language']
        
        if result['correct']:
            correct_predictions += 1
        
        results.append(result)
        
        # Show first few results
        if i < 5:
            print(f"Text: '{item['text'][:60]}...'")
            print(f"True: {detector._get_language_name(item['true_language'])}")
            print(f"Predicted: {result['language_name']} ({result['confidence']:.3f})")
            print(f"Correct: {result['correct']}\n")
    
    accuracy = correct_predictions / len(results)
    print(f"Final Accuracy: {accuracy:.2%} ({correct_predictions}/{len(results)})")

if __name__ == "__main__":
    main()
