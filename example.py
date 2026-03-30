#!/usr/bin/env python3
"""Example script demonstrating customer feedback analysis."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.synthetic_data import SyntheticFeedbackData
from src.models.sentiment_models import BaselineSentimentModel
from src.utils.seed import set_seed


def main():
    """Run a simple example of customer feedback analysis."""
    print("Customer Feedback Analysis - Example")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Generate synthetic data
    print("1. Generating synthetic feedback data...")
    data_generator = SyntheticFeedbackData(n_samples=100)
    feedback_df = data_generator.generate_feedback_data()
    print(f"   Generated {len(feedback_df)} feedback samples")
    
    # Show data overview
    print("\n2. Data Overview:")
    print(f"   Categories: {feedback_df['category'].nunique()}")
    print(f"   Channels: {feedback_df['channel'].nunique()}")
    print(f"   Sentiment distribution:")
    sentiment_counts = feedback_df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"     {sentiment}: {count} ({count/len(feedback_df):.1%})")
    
    # Test baseline models
    print("\n3. Testing baseline sentiment analysis models...")
    
    # Sample texts for testing
    test_texts = [
        "Great service and fast delivery!",
        "The product quality was terrible.",
        "Average experience, nothing special.",
        "Very helpful customer support.",
        "The item arrived late and was damaged."
    ]
    
    # Test different baseline models
    models = ["textblob", "vader"]
    
    for model_name in models:
        print(f"\n   Testing {model_name.upper()} model:")
        model = BaselineSentimentModel(method=model_name)
        predictions = model.predict(test_texts)
        
        for text, prediction in zip(test_texts, predictions):
            print(f"     '{text[:30]}...' -> {prediction}")
    
    # Show sample data
    print("\n4. Sample feedback data:")
    print(feedback_df[['feedback_text', 'category', 'sentiment_label', 'rating']].head())
    
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print("\nTo run the full system:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train models: python scripts/train.py")
    print("3. Run demo: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
