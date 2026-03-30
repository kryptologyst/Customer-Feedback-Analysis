"""Test cases for customer feedback analysis."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.synthetic_data import SyntheticFeedbackData
from src.data.schemas import FeedbackSchema, CustomerSchema, validate_feedback_data
from src.models.sentiment_models import BaselineSentimentModel
from src.utils.seed import set_seed


class TestSyntheticData:
    """Test synthetic data generation."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        generator = SyntheticFeedbackData(n_samples=100)
        assert generator.n_samples == 100
        assert len(generator.categories) == 5
        assert generator.sentiment_distribution["positive"] == 0.4
    
    def test_feedback_data_generation(self):
        """Test feedback data generation."""
        set_seed(42)
        generator = SyntheticFeedbackData(n_samples=50)
        df = generator.generate_feedback_data()
        
        assert len(df) == 50
        assert "feedback_text" in df.columns
        assert "sentiment_label" in df.columns
        assert "category" in df.columns
        
        # Check sentiment distribution
        sentiment_counts = df["sentiment_label"].value_counts()
        assert len(sentiment_counts) == 3  # positive, neutral, negative
    
    def test_customer_data_generation(self):
        """Test customer data generation."""
        set_seed(42)
        generator = SyntheticFeedbackData(n_samples=50)
        feedback_df = generator.generate_feedback_data()
        customer_df = generator.generate_customer_data()
        
        assert len(customer_df) > 0
        assert "customer_id" in customer_df.columns
        assert "segment" in customer_df.columns
        
        # Check that all customers from feedback are in customer data
        feedback_customers = set(feedback_df["customer_id"].unique())
        customer_customers = set(customer_df["customer_id"].unique())
        assert feedback_customers.issubset(customer_customers)


class TestDataSchemas:
    """Test data validation schemas."""
    
    def test_feedback_schema_validation(self):
        """Test feedback schema validation."""
        valid_data = {
            "feedback_id": ["FB_001", "FB_002"],
            "customer_id": ["CUST_001", "CUST_002"],
            "feedback_text": ["Great service!", "Poor quality."],
            "category": ["customer_service", "product_quality"],
            "sentiment_label": ["positive", "negative"],
            "timestamp": [datetime.now(), datetime.now()],
            "channel": ["email", "survey"],
            "rating": [5, 2],
            "response_time_hours": [2.5, 24.0],
            "escalated": [False, True]
        }
        
        df = pd.DataFrame(valid_data)
        validated_df = validate_feedback_data(df)
        assert len(validated_df) == 2
    
    def test_feedback_schema_invalid_data(self):
        """Test feedback schema with invalid data."""
        invalid_data = {
            "feedback_id": ["FB_001"],
            "customer_id": ["CUST_001"],
            "feedback_text": [""],  # Empty text should fail
            "category": ["invalid_category"],  # Invalid category
            "sentiment_label": ["positive"],
            "timestamp": [datetime.now()],
            "channel": ["email"],
            "rating": [5],
            "response_time_hours": [2.5],
            "escalated": [False]
        }
        
        df = pd.DataFrame(invalid_data)
        with pytest.raises(ValueError):
            validate_feedback_data(df)


class TestBaselineModels:
    """Test baseline sentiment analysis models."""
    
    def test_textblob_model(self):
        """Test TextBlob baseline model."""
        model = BaselineSentimentModel(method="textblob")
        
        test_texts = [
            "Great service!",
            "Terrible experience.",
            "It was okay."
        ]
        
        predictions = model.predict(test_texts)
        assert len(predictions) == 3
        assert all(pred in ["positive", "neutral", "negative"] for pred in predictions)
    
    def test_vader_model(self):
        """Test VADER baseline model."""
        model = BaselineSentimentModel(method="vader")
        
        test_texts = [
            "This is amazing!",
            "This is terrible.",
            "This is fine."
        ]
        
        predictions = model.predict(test_texts)
        assert len(predictions) == 3
        assert all(pred in ["positive", "neutral", "negative"] for pred in predictions)
    
    def test_naive_bayes_model(self):
        """Test Naive Bayes baseline model."""
        model = BaselineSentimentModel(method="naive_bayes")
        
        # Training data
        train_texts = [
            "Great service!", "Amazing product!", "Love it!",
            "Terrible experience.", "Poor quality.", "Hate it.",
            "It's okay.", "Average.", "Not bad."
        ]
        train_labels = [
            "positive", "positive", "positive",
            "negative", "negative", "negative",
            "neutral", "neutral", "neutral"
        ]
        
        # Train model
        model.train(train_texts, train_labels)
        
        # Test predictions
        test_texts = ["Great!", "Terrible!", "Okay."]
        predictions = model.predict(test_texts)
        
        assert len(predictions) == 3
        assert all(pred in ["positive", "neutral", "negative"] for pred in predictions)


class TestUtilities:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers to test reproducibility
        np.random.seed(42)
        first_run = np.random.random(5)
        
        set_seed(42)
        np.random.seed(42)
        second_run = np.random.random(5)
        
        np.testing.assert_array_equal(first_run, second_run)


if __name__ == "__main__":
    pytest.main([__file__])
