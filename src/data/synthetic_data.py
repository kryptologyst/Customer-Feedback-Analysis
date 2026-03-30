"""Synthetic data generation for customer feedback analysis."""

import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

try:
    from faker import Faker
except ImportError:
    # Fallback if faker is not available
    class Faker:
        def __init__(self):
            pass
        def sentence(self):
            return "This is a sample sentence."
        def word(self):
            return "sample"

from .schemas import FeedbackSchema, CustomerSchema


class SyntheticFeedbackData:
    """Generate synthetic customer feedback data for testing and demonstration."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_categories: int = 5,
        sentiment_distribution: Dict[str, float] = None,
        min_length: int = 10,
        max_length: int = 200,
        avg_length: int = 50,
        categories: List[str] = None,
        customer_attributes: Dict[str, Any] = None,
        seed: int = 42
    ):
        """Initialize synthetic data generator.
        
        Args:
            n_samples: Number of feedback samples to generate
            n_categories: Number of feedback categories
            sentiment_distribution: Distribution of sentiment labels
            min_length: Minimum text length
            max_length: Maximum text length
            avg_length: Average text length
            categories: List of feedback categories
            customer_attributes: Customer attribute configuration
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_categories = n_categories
        self.sentiment_distribution = sentiment_distribution or {
            "positive": 0.4,
            "neutral": 0.3,
            "negative": 0.3
        }
        self.min_length = min_length
        self.max_length = max_length
        self.avg_length = avg_length
        self.categories = categories or [
            "product_quality",
            "customer_service", 
            "shipping_delivery",
            "website_app",
            "pricing_value"
        ]
        self.customer_attributes = customer_attributes or {
            "n_segments": 3,
            "include_demographics": True,
            "anonymize_data": True
        }
        self.seed = seed
        
        # Initialize Faker
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # Define feedback templates
        self._setup_feedback_templates()
        
    def _setup_feedback_templates(self) -> None:
        """Setup feedback templates for different categories and sentiments."""
        self.feedback_templates = {
            "product_quality": {
                "positive": [
                    "The product quality is excellent!",
                    "Really impressed with the build quality.",
                    "High-quality materials and craftsmanship.",
                    "The product exceeded my expectations.",
                    "Outstanding quality for the price."
                ],
                "neutral": [
                    "The product quality is okay.",
                    "Quality seems average for this price range.",
                    "Nothing special but not bad either.",
                    "The quality is acceptable.",
                    "Standard quality, as expected."
                ],
                "negative": [
                    "Poor product quality, very disappointed.",
                    "The quality is terrible for the price.",
                    "Cheap materials, broke after a few uses.",
                    "Quality control seems lacking.",
                    "Not worth the money, poor construction."
                ]
            },
            "customer_service": {
                "positive": [
                    "Customer service was amazing!",
                    "Very helpful and friendly support team.",
                    "Quick response and resolved my issue.",
                    "Excellent customer service experience.",
                    "Support team went above and beyond."
                ],
                "neutral": [
                    "Customer service was okay.",
                    "Average support experience.",
                    "Got the help I needed eventually.",
                    "Standard customer service.",
                    "Support was adequate."
                ],
                "negative": [
                    "Terrible customer service experience.",
                    "Very unhelpful support team.",
                    "Took forever to get a response.",
                    "Customer service needs improvement.",
                    "Poor support, very frustrating."
                ]
            },
            "shipping_delivery": {
                "positive": [
                    "Fast and reliable shipping!",
                    "Package arrived early and in perfect condition.",
                    "Excellent delivery service.",
                    "Quick shipping, well packaged.",
                    "Delivery exceeded expectations."
                ],
                "neutral": [
                    "Shipping was okay.",
                    "Standard delivery time.",
                    "Package arrived as expected.",
                    "Average shipping experience.",
                    "Delivery was fine."
                ],
                "negative": [
                    "Shipping was very slow.",
                    "Package arrived damaged.",
                    "Poor delivery service.",
                    "Late delivery, no communication.",
                    "Terrible shipping experience."
                ]
            },
            "website_app": {
                "positive": [
                    "Great website, easy to navigate!",
                    "Love the app interface.",
                    "Website is user-friendly and fast.",
                    "Excellent mobile app experience.",
                    "Intuitive design and functionality."
                ],
                "neutral": [
                    "Website is okay.",
                    "App works fine.",
                    "Average user experience.",
                    "Website is functional.",
                    "App does what it needs to do."
                ],
                "negative": [
                    "Website is confusing and slow.",
                    "App crashes frequently.",
                    "Poor user interface design.",
                    "Website needs major improvements.",
                    "Terrible app experience."
                ]
            },
            "pricing_value": {
                "positive": [
                    "Great value for money!",
                    "Very reasonable pricing.",
                    "Excellent price-to-quality ratio.",
                    "Good value proposition.",
                    "Fair pricing for what you get."
                ],
                "neutral": [
                    "Pricing is okay.",
                    "Average value for money.",
                    "Price seems reasonable.",
                    "Standard pricing.",
                    "Value is acceptable."
                ],
                "negative": [
                    "Too expensive for what you get.",
                    "Poor value for money.",
                    "Overpriced compared to competitors.",
                    "Not worth the price.",
                    "Expensive and disappointing."
                ]
            }
        }
    
    def _generate_feedback_text(
        self, 
        category: str, 
        sentiment: str, 
        length_variance: float = 0.3
    ) -> str:
        """Generate feedback text based on category and sentiment."""
        templates = self.feedback_templates.get(category, {}).get(sentiment, [])
        
        if not templates:
            # Fallback to generic templates
            templates = [
                f"This is a {sentiment} experience with {category.replace('_', ' ')}.",
                f"I have {sentiment} feelings about the {category.replace('_', ' ')}.",
                f"The {category.replace('_', ' ')} was {sentiment}."
            ]
        
        base_text = random.choice(templates)
        
        # Add variation to text length
        target_length = int(self.avg_length * (1 + random.uniform(-length_variance, length_variance)))
        target_length = max(self.min_length, min(self.max_length, target_length))
        
        # Extend text if needed
        while len(base_text) < target_length:
            extensions = [
                f" {self.fake.sentence().lower()}",
                f" {self.fake.word()}",
                f" Really {self.fake.word()}.",
                f" Very {self.fake.word()} indeed."
            ]
            base_text += random.choice(extensions)
        
        # Truncate if too long
        if len(base_text) > target_length:
            base_text = base_text[:target_length].rsplit(' ', 1)[0] + "."
        
        return base_text.strip()
    
    def _anonymize_customer_id(self, customer_id: str) -> str:
        """Anonymize customer ID using hashing."""
        if self.customer_attributes.get("anonymize_data", True):
            return hashlib.sha256(customer_id.encode()).hexdigest()[:12]
        return customer_id
    
    def generate_feedback_data(self) -> pd.DataFrame:
        """Generate synthetic feedback data."""
        data = []
        
        for i in range(self.n_samples):
            # Generate customer ID
            customer_id = f"CUST_{i:06d}"
            customer_id = self._anonymize_customer_id(customer_id)
            
            # Sample category and sentiment
            category = random.choice(self.categories)
            sentiment = np.random.choice(
                list(self.sentiment_distribution.keys()),
                p=list(self.sentiment_distribution.values())
            )
            
            # Generate feedback text
            feedback_text = self._generate_feedback_text(category, sentiment)
            
            # Generate timestamp (last 6 months)
            days_ago = random.randint(1, 180)
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            # Generate channel
            channel = random.choice([
                "email", "survey", "review", "support_ticket", "social_media"
            ])
            
            # Generate rating (correlated with sentiment)
            if sentiment == "positive":
                rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
            elif sentiment == "neutral":
                rating = random.choices([3, 4], weights=[0.7, 0.3])[0]
            else:
                rating = random.choices([1, 2], weights=[0.7, 0.3])[0]
            
            # Generate response time (correlated with sentiment and channel)
            if channel in ["support_ticket", "email"]:
                if sentiment == "negative":
                    response_time = random.uniform(2, 48)  # Slower for negative
                else:
                    response_time = random.uniform(1, 24)
            else:
                response_time = None
            
            # Generate escalation (more likely for negative sentiment)
            escalated = sentiment == "negative" and random.random() < 0.3
            
            data.append({
                "feedback_id": f"FB_{i:06d}",
                "customer_id": customer_id,
                "feedback_text": feedback_text,
                "category": category,
                "sentiment_label": sentiment,
                "timestamp": timestamp,
                "channel": channel,
                "rating": rating,
                "response_time_hours": response_time,
                "escalated": escalated
            })
        
        df = pd.DataFrame(data)
        return FeedbackSchema.validate(df)
    
    def generate_customer_data(self) -> pd.DataFrame:
        """Generate synthetic customer data."""
        data = []
        unique_customers = set()
        
        # Get unique customers from feedback data
        feedback_df = self.generate_feedback_data()
        unique_customers = set(feedback_df["customer_id"].unique())
        
        segments = ["premium", "standard", "basic"]
        segment_weights = [0.2, 0.5, 0.3]
        
        for customer_id in unique_customers:
            # Generate customer attributes
            segment = np.random.choice(segments, p=segment_weights)
            
            # Registration date (last 2 years)
            days_ago = random.randint(30, 730)
            registration_date = datetime.now() - timedelta(days=days_ago)
            
            # Generate order history based on segment
            if segment == "premium":
                total_orders = random.randint(20, 100)
                total_spent = random.uniform(2000, 10000)
            elif segment == "standard":
                total_orders = random.randint(5, 30)
                total_spent = random.uniform(200, 2000)
            else:
                total_orders = random.randint(1, 10)
                total_spent = random.uniform(50, 500)
            
            # Last order date
            if total_orders > 0:
                days_since_last = random.randint(1, 90)
                last_order_date = datetime.now() - timedelta(days=days_since_last)
            else:
                last_order_date = None
            
            data.append({
                "customer_id": customer_id,
                "segment": segment,
                "registration_date": registration_date,
                "total_orders": total_orders,
                "total_spent": round(total_spent, 2),
                "last_order_date": last_order_date
            })
        
        df = pd.DataFrame(data)
        return CustomerSchema.validate(df)
    
    def save_data(self, output_dir: str = "data") -> Dict[str, str]:
        """Generate and save synthetic data."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data
        feedback_df = self.generate_feedback_data()
        customer_df = self.generate_customer_data()
        
        # Save to files
        feedback_path = os.path.join(output_dir, "feedback_synthetic.csv")
        customer_path = os.path.join(output_dir, "customers_synthetic.csv")
        
        feedback_df.to_csv(feedback_path, index=False)
        customer_df.to_csv(customer_path, index=False)
        
        return {
            "feedback": feedback_path,
            "customers": customer_path
        }
