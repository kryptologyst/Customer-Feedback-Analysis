"""Data schemas and validation for customer feedback analysis."""

from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
import pandera as pa
from pandera.typing import Series


class FeedbackSchema(pa.DataFrameModel):
    """Schema for customer feedback data."""
    
    feedback_id: Series[str] = pa.Field(description="Unique identifier for feedback")
    customer_id: Series[str] = pa.Field(description="Anonymized customer identifier")
    feedback_text: Series[str] = pa.Field(
        min_length=1,
        max_length=1000,
        description="Customer feedback text"
    )
    category: Series[str] = pa.Field(
        isin=["product_quality", "customer_service", "shipping_delivery", "website_app", "pricing_value"],
        description="Feedback category"
    )
    sentiment_label: Series[str] = pa.Field(
        isin=["positive", "neutral", "negative"],
        description="Ground truth sentiment label"
    )
    timestamp: Series[datetime] = pa.Field(description="When feedback was received")
    channel: Series[str] = pa.Field(
        isin=["email", "survey", "review", "support_ticket", "social_media"],
        description="Channel where feedback was received"
    )
    rating: Series[Optional[int]] = pa.Field(
        ge=1,
        le=5,
        nullable=True,
        description="Optional numeric rating (1-5)"
    )
    response_time_hours: Series[Optional[float]] = pa.Field(
        ge=0,
        nullable=True,
        description="Time to respond to feedback in hours"
    )
    escalated: Series[bool] = pa.Field(description="Whether feedback was escalated")
    
    class Config:
        """Pandera configuration."""
        coerce = True


class CustomerSchema(pa.DataFrameModel):
    """Schema for customer data."""
    
    customer_id: Series[str] = pa.Field(description="Anonymized customer identifier")
    segment: Series[str] = pa.Field(
        isin=["premium", "standard", "basic"],
        description="Customer segment"
    )
    registration_date: Series[datetime] = pa.Field(description="Customer registration date")
    total_orders: Series[int] = pa.Field(ge=0, description="Total number of orders")
    total_spent: Series[float] = pa.Field(ge=0, description="Total amount spent")
    last_order_date: Series[Optional[datetime]] = pa.Field(
        nullable=True,
        description="Date of last order"
    )
    
    class Config:
        """Pandera configuration."""
        coerce = True


def validate_feedback_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate feedback data against schema."""
    try:
        return FeedbackSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Data validation failed: {e}")


def validate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate customer data against schema."""
    try:
        return CustomerSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Data validation failed: {e}")
