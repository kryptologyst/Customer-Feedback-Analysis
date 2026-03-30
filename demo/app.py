"""Streamlit demo application for customer feedback analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.synthetic_data import SyntheticFeedbackData
from src.models.sentiment_models import BERTSentimentModel, BaselineSentimentModel
from src.eval.comprehensive_evaluator import ComprehensiveEvaluator
from src.utils.seed import set_seed

# Page configuration
st.set_page_config(
    page_title="Customer Feedback Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'evaluation_complete' not in st.session_state:
    st.session_state.evaluation_complete = False


def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">Customer Feedback Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This is a research and educational tool only.</strong> 
        The sentiment analysis results should not be used for automated decision-making 
        without human review. Always validate predictions with domain experts and 
        consider the context of individual feedback before taking action.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Data generation parameters
        st.subheader("Data Generation")
        n_samples = st.slider("Number of samples", 100, 2000, 1000)
        sentiment_pos = st.slider("Positive sentiment %", 0.1, 0.8, 0.4)
        sentiment_neu = st.slider("Neutral sentiment %", 0.1, 0.8, 0.3)
        sentiment_neg = 1.0 - sentiment_pos - sentiment_neu
        
        if sentiment_neg < 0:
            st.error("Sentiment percentages must sum to 1.0")
            return
        
        st.write(f"Negative sentiment: {sentiment_neg:.1%}")
        
        # Model parameters
        st.subheader("Model Configuration")
        model_type = st.selectbox(
            "Model Type",
            ["BERT (DistilBERT)", "TextBlob", "VADER", "Naive Bayes"]
        )
        
        if model_type == "BERT (DistilBERT)":
            max_length = st.slider("Max sequence length", 128, 512, 256)
            learning_rate = st.selectbox("Learning rate", [1e-5, 2e-5, 5e-5], index=1)
            num_epochs = st.slider("Number of epochs", 1, 10, 3)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Evaluation", 
        "🔍 Analysis", 
        "📋 Insights"
    ])
    
    with tab1:
        show_data_overview(n_samples, sentiment_pos, sentiment_neu, sentiment_neg)
    
    with tab2:
        show_model_training(model_type, max_length, learning_rate, num_epochs)
    
    with tab3:
        show_evaluation()
    
    with tab4:
        show_analysis()
    
    with tab5:
        show_insights()


def show_data_overview(n_samples, sentiment_pos, sentiment_neu, sentiment_neg):
    """Show data overview tab."""
    st.header("📊 Data Overview")
    
    if st.button("Generate Synthetic Data", key="generate_data"):
        with st.spinner("Generating synthetic feedback data..."):
            # Set seed for reproducibility
            set_seed(42)
            
            # Generate data
            data_generator = SyntheticFeedbackData(
                n_samples=n_samples,
                sentiment_distribution={
                    "positive": sentiment_pos,
                    "neutral": sentiment_neu,
                    "negative": sentiment_neg
                }
            )
            
            feedback_df = data_generator.generate_feedback_data()
            customer_df = data_generator.generate_customer_data()
            
            # Store in session state
            st.session_state.feedback_df = feedback_df
            st.session_state.customer_df = customer_df
            st.session_state.data_generated = True
            
            st.success(f"Generated {len(feedback_df)} feedback samples and {len(customer_df)} customer records!")
    
    if st.session_state.data_generated:
        feedback_df = st.session_state.feedback_df
        customer_df = st.session_state.customer_df
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Feedback", len(feedback_df))
        
        with col2:
            st.metric("Unique Customers", feedback_df['customer_id'].nunique())
        
        with col3:
            st.metric("Categories", feedback_df['category'].nunique())
        
        with col4:
            st.metric("Channels", feedback_df['channel'].nunique())
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = feedback_df['sentiment_label'].value_counts()
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'neutral': '#ffc107', 
                    'negative': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=sentiment_counts.index,
                y=sentiment_counts.values,
                title="Sentiment Counts",
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#28a745',
                    'neutral': '#ffc107',
                    'negative': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Category analysis
        st.subheader("Feedback by Category")
        category_sentiment = pd.crosstab(feedback_df['category'], feedback_df['sentiment_label'])
        
        fig = px.bar(
            category_sentiment,
            title="Sentiment by Category",
            barmode='group'
        )
        fig.update_layout(xaxis_title="Category", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.subheader("Feedback Over Time")
        feedback_df['date'] = pd.to_datetime(feedback_df['timestamp']).dt.date
        daily_feedback = feedback_df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        
        fig = px.line(
            daily_feedback,
            x='date',
            y='count',
            color='sentiment_label',
            title="Daily Feedback Volume by Sentiment"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample data
        st.subheader("Sample Feedback Data")
        st.dataframe(feedback_df.head(10), use_container_width=True)


def show_model_training(model_type, max_length, learning_rate, num_epochs):
    """Show model training tab."""
    st.header("🤖 Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    feedback_df = st.session_state.feedback_df
    
    if st.button("Train Model", key="train_model"):
        with st.spinner("Training model..."):
            try:
                if model_type == "BERT (DistilBERT)":
                    # Initialize BERT model
                    model = BERTSentimentModel(
                        model_name="distilbert-base-uncased",
                        max_length=max_length,
                        seed=42
                    )
                    
                    # Prepare data
                    train_loader, val_loader, test_loader = model.prepare_data(
                        feedback_df, test_size=0.2, val_size=0.1
                    )
                    
                    # Train model
                    trainer = model.train(
                        train_loader,
                        val_loader,
                        learning_rate=learning_rate,
                        num_epochs=num_epochs
                    )
                    
                    # Store model and test data
                    st.session_state.model = model
                    st.session_state.test_loader = test_loader
                    
                else:
                    # Baseline models
                    baseline_method = model_type.lower().replace(" ", "_")
                    model = BaselineSentimentModel(method=baseline_method)
                    
                    # Prepare data for baseline models
                    texts = feedback_df['feedback_text'].tolist()
                    labels = feedback_df['sentiment_label'].tolist()
                    
                    if baseline_method == "naive_bayes":
                        model.train(texts, labels)
                    
                    # Store model and test data
                    st.session_state.model = model
                    st.session_state.test_texts = texts
                    st.session_state.test_labels = labels
                
                st.session_state.model_trained = True
                st.success(f"{model_type} model trained successfully!")
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")


def show_evaluation():
    """Show evaluation tab."""
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first in the Model Training tab.")
        return
    
    if st.button("Run Evaluation", key="run_evaluation"):
        with st.spinner("Running comprehensive evaluation..."):
            try:
                evaluator = ComprehensiveEvaluator()
                model = st.session_state.model
                
                if hasattr(model, 'evaluate'):
                    # BERT model evaluation
                    test_loader = st.session_state.test_loader
                    results = model.evaluate(test_loader)
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                    
                    with col2:
                        st.metric("F1 Macro", f"{results['f1_macro']:.3f}")
                    
                    with col3:
                        st.metric("Precision Macro", f"{results['precision_macro']:.3f}")
                    
                    with col4:
                        st.metric("Recall Macro", f"{results['recall_macro']:.3f}")
                    
                    # Confusion matrix
                    st.subheader("Confusion Matrix")
                    cm = results['confusion_matrix']
                    
                    fig = px.imshow(
                        cm,
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Negative", "Neutral", "Positive"],
                        y=["Negative", "Neutral", "Positive"]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification report
                    st.subheader("Detailed Classification Report")
                    report_df = pd.DataFrame(results['classification_report']).transpose()
                    st.dataframe(report_df, use_container_width=True)
                
                else:
                    # Baseline model evaluation
                    test_texts = st.session_state.test_texts
                    test_labels = st.session_state.test_labels
                    
                    predictions = model.predict(test_texts)
                    
                    # Calculate accuracy
                    accuracy = sum(1 for p, t in zip(predictions, test_labels) if p == t) / len(test_labels)
                    
                    st.metric("Accuracy", f"{accuracy:.3f}")
                    
                    # Show sample predictions
                    st.subheader("Sample Predictions")
                    sample_df = pd.DataFrame({
                        'Text': test_texts[:10],
                        'True Label': test_labels[:10],
                        'Predicted Label': predictions[:10],
                        'Correct': [p == t for p, t in zip(predictions[:10], test_labels[:10])]
                    })
                    st.dataframe(sample_df, use_container_width=True)
                
                st.session_state.evaluation_complete = True
                st.success("Evaluation completed successfully!")
                
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")


def show_analysis():
    """Show analysis tab."""
    st.header("🔍 Interactive Analysis")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first.")
        return
    
    feedback_df = st.session_state.feedback_df
    
    # Interactive text analysis
    st.subheader("Analyze Custom Text")
    
    custom_text = st.text_area(
        "Enter feedback text to analyze:",
        placeholder="Enter customer feedback here...",
        height=100
    )
    
    if custom_text and st.session_state.model_trained:
        if st.button("Analyze Text"):
            model = st.session_state.model
            
            if hasattr(model, 'predict'):
                predictions, probabilities = model.predict([custom_text])
                sentiment = predictions[0]
                confidence = max(probabilities[0])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Sentiment", sentiment.title())
                
                with col2:
                    st.metric("Confidence", f"{confidence:.3f}")
                
                # Show probability distribution
                fig = px.bar(
                    x=["Negative", "Neutral", "Positive"],
                    y=probabilities[0],
                    title="Sentiment Probabilities",
                    color=["Negative", "Neutral", "Positive"],
                    color_discrete_map={
                        'Negative': '#dc3545',
                        'Neutral': '#ffc107',
                        'Positive': '#28a745'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Category analysis
    st.subheader("Category Performance Analysis")
    
    if st.session_state.evaluation_complete:
        # This would show category-specific performance metrics
        st.info("Category-specific analysis would be implemented here with trained model results.")
    
    # Customer segment analysis
    st.subheader("Customer Segment Analysis")
    
    if 'customer_df' in st.session_state:
        customer_df = st.session_state.customer_df
        
        # Merge with feedback data
        merged_df = feedback_df.merge(customer_df, on='customer_id', how='left')
        
        # Segment analysis
        segment_sentiment = pd.crosstab(merged_df['segment'], merged_df['sentiment_label'])
        
        fig = px.bar(
            segment_sentiment,
            title="Sentiment by Customer Segment",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)


def show_insights():
    """Show insights and recommendations tab."""
    st.header("📋 Business Insights & Recommendations")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first.")
        return
    
    feedback_df = st.session_state.feedback_df
    
    # Key insights
    st.subheader("Key Insights")
    
    # Calculate insights
    total_feedback = len(feedback_df)
    positive_rate = (feedback_df['sentiment_label'] == 'positive').mean()
    negative_rate = (feedback_df['sentiment_label'] == 'negative').mean()
    escalation_rate = feedback_df['escalated'].mean()
    
    # Most common categories
    top_category = feedback_df['category'].mode()[0]
    top_channel = feedback_df['channel'].mode()[0]
    
    # Response time analysis
    response_times = feedback_df['response_time_hours'].dropna()
    avg_response_time = response_times.mean() if len(response_times) > 0 else 0
    
    # Display insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card success-metric">
            <h4>Customer Satisfaction</h4>
            <p><strong>{positive_rate:.1%}</strong> of feedback is positive</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card warning-metric">
            <h4>Areas for Improvement</h4>
            <p><strong>{negative_rate:.1%}</strong> of feedback is negative</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card danger-metric">
            <h4>Escalation Rate</h4>
            <p><strong>{escalation_rate:.1%}</strong> of feedback requires escalation</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Response Time</h4>
            <p>Average response time: <strong>{avg_response_time:.1f} hours</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.subheader("Recommendations")
    
    recommendations = []
    
    if negative_rate > 0.3:
        recommendations.append("🔴 High negative sentiment rate - investigate root causes")
    
    if escalation_rate > 0.2:
        recommendations.append("⚠️ High escalation rate - improve first-line support")
    
    if avg_response_time > 24:
        recommendations.append("⏰ Slow response times - optimize support processes")
    
    if top_category == "customer_service":
        recommendations.append("👥 Focus on customer service training and processes")
    
    if top_category == "product_quality":
        recommendations.append("🏭 Review product quality control measures")
    
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("Overall feedback looks good! Continue monitoring key metrics.")
    
    # Action items
    st.subheader("Suggested Action Items")
    
    action_items = [
        "📊 Set up automated sentiment monitoring dashboard",
        "🎯 Implement targeted follow-up for negative feedback",
        "📈 Track sentiment trends over time",
        "🔄 Establish feedback response SLA",
        "📚 Create knowledge base for common issues",
        "👥 Train support team on sentiment analysis insights"
    ]
    
    for item in action_items:
        st.write(item)
    
    # Export options
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Feedback Data"):
            csv = feedback_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"feedback_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export Insights Report"):
            # Generate insights report
            report = f"""
Customer Feedback Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- Total Feedback: {total_feedback:,}
- Positive Rate: {positive_rate:.1%}
- Negative Rate: {negative_rate:.1%}
- Escalation Rate: {escalation_rate:.1%}
- Average Response Time: {avg_response_time:.1f} hours

Top Categories:
- Most Common: {top_category}
- Most Common Channel: {top_channel}

Recommendations:
{chr(10).join(recommendations) if recommendations else "No specific recommendations at this time."}
            """
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"insights_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("Export Visualizations"):
            st.info("Visualization export feature would be implemented here.")


if __name__ == "__main__":
    main()
