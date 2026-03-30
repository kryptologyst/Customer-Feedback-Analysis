# Customer Feedback Analysis

Advanced Customer Feedback Analysis with Sentiment Analysis and Business Insights

## ⚠️ Important Disclaimer

**This is a research and educational tool only.** The sentiment analysis results should not be used for automated decision-making without human review. Always validate predictions with domain experts and consider the context of individual feedback before taking action.

## Overview

This project provides a comprehensive framework for analyzing customer feedback using advanced sentiment analysis techniques. It includes:

- **Advanced Models**: BERT-based sentiment analysis with multiple baseline comparisons
- **Business Metrics**: Customer satisfaction, escalation rates, response time analysis
- **Interactive Demo**: Streamlit dashboard for real-time analysis
- **Comprehensive Evaluation**: ML metrics, business impact analysis, and statistical testing
- **Privacy Compliance**: PII anonymization and GDPR-compliant data handling

## Features

### Sentiment Analysis Models
- **BERT (DistilBERT)**: State-of-the-art transformer-based sentiment classification
- **Baseline Models**: TextBlob, VADER, Naive Bayes for comparison
- **Multi-class Classification**: Positive, Neutral, Negative sentiment detection

### Business Analytics
- **Customer Satisfaction Scoring**: Overall satisfaction metrics and trends
- **Category Analysis**: Performance analysis by feedback category
- **Escalation Detection**: Identify feedback requiring immediate attention
- **Response Time Analysis**: Track and optimize support response times
- **Customer Segmentation**: Analyze sentiment by customer segments

### Evaluation Framework
- **ML Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Business Metrics**: Satisfaction scores, escalation rates, category performance
- **Statistical Testing**: Confidence intervals, significance tests
- **Model Comparison**: Comprehensive leaderboard with baseline comparisons
- **Cost-Benefit Analysis**: Business impact assessment with ROI calculations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Customer-Feedback-Analysis.git
cd git clone https://github.com/kryptologyst/Customer-Feedback-Analysis


# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Generate Data and Train Model

```bash
# Generate synthetic data and train model
python scripts/train.py

# Or with custom configuration
python scripts/train.py --config configs/config.yaml
```

### Run Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
customer-feedback-analysis/
├── src/                          # Source code
│   ├── data/                     # Data handling and schemas
│   │   ├── schemas.py           # Data validation schemas
│   │   └── synthetic_data.py    # Synthetic data generation
│   ├── models/                  # Model implementations
│   │   └── sentiment_models.py  # Sentiment analysis models
│   ├── eval/                    # Evaluation framework
│   │   └── comprehensive_evaluator.py
│   ├── utils/                   # Utility functions
│   │   ├── seed.py             # Reproducibility utilities
│   │   └── logging_config.py   # Logging configuration
│   └── viz/                     # Visualization utilities
├── configs/                     # Configuration files
│   ├── config.yaml             # Main configuration
│   ├── data/                   # Data configuration
│   ├── model/                  # Model configuration
│   └── evaluation/             # Evaluation configuration
├── scripts/                    # Training and utility scripts
│   └── train.py               # Main training script
├── demo/                      # Interactive demo
│   └── app.py                # Streamlit application
├── tests/                     # Unit tests
├── assets/                    # Generated outputs and reports
├── data/                      # Data storage
├── models/                    # Trained model storage
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Configuration

The project uses Hydra/OmegaConf for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration with paths, seeds, and general settings
- `configs/data/synthetic.yaml`: Data generation parameters
- `configs/model/bert_sentiment.yaml`: Model architecture and training parameters
- `configs/evaluation/comprehensive.yaml`: Evaluation metrics and testing configuration

## Data Schema

### Feedback Data
- `feedback_id`: Unique identifier
- `customer_id`: Anonymized customer identifier
- `feedback_text`: Customer feedback text (1-1000 characters)
- `category`: Feedback category (product_quality, customer_service, etc.)
- `sentiment_label`: Ground truth sentiment (positive, neutral, negative)
- `timestamp`: When feedback was received
- `channel`: Channel (email, survey, review, support_ticket, social_media)
- `rating`: Optional numeric rating (1-5)
- `response_time_hours`: Time to respond in hours
- `escalated`: Whether feedback was escalated

### Customer Data
- `customer_id`: Anonymized customer identifier
- `segment`: Customer segment (premium, standard, basic)
- `registration_date`: Customer registration date
- `total_orders`: Total number of orders
- `total_spent`: Total amount spent
- `last_order_date`: Date of last order

## Usage Examples

### Basic Training

```python
from src.data.synthetic_data import SyntheticFeedbackData
from src.models.sentiment_models import BERTSentimentModel

# Generate synthetic data
data_generator = SyntheticFeedbackData(n_samples=1000)
feedback_df = data_generator.generate_feedback_data()

# Train BERT model
model = BERTSentimentModel()
train_loader, val_loader, test_loader = model.prepare_data(feedback_df)
trainer = model.train(train_loader, val_loader)

# Evaluate model
results = model.evaluate(test_loader)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Custom Analysis

```python
from src.eval.comprehensive_evaluator import ComprehensiveEvaluator

# Initialize evaluator
evaluator = ComprehensiveEvaluator()

# Evaluate model with business metrics
results = evaluator.evaluate_model(
    model, X_test, y_test, model_name="BERT"
)

# Generate comprehensive report
report_path = evaluator.generate_evaluation_report(results)
```

## Evaluation Metrics

### Machine Learning Metrics
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Macro-averaged metrics across sentiment classes
- **ROC-AUC**: One-vs-Rest AUC for multi-class classification
- **Confusion Matrix**: Detailed classification breakdown

### Business Metrics
- **Satisfaction Score**: Overall customer satisfaction (0-1 scale)
- **Escalation Rate**: Percentage of feedback requiring escalation
- **Response Time Analysis**: Average and distribution of response times
- **Category Performance**: Sentiment analysis by feedback category
- **Customer Segment Analysis**: Sentiment trends by customer segment

### Statistical Analysis
- **Confidence Intervals**: Statistical confidence for accuracy metrics
- **Significance Testing**: Model comparison with statistical tests
- **Cross-Validation**: Robust performance estimation with k-fold CV

## Privacy and Compliance

### Data Protection
- **PII Anonymization**: Customer IDs are hashed/anonymized
- **Data Retention**: Configurable retention policies
- **GDPR Compliance**: Built-in privacy controls and data handling

### Ethical Considerations
- **Human-in-the-Loop**: All predictions require human validation
- **Bias Detection**: Fairness analysis for protected attributes
- **Transparency**: Explainable AI with SHAP analysis
- **Audit Trail**: Complete logging of model decisions

## Limitations

1. **Model Performance**: Accuracy depends on data quality and domain relevance
2. **Context Sensitivity**: Sentiment analysis may miss nuanced context
3. **Language Limitations**: Optimized for English text
4. **Temporal Drift**: Models may need retraining over time
5. **Bias Risk**: Models may inherit biases from training data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run linting
black src/ tests/
ruff check src/ tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{customer_feedback_analysis,
  title={Customer Feedback Analysis},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Customer-Feedback-Analysis}
}
```

## Support

For questions and support:
- Create an issue in the GitHub repository
- Check the documentation in the `docs/` folder
- Review the example notebooks in `notebooks/`

## Changelog

### Version 1.0.0
- Initial release with BERT-based sentiment analysis
- Comprehensive evaluation framework
- Interactive Streamlit demo
- Privacy-compliant data handling
- Business metrics and insights
# Customer-Feedback-Analysis
