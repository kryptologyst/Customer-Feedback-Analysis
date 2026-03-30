"""Main training script for customer feedback analysis."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

from src.data.synthetic_data import SyntheticFeedbackData
from src.models.sentiment_models import BERTSentimentModel, BaselineSentimentModel
from src.eval.comprehensive_evaluator import ComprehensiveEvaluator
from src.utils.seed import set_seed
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def generate_data(config: Dict[str, Any]) -> Dict[str, str]:
    """Generate synthetic data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with data file paths
    """
    logger.info("Generating synthetic data...")
    
    data_config = config["data"]
    data_generator = SyntheticFeedbackData(
        n_samples=data_config["n_samples"],
        n_categories=data_config["n_categories"],
        sentiment_distribution=data_config["sentiment_distribution"],
        min_length=data_config["min_length"],
        max_length=data_config["max_length"],
        avg_length=data_config["avg_length"],
        categories=data_config["categories"],
        customer_attributes=data_config["customer_attributes"],
        seed=config["seed"]
    )
    
    # Generate and save data
    data_paths = data_generator.save_data(config["paths"]["data_dir"])
    
    logger.info(f"Data generated and saved to: {data_paths}")
    return data_paths


def train_model(config: Dict[str, Any], data_paths: Dict[str, str]) -> Any:
    """Train sentiment analysis model.
    
    Args:
        config: Configuration dictionary
        data_paths: Dictionary with data file paths
        
    Returns:
        Trained model
    """
    logger.info("Training sentiment analysis model...")
    
    # Load data
    feedback_df = pd.read_csv(data_paths["feedback"])
    logger.info(f"Loaded {len(feedback_df)} feedback samples")
    
    # Initialize model
    model_config = config["model"]
    model = BERTSentimentModel(
        model_name=model_config["model_name"],
        num_labels=model_config["num_labels"],
        max_length=model_config["max_length"],
        dropout_rate=model_config["dropout_rate"],
        seed=config["seed"]
    )
    
    # Prepare data
    train_loader, val_loader, test_loader = model.prepare_data(
        feedback_df,
        test_size=config["evaluation"]["test_size"],
        val_size=config["evaluation"]["val_size"],
        random_state=config["evaluation"]["random_state"]
    )
    
    # Train model
    trainer = model.train(
        train_loader,
        val_loader,
        learning_rate=model_config["learning_rate"],
        num_epochs=model_config["num_epochs"],
        warmup_steps=model_config["warmup_steps"],
        weight_decay=model_config["weight_decay"],
        save_best_model=model_config["save_best_model"],
        output_dir=f"{config['paths']['models_dir']}/bert_sentiment"
    )
    
    logger.info("Model training completed")
    return model, test_loader


def evaluate_model(
    model: Any, 
    test_loader: Any, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate trained model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating model...")
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        metrics=config["evaluation"]["metrics"],
        business_metrics=config["business_metrics"],
        cross_validation=config["evaluation"].get("cross_validation", {}),
        statistical_tests=config["evaluation"].get("statistical_tests", {}),
        model_comparison=config["evaluation"].get("model_comparison", {}),
        business_impact=config["evaluation"].get("business_impact", {})
    )
    
    # Evaluate model
    results = model.evaluate(test_loader)
    
    # Generate evaluation report
    report_path = evaluator.generate_evaluation_report(
        {"bert_model": results},
        config["paths"]["assets_dir"]
    )
    
    logger.info(f"Evaluation completed. Report saved to: {report_path}")
    return results


def compare_baseline_models(
    config: Dict[str, Any], 
    data_paths: Dict[str, str]
) -> Dict[str, Any]:
    """Compare with baseline models.
    
    Args:
        config: Configuration dictionary
        data_paths: Dictionary with data file paths
        
    Returns:
        Comparison results
    """
    logger.info("Comparing with baseline models...")
    
    # Load data
    feedback_df = pd.read_csv(data_paths["feedback"])
    
    # Prepare data for baseline models
    texts = feedback_df['feedback_text'].tolist()
    labels = feedback_df['sentiment_label'].tolist()
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()
    
    # Train and evaluate baseline models
    baseline_results = []
    
    for method in ["textblob", "vader", "naive_bayes"]:
        logger.info(f"Training {method} baseline model...")
        
        model = BaselineSentimentModel(method=method)
        
        if method == "naive_bayes":
            model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
        
        baseline_results.append({
            "model_name": method,
            "accuracy": accuracy,
            "predictions": predictions
        })
    
    # Compare models
    comparison_results = evaluator.compare_models(baseline_results)
    
    logger.info("Baseline model comparison completed")
    return comparison_results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train customer feedback analysis model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Skip data generation step"
    )
    parser.add_argument(
        "--skip-baseline-comparison",
        action="store_true",
        help="Skip baseline model comparison"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seeds
    set_seed(config["seed"])
    
    # Create output directories
    for path_key in ["data_dir", "models_dir", "assets_dir", "logs_dir"]:
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate data
        if not args.skip_data_generation:
            data_paths = generate_data(config)
        else:
            # Assume data already exists
            data_paths = {
                "feedback": f"{config['paths']['data_dir']}/feedback_synthetic.csv",
                "customers": f"{config['paths']['data_dir']}/customers_synthetic.csv"
            }
        
        # Train main model
        model, test_loader = train_model(config, data_paths)
        
        # Evaluate main model
        main_results = evaluate_model(model, test_loader, config)
        
        # Compare with baseline models
        if not args.skip_baseline_comparison:
            baseline_results = compare_baseline_models(config, data_paths)
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Model: {config['model']['model_name']}")
        print(f"Accuracy: {main_results['accuracy']:.4f}")
        print(f"F1 Macro: {main_results['f1_macro']:.4f}")
        print(f"Precision Macro: {main_results['precision_macro']:.4f}")
        print(f"Recall Macro: {main_results['recall_macro']:.4f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
