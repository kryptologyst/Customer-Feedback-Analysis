"""Comprehensive evaluation framework for sentiment analysis models."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap

from ..utils.logging_config import get_logger
from ..models.sentiment_models import BERTSentimentModel, BaselineSentimentModel

logger = get_logger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive evaluation framework for sentiment analysis models."""
    
    def __init__(
        self,
        metrics: List[str] = None,
        business_metrics: List[str] = None,
        cross_validation: Dict[str, Any] = None,
        statistical_tests: Dict[str, Any] = None,
        model_comparison: Dict[str, Any] = None,
        business_impact: Dict[str, Any] = None
    ):
        """Initialize evaluator.
        
        Args:
            metrics: List of ML metrics to compute
            business_metrics: List of business metrics to compute
            cross_validation: Cross-validation configuration
            statistical_tests: Statistical testing configuration
            model_comparison: Model comparison configuration
            business_impact: Business impact analysis configuration
        """
        self.metrics = metrics or [
            "accuracy", "precision_macro", "recall_macro", 
            "f1_macro", "roc_auc_ovr", "confusion_matrix"
        ]
        self.business_metrics = business_metrics or [
            "satisfaction_score", "response_time_analysis", 
            "escalation_rate", "category_performance", "sentiment_trends"
        ]
        self.cross_validation = cross_validation or {
            "enabled": True,
            "cv_folds": 5,
            "stratify": True,
            "shuffle": True
        }
        self.statistical_tests = statistical_tests or {
            "enabled": True,
            "confidence_level": 0.95,
            "multiple_comparison_correction": "bonferroni"
        }
        self.model_comparison = model_comparison or {
            "enabled": True,
            "baseline_models": ["textblob", "vader", "naive_bayes"],
            "significance_test": True
        }
        self.business_impact = business_impact or {
            "enabled": True,
            "cost_matrix": {
                "false_positive": 1.0,
                "false_negative": 5.0,
                "true_positive": 0.0,
                "true_negative": 0.0
            },
            "roi_calculation": True
        }
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Evaluate a single model comprehensively.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            X_train: Training features (optional)
            y_train: Training labels (optional)
            model_name: Name of the model
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {"model_name": model_name}
        
        # Get predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
        else:
            # For BERT models
            predictions, probabilities = model.predict(X_test)
            results["probabilities"] = probabilities
        
        # Convert string labels to numeric if needed
        if isinstance(y_test[0], str):
            label_map = {"negative": 0, "neutral": 1, "positive": 2}
            y_test_numeric = np.array([label_map[label] for label in y_test])
            pred_numeric = np.array([label_map[pred] for pred in predictions])
        else:
            y_test_numeric = y_test
            pred_numeric = predictions
        
        # ML Metrics
        results.update(self._compute_ml_metrics(y_test_numeric, pred_numeric))
        
        # Cross-validation if training data available
        if X_train is not None and y_train is not None and self.cross_validation["enabled"]:
            results["cross_validation"] = self._cross_validate_model(
                model, X_train, y_train
            )
        
        # Statistical significance tests
        if self.statistical_tests["enabled"]:
            results["statistical_tests"] = self._statistical_tests(
                y_test_numeric, pred_numeric
            )
        
        # Business metrics
        if self.business_metrics:
            results["business_metrics"] = self._compute_business_metrics(
                y_test, predictions, X_test
            )
        
        # Business impact analysis
        if self.business_impact["enabled"]:
            results["business_impact"] = self._analyze_business_impact(
                y_test_numeric, pred_numeric
            )
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
    
    def _compute_ml_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute standard ML metrics."""
        metrics = {}
        
        if "accuracy" in self.metrics:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        if any(m in self.metrics for m in ["precision_macro", "recall_macro", "f1_macro"]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            metrics.update({
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1
            })
        
        if "confusion_matrix" in self.metrics:
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        
        if "classification_report" in self.metrics:
            label_names = ["negative", "neutral", "positive"]
            metrics["classification_report"] = classification_report(
                y_true, y_pred, target_names=label_names, output_dict=True
            )
        
        return metrics
    
    def _cross_validate_model(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """Perform cross-validation."""
        cv_results = {}
        
        if hasattr(model, 'predict'):
            # For sklearn-compatible models
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.cross_validation["cv_folds"],
                scoring='accuracy'
            )
            cv_results = {
                "cv_scores": cv_scores.tolist(),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "cv_folds": self.cross_validation["cv_folds"]
            }
        else:
            logger.warning("Cross-validation not supported for this model type")
        
        return cv_results
    
    def _statistical_tests(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        # McNemar's test for paired samples
        # This would require predictions from two models
        # For now, return basic statistics
        return {
            "accuracy_confidence_interval": self._confidence_interval(
                accuracy_score(y_true, y_pred), len(y_true)
            ),
            "confidence_level": self.statistical_tests["confidence_level"]
        }
    
    def _confidence_interval(
        self, 
        accuracy: float, 
        n_samples: int, 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for accuracy."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_score * np.sqrt((accuracy * (1 - accuracy)) / n_samples)
        return (accuracy - margin_error, accuracy + margin_error)
    
    def _compute_business_metrics(
        self, 
        y_true: List[str], 
        y_pred: List[str], 
        X_test: Any
    ) -> Dict[str, Any]:
        """Compute business-relevant metrics."""
        business_metrics = {}
        
        if "satisfaction_score" in self.business_metrics:
            # Calculate overall satisfaction score
            satisfaction_map = {"positive": 1, "neutral": 0.5, "negative": 0}
            true_scores = [satisfaction_map[label] for label in y_true]
            pred_scores = [satisfaction_map[label] for label in y_pred]
            
            business_metrics["satisfaction_score"] = {
                "true_mean": np.mean(true_scores),
                "predicted_mean": np.mean(pred_scores),
                "correlation": np.corrcoef(true_scores, pred_scores)[0, 1]
            }
        
        if "escalation_rate" in self.business_metrics:
            # Calculate escalation rates by sentiment
            escalation_rates = {}
            for sentiment in ["positive", "neutral", "negative"]:
                sentiment_mask = np.array(y_true) == sentiment
                if np.any(sentiment_mask):
                    # This would need escalation data from the dataset
                    escalation_rates[sentiment] = 0.0  # Placeholder
            business_metrics["escalation_rate"] = escalation_rates
        
        return business_metrics
    
    def _analyze_business_impact(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze business impact of model predictions."""
        cm = confusion_matrix(y_true, y_pred)
        cost_matrix = self.business_impact["cost_matrix"]
        
        # Calculate costs
        costs = {
            "true_positive": cm[1, 1] * cost_matrix["true_positive"],
            "true_negative": cm[0, 0] * cost_matrix["true_negative"],
            "false_positive": cm[0, 1] * cost_matrix["false_positive"],
            "false_negative": cm[1, 0] * cost_matrix["false_negative"]
        }
        
        total_cost = sum(costs.values())
        
        return {
            "confusion_matrix": cm.tolist(),
            "costs": costs,
            "total_cost": total_cost,
            "cost_per_prediction": total_cost / len(y_true)
        }
    
    def compare_models(
        self, 
        model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple models and create leaderboard."""
        logger.info("Comparing models...")
        
        # Create leaderboard
        leaderboard = []
        for result in model_results:
            leaderboard.append({
                "model": result["model_name"],
                "accuracy": result.get("accuracy", 0),
                "f1_macro": result.get("f1_macro", 0),
                "precision_macro": result.get("precision_macro", 0),
                "recall_macro": result.get("recall_macro", 0)
            })
        
        # Sort by accuracy
        leaderboard.sort(key=lambda x: x["accuracy"], reverse=True)
        
        comparison_results = {
            "leaderboard": leaderboard,
            "best_model": leaderboard[0]["model"] if leaderboard else None,
            "model_count": len(model_results)
        }
        
        # Statistical significance tests between models
        if len(model_results) > 1 and self.model_comparison["significance_test"]:
            comparison_results["significance_tests"] = self._model_significance_tests(
                model_results
            )
        
        return comparison_results
    
    def _model_significance_tests(
        self, 
        model_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform significance tests between models."""
        # This would require access to individual predictions
        # For now, return placeholder
        return {
            "note": "Significance tests require individual model predictions",
            "models_compared": len(model_results)
        }
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, Any], 
        output_dir: str = "assets"
    ) -> str:
        """Generate comprehensive evaluation report."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        self._create_confusion_matrix_plot(results, output_dir)
        self._create_metrics_comparison_plot(results, output_dir)
        
        # Generate text report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write("Customer Feedback Sentiment Analysis - Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, result in results.items():
                if isinstance(result, dict) and "model_name" in result:
                    f.write(f"Model: {result['model_name']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Accuracy: {result.get('accuracy', 'N/A'):.4f}\n")
                    f.write(f"F1 Macro: {result.get('f1_macro', 'N/A'):.4f}\n")
                    f.write(f"Precision Macro: {result.get('precision_macro', 'N/A'):.4f}\n")
                    f.write(f"Recall Macro: {result.get('recall_macro', 'N/A'):.4f}\n")
                    f.write("\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def _create_confusion_matrix_plot(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """Create confusion matrix visualization."""
        for model_name, result in results.items():
            if isinstance(result, dict) and "confusion_matrix" in result:
                cm = result["confusion_matrix"]
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=["Negative", "Neutral", "Positive"],
                    yticklabels=["Negative", "Neutral", "Positive"]
                )
                plt.title(f"Confusion Matrix - {result['model_name']}")
                plt.ylabel("True Label")
                plt.xlabel("Predicted Label")
                
                plot_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
    
    def _create_metrics_comparison_plot(
        self, 
        results: Dict[str, Any], 
        output_dir: str
    ) -> None:
        """Create metrics comparison visualization."""
        models = []
        metrics = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
        metric_values = {metric: [] for metric in metrics}
        
        for model_name, result in results.items():
            if isinstance(result, dict) and "model_name" in result:
                models.append(result["model_name"])
                for metric in metrics:
                    metric_values[metric].append(result.get(metric, 0))
        
        if models:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                axes[i].bar(models, metric_values[metric])
                axes[i].set_title(f"{metric.replace('_', ' ').title()}")
                axes[i].set_ylabel("Score")
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, "metrics_comparison.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
