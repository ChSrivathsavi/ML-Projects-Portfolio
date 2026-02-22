"""
Model evaluation utilities for ML projects.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """Class for comprehensive model evaluation."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.figures = {}
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: Optional[np.ndarray] = None,
                              model_name: str = "model") -> Dict[str, float]:
        """Comprehensive classification evaluation."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # ROC AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = auc(*roc_curve(y_true, y_pred_proba[:, 1])[:2])
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "model") -> Dict[str, float]:
        """Comprehensive regression evaluation."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        self.evaluation_results[model_name] = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            model_name: str = "model", save_path: Optional[str] = None) -> None:
        """Plot confusion matrix for classification."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(len(cm))],
                   yticklabels=[f'Class {i}' for i in range(len(cm))])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str = "model", save_path: Optional[str] = None) -> None:
        """Plot ROC curve for binary classification."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "model", save_path: Optional[str] = None) -> None:
        """Plot regression results with actual vs predicted."""
        plt.figure(figsize=(10, 6))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'Actual vs Predicted - {model_name}')
        
        # Residual plot
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, metric: str = 'accuracy' if 'classification' in str(evaluation_results) else 'r2') -> pd.DataFrame:
        """Compare performance of multiple models."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        comparison_df = pd.DataFrame(self.evaluation_results).T
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in evaluation results")
        
        return comparison_df.sort_values(metric, ascending=False)
    
    def plot_model_comparison(self, metric: str = 'accuracy' if 'classification' in str(evaluation_results) else 'r2',
                            save_path: Optional[str] = None) -> None:
        """Plot comparison of multiple models."""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        comparison_df = self.compare_models(metric)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(comparison_df.index, comparison_df[metric])
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    model_name: str = "model") -> str:
        """Generate detailed classification report."""
        return classification_report(y_true, y_pred)
    
    def cross_validate_scores(self, scores: List[float], model_name: str = "model") -> Dict[str, float]:
        """Summarize cross-validation scores."""
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'scores': scores
        }
    
    def feature_importance_analysis(self, feature_names: List[str], importances: np.ndarray,
                                 model_name: str = "model", top_n: int = 10,
                                 save_path: Optional[str] = None) -> None:
        """Plot feature importance."""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_results(self, file_path: str) -> None:
        """Save evaluation results to a JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    print("Model evaluation utilities loaded successfully!")
