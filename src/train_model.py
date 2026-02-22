"""
Model training utilities for ML projects.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from typing import Dict, Any, Tuple, Optional
import joblib


class ModelTrainer:
    """Class for training and managing ML models."""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_performance = {}
    
    def get_classification_models(self) -> Dict[str, Any]:
        """Get dictionary of classification models."""
        return {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'knn': KNeighborsClassifier()
        }
    
    def get_regression_models(self) -> Dict[str, Any]:
        """Get dictionary of regression models."""
        return {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(random_state=42),
            'svm': SVR(),
            'knn': KNeighborsRegressor()
        }
    
    def train_single_model(self, model_name: str, model: Any, X_train: np.ndarray, 
                          y_train: np.ndarray, task_type: str = 'classification') -> Dict[str, Any]:
        """Train a single model and return performance metrics."""
        # Train the model
        model.fit(X_train, y_train)
        
        # Store the trained model
        self.trained_models[model_name] = model
        
        return {
            'model_name': model_name,
            'model': model,
            'trained': True
        }
    
    def train_multiple_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                            task_type: str = 'classification') -> Dict[str, Dict[str, Any]]:
        """Train multiple models and compare performance."""
        if task_type == 'classification':
            models = self.get_classification_models()
        else:
            models = self.get_regression_models()
        
        results = {}
        
        for name, model in models.items():
            try:
                result = self.train_single_model(name, model, X_train, y_train, task_type)
                results[name] = result
                print(f"Successfully trained {name}")
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                results[name] = {'model_name': name, 'error': str(e)}
        
        return results
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray,
                      task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred)
            }
            # Additional classification metrics
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
                metrics['precision_macro'] = report['macro avg']['precision']
                metrics['recall_macro'] = report['macro avg']['recall']
                metrics['f1_macro'] = report['macro avg']['f1-score']
            except:
                pass
        else:
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
        
        self.model_performance[model_name] = metrics
        return metrics
    
    def get_best_model(self, task_type: str = 'classification') -> Tuple[str, Any, Dict[str, float]]:
        """Get the best performing model based on evaluation metrics."""
        if not self.model_performance:
            raise ValueError("No models have been evaluated yet")
        
        if task_type == 'classification':
            best_model_name = max(self.model_performance.keys(), 
                                key=lambda x: self.model_performance[x]['accuracy'])
        else:
            best_model_name = max(self.model_performance.keys(), 
                                key=lambda x: self.model_performance[x]['r2'])
        
        return (best_model_name, 
                self.trained_models[best_model_name], 
                self.model_performance[best_model_name])
    
    def save_model(self, model_name: str, file_path: str) -> None:
        """Save a trained model to disk."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        joblib.dump(self.trained_models[model_name], file_path)
        print(f"Model {model_name} saved to {file_path}")
    
    def load_model(self, model_name: str, file_path: str) -> None:
        """Load a trained model from disk."""
        self.trained_models[model_name] = joblib.load(file_path)
        print(f"Model {model_name} loaded from {file_path}")
    
    def save_all_models(self, directory: str) -> None:
        """Save all trained models to a directory."""
        for model_name in self.trained_models:
            file_path = f"{directory}/{model_name}.pkl"
            self.save_model(model_name, file_path)
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        return self.trained_models[model_name].predict(X)


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    print("Model training utilities loaded successfully!")
