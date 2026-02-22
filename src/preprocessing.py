"""
Data preprocessing utilities for ML projects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


class DataPreprocessor:
    """Class for handling data preprocessing tasks."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_cleaned = df.copy()
        
        if strategy == 'mean':
            df_cleaned.fillna(df_cleaned.select_dtypes(include=[np.number]).mean(), inplace=True)
        elif strategy == 'median':
            df_cleaned.fillna(df_cleaned.select_dtypes(include=[np.number]).median(), inplace=True)
        elif strategy == 'mode':
            df_cleaned.fillna(df_cleaned.mode().iloc[0], inplace=True)
        elif strategy == 'drop':
            df_cleaned.dropna(inplace=True)
        
        return df_cleaned
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        return df_encoded
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale numerical features using StandardScaler."""
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def preprocess_pipeline(self, file_path: str, target_column: str, 
                           handle_missing: bool = True, encode_categorical: bool = True,
                           scale_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline."""
        # Load data
        df = self.load_data(file_path)
        
        # Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df)
        
        # Encode categorical features
        if encode_categorical:
            df = self.encode_categorical_features(df)
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Scale features
        if scale_features:
            X = self.scale_features(X, fit=True)
        
        # Split data
        return self.split_data(X, y)


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    print("Data preprocessing utilities loaded successfully!")
