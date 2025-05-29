"""
Module pour le prétraitement des données de réservoir.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ReservoirDataPreprocessor:
    """Classe pour le prétraitement des données de réservoir."""
    
    def __init__(self):
        """Initialise le prétraiteur de données."""
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None
    
    def preprocess(self, features_df, targets_df, test_size=0.2, random_state=42):
        """
        Prétraite les données de features et targets.
        
        Args:
            features_df (pd.DataFrame): DataFrame des features
            targets_df (pd.DataFrame): DataFrame des targets
            test_size (float): Proportion des données de test
            random_state (int): Seed pour la reproductibilité
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, target_names)
        """
        # Sauvegarder les noms des colonnes
        self.feature_names = features_df.columns.tolist()
        self.target_names = targets_df.columns.tolist()
        
        # Normaliser les features
        X = self.feature_scaler.fit_transform(features_df)
        X = pd.DataFrame(X, columns=self.feature_names)
        
        # Normaliser les targets
        y = self.target_scaler.fit_transform(targets_df)
        y = pd.DataFrame(y, columns=self.target_names)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, self.feature_names, self.target_names
    
    def inverse_transform_targets(self, y_pred):
        """
        Inverse la normalisation des prédictions.
        
        Args:
            y_pred (np.ndarray): Prédictions normalisées
            
        Returns:
            pd.DataFrame: Prédictions dans l'échelle originale
        """
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
        return pd.DataFrame(
            self.target_scaler.inverse_transform(y_pred),
            columns=self.target_names
        ) 