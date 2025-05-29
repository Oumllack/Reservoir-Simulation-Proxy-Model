"""
Module pour le prétraitement des données de simulation de réservoir.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict

class ReservoirDataPreprocessor:
    """Classe pour le prétraitement des données de simulation de réservoir."""
    
    def __init__(self):
        """Initialise les scalers pour la normalisation des données."""
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Prépare les features pour l'entraînement du modèle.
        
        Args:
            df: DataFrame contenant les données brutes
            
        Returns:
            Tuple contenant:
            - DataFrame des features prétraitées
            - Dictionnaire des scalers utilisés
        """
        # Sélection des features statiques (paramètres du réservoir)
        static_features = ['porosity', 'kh', 'kv', 'ntg', 'owc', 'initial_pressure']
        
        # Création de features dérivées
        df['kh_kv_ratio'] = df['kh'] / df['kv']
        df['time_normalized'] = df['time'] / df['time'].max()
        
        # Features pour l'entraînement
        features = static_features + ['kh_kv_ratio', 'time_normalized']
        
        # Normalisation des features
        X = self.feature_scaler.fit_transform(df[features])
        
        # Création du DataFrame des features
        X_df = pd.DataFrame(X, columns=features)
        X_df['simulation_id'] = df['simulation_id']
        X_df['time'] = df['time']
        
        return X_df, {'feature_scaler': self.feature_scaler}
    
    def prepare_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Prépare les variables cibles pour l'entraînement du modèle.
        
        Args:
            df: DataFrame contenant les données brutes
            
        Returns:
            Tuple contenant:
            - DataFrame des targets prétraitées
            - Dictionnaire des scalers utilisés
        """
        # Sélection des variables cibles
        targets = ['qo', 'water_cut', 'pwf', 'sw']
        
        # Normalisation des targets
        y = self.target_scaler.fit_transform(df[targets])
        
        # Création du DataFrame des targets
        y_df = pd.DataFrame(y, columns=targets)
        y_df['simulation_id'] = df['simulation_id']
        y_df['time'] = df['time']
        
        return y_df, {'target_scaler': self.target_scaler}
    
    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse la transformation des variables cibles.
        
        Args:
            y: Array numpy des prédictions normalisées
            
        Returns:
            Array numpy des prédictions dans l'échelle originale
        """
        return self.target_scaler.inverse_transform(y)

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """
    Prépare les données pour l'entraînement du modèle.
    
    Args:
        df: DataFrame contenant les données brutes
        
    Returns:
        Tuple contenant:
        - DataFrame des features
        - DataFrame des targets
        - Dictionnaire des scalers
    """
    preprocessor = ReservoirDataPreprocessor()
    
    # Préparation des features et targets
    X_df, feature_scalers = preprocessor.prepare_features(df)
    y_df, target_scalers = preprocessor.prepare_targets(df)
    
    # Fusion des scalers
    scalers = {**feature_scalers, **target_scalers}
    
    return X_df, y_df, scalers

if __name__ == "__main__":
    # Test du prétraitement
    df = pd.read_csv('data/raw/synthetic_reservoir_data.csv')
    X_df, y_df, scalers = prepare_training_data(df)
    
    # Sauvegarde des données prétraitées
    X_df.to_csv('data/processed/features.csv', index=False)
    y_df.to_csv('data/processed/targets.csv', index=False)
    
    print("Données prétraitées sauvegardées dans 'data/processed/'")
    print(f"Nombre de features : {X_df.shape[1] - 1}")  # -1 pour simulation_id
    print(f"Nombre de targets : {y_df.shape[1] - 1}")   # -1 pour simulation_id 