"""
Module pour le modèle de base (régression linéaire) de prédiction des paramètres de réservoir.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, List
import joblib
import os

class ReservoirBaselineModel:
    """Modèle de base utilisant la régression linéaire pour prédire les paramètres de réservoir."""
    
    def __init__(self):
        """Initialise le modèle de base."""
        self.models = {
            'qo': LinearRegression(),
            'water_cut': LinearRegression(),
            'pwf': LinearRegression(),
            'sw': LinearRegression()
        }
        self.feature_names = None
        self.metrics = {}
        
    def prepare_data(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            X_df: DataFrame des features
            y_df: DataFrame des targets
            
        Returns:
            Tuple contenant les dictionnaires de features et targets
        """
        # Stockage des noms des features
        self.feature_names = [col for col in X_df.columns if col != 'simulation_id']
        
        # Préparation des données
        X = X_df[self.feature_names].values
        y = {
            target: y_df[target].values.reshape(-1, 1)
            for target in ['qo', 'water_cut', 'pwf', 'sw']
        }
        
        return X, y
    
    def train(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, float]:
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            X_df: DataFrame des features
            y_df: DataFrame des targets
            
        Returns:
            Dictionnaire des métriques d'évaluation
        """
        # Préparation des données
        X, y = self.prepare_data(X_df, y_df)
        
        # Entraînement des modèles
        for target, model in self.models.items():
            model.fit(X, y[target])
            
        # Évaluation sur les données d'entraînement
        metrics = self.evaluate(X_df, y_df)
        self.metrics = metrics
        
        return metrics
    
    def predict(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X_df: DataFrame des features
            
        Returns:
            DataFrame des prédictions
        """
        X = X_df[self.feature_names].values
        predictions = {}
        
        for target, model in self.models.items():
            # S'assurer que les prédictions sont unidimensionnelles
            predictions[target] = model.predict(X).flatten()
            
        # Création du DataFrame des prédictions
        pred_df = pd.DataFrame(predictions)
        pred_df['simulation_id'] = X_df['simulation_id']
        pred_df['time'] = X_df['time'].values
        
        return pred_df
    
    def evaluate(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Évalue les performances du modèle.
        
        Args:
            X_df: DataFrame des features
            y_df: DataFrame des targets
            
        Returns:
            Dictionnaire des métriques par target
        """
        predictions = self.predict(X_df)
        metrics = {}
        
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            y_true = y_df[target].values
            y_pred = predictions[target].values
            
            metrics[target] = {
                'r2': r2_score(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
            
        return metrics
    
    def save(self, path: str):
        """
        Sauvegarde le modèle et ses métriques.
        
        Args:
            path: Chemin de sauvegarde
        """
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarde des modèles
        for target, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'{target}_model.joblib'))
            
        # Sauvegarde des métriques
        pd.DataFrame(self.metrics).to_csv(os.path.join(path, 'metrics.csv'))
        
        # Sauvegarde des noms des features
        pd.Series(self.feature_names).to_csv(os.path.join(path, 'feature_names.csv'), index=False)
    
    @classmethod
    def load(cls, path: str) -> 'ReservoirBaselineModel':
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin du modèle
            
        Returns:
            Instance du modèle chargé
        """
        model = cls()
        
        # Chargement des modèles
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            model.models[target] = joblib.load(os.path.join(path, f'{target}_model.joblib'))
            
        # Chargement des métriques
        model.metrics = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=0).to_dict()
        
        # Chargement des noms des features
        model.feature_names = pd.read_csv(os.path.join(path, 'feature_names.csv')).values.flatten().tolist()
        
        return model

if __name__ == "__main__":
    # Test du modèle
    # Chargement des données
    X_df = pd.read_csv('data/processed/features.csv')
    y_df = pd.read_csv('data/processed/targets.csv')
    
    # Création et entraînement du modèle
    model = ReservoirBaselineModel()
    metrics = model.train(X_df, y_df)
    
    # Affichage des métriques
    print("\nMétriques d'évaluation :")
    for target, target_metrics in metrics.items():
        print(f"\n{target.upper()}:")
        for metric_name, value in target_metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Sauvegarde du modèle
    model.save('models/baseline')
    print("\nModèle sauvegardé dans 'models/baseline/'") 