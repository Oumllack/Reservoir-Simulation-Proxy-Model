"""
Module pour les modèles avancés de prédiction des paramètres de réservoir.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.models.baseline_model import ReservoirBaselineModel

class ReservoirAdvancedModel(ReservoirBaselineModel):
    """Modèle avancé utilisant différents algorithmes pour prédire les paramètres de réservoir."""
    
    def __init__(self, model_type: str = 'xgb'):
        """
        Initialise le modèle avancé.
        
        Args:
            model_type: Type de modèle à utiliser ('xgb', 'rf', 'dt', 'lr')
        """
        super().__init__()
        self.model_type = model_type
        
        # Initialisation des modèles selon le type choisi
        if model_type == 'xgb':
            self.models = {t: xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1) for t in ['qo', 'water_cut', 'pwf', 'sw']}
        elif model_type == 'rf':
            self.models = {
                t: RandomForestRegressor(
                    n_estimators=50,
                    max_depth=10,
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                ) for t in ['qo', 'water_cut', 'pwf', 'sw']
            }
        elif model_type == 'dt':
            self.models = {t: DecisionTreeRegressor(random_state=42) for t in ['qo', 'water_cut', 'pwf', 'sw']}
        elif model_type == 'lr':
            self.models = {t: LinearRegression() for t in ['qo', 'water_cut', 'pwf', 'sw']}
        else:
            raise ValueError("Le type de modèle doit être 'xgb', 'rf', 'dt' ou 'lr'")
    
    def train(self, X_df: pd.DataFrame, y_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Entraîne le modèle sur les données fournies."""
        # Préparation des données
        self.feature_names = X_df.drop(['simulation_id', 'time'], axis=1).columns.tolist()
        X = X_df[self.feature_names].values
        
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            y = y_df[target].values.ravel()  # Assure que y est 1D
            self.models[target].fit(X, y)
        
        return self.evaluate(X_df, y_df)
    
    def predict(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """Effectue des prédictions sur de nouvelles données."""
        X = X_df[self.feature_names].values
        predictions = {}
        
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            predictions[target] = self.models[target].predict(X).flatten()
        
        pred_df = pd.DataFrame(predictions)
        pred_df['simulation_id'] = X_df['simulation_id']
        pred_df['time'] = X_df['time']
        return pred_df
    
    def save(self, path: str):
        """Sauvegarde le modèle et ses métriques."""
        model_dir = os.path.join(path, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        for target, model in self.models.items():
            joblib.dump(model, os.path.join(model_dir, f'{target}_model.joblib'))
        
        pd.DataFrame(self.metrics).to_csv(os.path.join(model_dir, 'metrics.csv'))
        pd.Series(self.feature_names).to_csv(os.path.join(model_dir, 'feature_names.csv'), index=False)
    
    @classmethod
    def load(cls, path: str, model_type: str) -> 'ReservoirAdvancedModel':
        """Charge un modèle sauvegardé."""
        model = cls(model_type=model_type)
        model_dir = os.path.join(path, model_type)
        
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            model.models[target] = joblib.load(os.path.join(model_dir, f'{target}_model.joblib'))
        
        model.metrics = pd.read_csv(os.path.join(model_dir, 'metrics.csv'), index_col=0).to_dict()
        model.feature_names = pd.read_csv(os.path.join(model_dir, 'feature_names.csv')).values.flatten().tolist()
        return model 