"""
Script pour préparer les données pour l'analyse avancée.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from preprocessing import ReservoirDataPreprocessor

def prepare_data():
    """Prépare les données pour l'analyse avancée."""
    # Charger les données brutes
    print("Chargement des données brutes...")
    raw_data = pd.read_csv("data/raw/synthetic_reservoir_data.csv")
    
    # Créer les features
    features = pd.DataFrame()
    features['porosity'] = raw_data['porosity']
    features['kh'] = raw_data['kh']
    features['kv'] = raw_data['kv']
    features['ntg'] = raw_data['ntg']
    features['owc'] = raw_data['owc']
    features['initial_pressure'] = raw_data['initial_pressure']
    features['kh_kv_ratio'] = raw_data['kh'] / raw_data['kv']
    features['time_normalized'] = raw_data['time'] / raw_data['time'].max()
    features['time'] = raw_data['time']
    features['simulation_id'] = raw_data['simulation_id']
    
    # Séparer les targets
    target_columns = ['qo', 'water_cut', 'pwf', 'sw']
    targets = raw_data[target_columns]
    
    # Sauvegarder les données prétraitées
    print("Sauvegarde des données prétraitées...")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    features.to_csv(processed_dir / "features.csv", index=False)
    targets.to_csv(processed_dir / "targets.csv", index=False)
    
    print("✅ Données prétraitées sauvegardées dans 'data/processed/'")
    print(f"Nombre de features : {len(features.columns)}")
    print(f"Nombre de targets : {len(target_columns)}")

if __name__ == "__main__":
    prepare_data() 