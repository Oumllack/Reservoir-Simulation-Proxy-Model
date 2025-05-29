import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.analysis.advanced_analysis import ReservoirAnalyzer
from src.models.advanced_models import ReservoirAdvancedModel

def load_data():
    """Charge les données prétraitées."""
    data_dir = Path("data/processed")
    features = pd.read_csv(data_dir / "features.csv")
    targets = pd.read_csv(data_dir / "targets.csv")
    return features, targets

def load_models():
    """Charge tous les modèles entraînés."""
    models = {}
    model_types = ['xgb', 'rf', 'dt', 'lr']
    
    for model_type in model_types:
        model = ReservoirAdvancedModel.load(f"models/{model_type}", model_type=model_type)
        models[model_type] = model
    
    return models

def run_analysis():
    """Exécute l'analyse avancée pour tous les modèles."""
    # Créer les répertoires pour les résultats
    analysis_dir = Path("analysis_results")
    analysis_dir.mkdir(exist_ok=True)
    
    # Charger les données et les modèles
    print("Chargement des données et des modèles...")
    features, targets = load_data()
    models = load_models()
    
    # Analyser chaque modèle
    for model_name, model in models.items():
        print(f"\nAnalyse du modèle {model_name.upper()}...")
        model_dir = analysis_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Initialiser l'analyseur
        analyzer = ReservoirAnalyzer(
            model=model,
            features_df=features,
            targets_df=targets
        )
        
        # Calculer et sauvegarder l'importance des features
        print("Calcul de l'importance des features...")
        importance = analyzer.compute_permutation_importance()
        importance.to_csv(model_dir / "feature_importance.csv")
        
        # Comparer avec la simulation complète
        print("Comparaison avec la simulation complète...")
        comparison = analyzer.compare_with_full_simulation()
        comparison.to_csv(model_dir / "simulation_comparison.csv")
        
        print(f"Analyse terminée pour le modèle {model_name.upper()}")
        print(f"Résultats sauvegardés dans {model_dir}")

if __name__ == "__main__":
    run_analysis() 