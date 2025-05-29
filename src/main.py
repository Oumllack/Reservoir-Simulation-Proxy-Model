"""
Script principal pour l'exécution du pipeline complet de modélisation proxy.
"""
import os
import pandas as pd
from src.data.generate_synthetic_data import generate_dataset
from src.features.preprocess import prepare_training_data
from src.models.baseline_model import ReservoirBaselineModel
from src.models.advanced_models import ReservoirAdvancedModel
from src.visualization.plot_results import ReservoirVisualizer

def main():
    """Exécute le pipeline complet de modélisation proxy."""
    print("🚀 Démarrage du pipeline de modélisation proxy...")
    
    # Création des répertoires nécessaires
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/baseline', exist_ok=True)
    os.makedirs('models/xgb', exist_ok=True)
    os.makedirs('models/rf', exist_ok=True)
    os.makedirs('models/dt', exist_ok=True)
    os.makedirs('models/lr', exist_ok=True)
    os.makedirs('figures/baseline', exist_ok=True)
    os.makedirs('figures/xgb', exist_ok=True)
    os.makedirs('figures/rf', exist_ok=True)
    os.makedirs('figures/dt', exist_ok=True)
    os.makedirs('figures/lr', exist_ok=True)
    
    # 1. Génération des données synthétiques
    print("\n📊 Génération des données synthétiques...")
    df = generate_dataset(n_simulations=1000)
    print(f"✅ {len(df)} points de données générés pour {df['simulation_id'].nunique()} simulations")
    
    # 2. Prétraitement des données
    print("\n🔄 Prétraitement des données...")
    X_df, y_df, scalers = prepare_training_data(df)
    print(f"✅ Données prétraitées : {X_df.shape[1]-1} features, {y_df.shape[1]-1} targets")
    
    # 3. Entraînement des modèles
    print("\n🤖 Entraînement des modèles...")
    models = {
        "baseline": ReservoirBaselineModel(),
        "xgb": ReservoirAdvancedModel(model_type="xgb"),
        "rf": ReservoirAdvancedModel(model_type="rf"),
        "dt": ReservoirAdvancedModel(model_type="dt"),
        "lr": ReservoirAdvancedModel(model_type="lr")
    }
    
    metrics_all = {}
    for name, model in models.items():
        print(f"\nEntraînement du modèle {name}...")
        metrics = model.train(X_df, y_df)
        metrics_all[name] = metrics
        model.save("models/" + (name if name != "baseline" else "baseline"))
        print(f"💾 Modèle {name} sauvegardé dans 'models/{name}/'")
    
    # 4. Affichage des métriques (comparaison avec le baseline)
    print("\n📈 Métriques d'évaluation (comparaison avec le baseline) :")
    baseline_metrics = metrics_all["baseline"]
    for name, metrics in metrics_all.items():
        if name == "baseline":
            continue
        print(f"\n{name.upper()} vs BASELINE :")
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            print(f"  {target.upper()} :")
            for metric in ['r2', 'mae', 'rmse']:
                diff = metrics[target][metric] - baseline_metrics[target][metric]
                print(f"    {metric} : {metrics[target][metric]:.4f} (diff: {diff:+.4f})")
    
    # 5. Génération des visualisations pour chaque modèle
    print("\n🎨 Génération des visualisations...")
    for name, model in models.items():
        print(f"\nGénération des visualisations pour {name}...")
        visualizer = ReservoirVisualizer(save_dir=f"figures/{name}")
        y_pred = model.predict(X_df)
        
        # Visualisation des prédictions vs valeurs réelles
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            visualizer.plot_predictions_vs_actual(y_df, y_pred, target)
        
        # Visualisation des séries temporelles pour quelques simulations
        for sim_id in [0, 1, 2]:
            visualizer.plot_time_series(y_df, y_pred, sim_id)
        
        # Visualisation de l'importance des features (uniquement pour les modèles qui le supportent)
        if name in ['baseline', 'rf', 'dt', 'xgb']:
            try:
                visualizer.plot_feature_importance(model, model.feature_names)
            except Exception as e:
                print(f"⚠️ Impossible de générer l'importance des features pour {name}: {str(e)}")
    
    print("\n✨ Pipeline terminé avec succès !")
    print("📊 Les figures sont disponibles dans les répertoires 'figures/<model_name>/'")
    print("🤖 Les modèles sont sauvegardés dans les répertoires 'models/<model_name>/'")

if __name__ == "__main__":
    main() 