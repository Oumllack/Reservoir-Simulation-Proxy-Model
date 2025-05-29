"""
Module pour la visualisation des résultats de prédiction du modèle proxy.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class ReservoirVisualizer:
    """Classe pour la visualisation des résultats de prédiction du modèle proxy."""
    
    def __init__(self, save_dir: str = 'figures'):
        """
        Initialise le visualiseur.
        
        Args:
            save_dir: Répertoire de sauvegarde des figures
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Configuration du style
        # plt.style.use('seaborn') # Commenté car le style 'seaborn' n'est plus valide
        sns.set_palette("husl")
        
    def plot_training_curves(self, metrics: Dict[str, Dict[str, float]], save: bool = True):
        """
        Trace les courbes d'apprentissage pour chaque target.
        
        Args:
            metrics: Dictionnaire des métriques par target
            save: Si True, sauvegarde les figures
        """
        targets = list(metrics.keys())
        metric_names = list(metrics[targets[0]].keys())
        
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4*len(metric_names)))
        if len(metric_names) == 1:
            axes = [axes]
            
        for i, metric in enumerate(metric_names):
            values = [metrics[target][metric] for target in targets]
            sns.barplot(x=targets, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} par Target')
            axes[i].set_ylabel(metric.upper())
            axes[i].set_xlabel('Target')
            
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()
        
    def plot_predictions_vs_actual(self, 
                                 y_true: pd.DataFrame, 
                                 y_pred: pd.DataFrame, 
                                 target: str,
                                 save: bool = True):
        """
        Trace les prédictions vs les valeurs réelles pour une target donnée.
        
        Args:
            y_true: DataFrame des valeurs réelles
            y_pred: DataFrame des prédictions
            target: Nom de la target à visualiser
            save: Si True, sauvegarde la figure
        """
        plt.figure(figsize=(10, 6))
        
        # Tracé des points
        plt.scatter(y_true[target], y_pred[target], alpha=0.5)
        
        # Ligne de régression
        z = np.polyfit(y_true[target], y_pred[target], 1)
        p = np.poly1d(z)
        plt.plot(y_true[target], p(y_true[target]), "r--", alpha=0.8)
        
        # Ligne y=x
        min_val = min(y_true[target].min(), y_pred[target].min())
        max_val = max(y_true[target].max(), y_pred[target].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.title(f'Prédictions vs Valeurs Réelles - {target.upper()}')
        plt.xlabel('Valeurs Réelles')
        plt.ylabel('Prédictions')
        
        # Calcul et affichage du R²
        r2 = np.corrcoef(y_true[target], y_pred[target])[0,1]**2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        if save:
            plt.savefig(os.path.join(self.save_dir, f'predictions_vs_actual_{target}.png'))
        plt.close()
        
    def plot_time_series(self, 
                        df_true: pd.DataFrame, 
                        df_pred: pd.DataFrame, 
                        simulation_id: int,
                        save: bool = True):
        """
        Trace les séries temporelles des prédictions et des valeurs réelles.
        
        Args:
            df_true: DataFrame des valeurs réelles
            df_pred: DataFrame des prédictions
            simulation_id: ID de la simulation à visualiser
            save: Si True, sauvegarde la figure
        """
        # Filtrage des données pour la simulation spécifique
        true_data = df_true[df_true['simulation_id'] == simulation_id]
        pred_data = df_pred[df_pred['simulation_id'] == simulation_id]
        
        targets = ['qo', 'water_cut', 'pwf', 'sw']
        fig, axes = plt.subplots(len(targets), 1, figsize=(12, 4*len(targets)))
        
        for i, target in enumerate(targets):
            ax = axes[i]
            ax.plot(true_data['time'], true_data[target], 'b-', label='Valeurs Réelles')
            ax.plot(pred_data['time'], pred_data[target], 'r--', label='Prédictions')
            ax.set_title(f'{target.upper()} - Simulation {simulation_id}')
            ax.set_xlabel('Temps (jours)')
            ax.set_ylabel(target.upper())
            ax.legend()
            
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, f'time_series_sim_{simulation_id}.png'))
        plt.close()
        
    def plot_feature_importance(self, model, feature_names):
        """Trace l'importance des features pour chaque target."""
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            plt.figure(figsize=(10, 6))
            
            # Gestion des différents types de modèles
            if hasattr(model.models[target], 'feature_importances_'):
                # Pour Random Forest et Decision Tree
                importance = model.models[target].feature_importances_
            elif hasattr(model.models[target], 'coef_'):
                # Pour la régression linéaire
                importance = np.abs(model.models[target].coef_).flatten()  # Assure que c'est 1D
            elif hasattr(model.models[target], 'get_booster'):
                # Pour XGBoost
                importance = model.models[target].get_booster().get_score(importance_type='weight')
                # Convertir le dictionnaire en array
                importance = np.array([importance.get(f, 0) for f in feature_names])
            else:
                continue  # Skip si le modèle n'a pas d'importance des features
            
            # Normaliser l'importance
            importance = importance / np.sum(importance)
            
            # Trier les features par importance
            indices = np.argsort(importance)
            plt.barh(range(len(indices)), importance[indices])
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Importance relative')
            plt.title(f'Importance des Features - {target.upper()}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f'feature_importance_{target}.png'))
            plt.close()

if __name__ == "__main__":
    # Test des visualisations
    # Chargement des données
    X_df = pd.read_csv('data/processed/features.csv')
    y_df = pd.read_csv('data/processed/targets.csv')
    
    # Chargement du modèle
    from src.models.baseline_model import ReservoirBaselineModel
    model = ReservoirBaselineModel.load('models/baseline')
    
    # Création du visualiseur
    visualizer = ReservoirVisualizer()
    
    # Visualisation des métriques d'entraînement
    visualizer.plot_training_curves(model.metrics)
    
    # Visualisation des prédictions vs valeurs réelles
    y_pred = model.predict(X_df)
    for target in ['qo', 'water_cut', 'pwf', 'sw']:
        visualizer.plot_predictions_vs_actual(y_df, y_pred, target)
    
    # Visualisation des séries temporelles pour quelques simulations
    for sim_id in [0, 1, 2]:
        visualizer.plot_time_series(y_df, y_pred, sim_id)
    
    # Visualisation de l'importance des features
    visualizer.plot_feature_importance(model, model.feature_names)
    
    print("Figures générées et sauvegardées dans le répertoire 'figures/'") 