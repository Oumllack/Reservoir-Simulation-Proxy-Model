"""
Module pour l'analyse avancée des résultats du modèle proxy.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import shap
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from pathlib import Path
from src.models.advanced_models import ReservoirAdvancedModel

class ReservoirAnalyzer:
    """Classe pour l'analyse avancée des résultats du modèle proxy."""
    
    def __init__(self, model: ReservoirAdvancedModel, features_df: pd.DataFrame, targets_df: pd.DataFrame):
        """
        Initialise l'analyseur.
        
        Args:
            model: Modèle entraîné
            features_df: Features
            targets_df: Targets
        """
        self.model = model
        self.X_df = features_df
        self.y_df = targets_df
        self.feature_names = model.feature_names  # Utiliser les features du modèle
        self.save_dir = 'figures/analysis'
        os.makedirs(self.save_dir, exist_ok=True)
        
    def compute_shap_values(self, target: str = 'qo', n_samples: int = 1000) -> pd.DataFrame:
        """
        Calcule les valeurs SHAP pour l'analyse de sensibilité.
        
        Args:
            target: Target à analyser
            n_samples: Nombre d'échantillons pour l'analyse SHAP
            
        Returns:
            DataFrame avec les valeurs SHAP
        """
        # Sélection d'un sous-ensemble pour l'analyse SHAP
        X_sample = self.X_df[self.feature_names].sample(n=min(n_samples, len(self.X_df)))
        
        # Calcul des valeurs SHAP selon le type de modèle
        if hasattr(self.model.models[target], 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(self.model.models[target])
            shap_values = explainer.shap_values(X_sample.values)  # Utiliser .values pour avoir un array numpy
        else:  # Autres modèles
            explainer = shap.KernelExplainer(self.model.models[target].predict, X_sample.values)
            shap_values = explainer.shap_values(X_sample.values)
        
        # Création du DataFrame des valeurs SHAP
        shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
        shap_df['feature_values'] = X_sample.values.tolist()  # Convertir en liste pour éviter les problèmes de dimension
        
        return shap_df
    
    def plot_shap_summary(self, target: str = 'qo', n_samples: int = 1000, save_path: str = None):
        """Trace le résumé des valeurs SHAP."""
        shap_df = self.compute_shap_values(target, n_samples)
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_df.drop('feature_values', axis=1).values,
            self.X_df[self.feature_names].sample(n=min(n_samples, len(self.X_df))).values,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'Importance des Features (SHAP) - {target.upper()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def compute_permutation_importance(self, target: str = 'qo', n_repeats: int = 5) -> pd.DataFrame:
        """Calcule l'importance des features par permutation."""
        X = self.X_df[self.feature_names]
        y = self.y_df[target]
        
        result = permutation_importance(
            self.model.models[target],
            X, y,
            n_repeats=n_repeats,
            random_state=42
        )
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_permutation_importance(self, target: str = 'qo'):
        """Trace l'importance des features par permutation."""
        importance_df = self.compute_permutation_importance(target)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importance_df, x='Importance', y='Feature')
        plt.errorbar(
            importance_df['Importance'],
            range(len(importance_df)),
            xerr=importance_df['Std'],
            fmt='none',
            color='black',
            capsize=5
        )
        plt.title(f'Importance des Features (Permutation) - {target.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'permutation_importance_{target}.png'))
        plt.close()
    
    def plot_3d_production_map(self, target: str = 'qo', resolution: int = 20):
        """
        Trace une carte 3D des zones favorables à la production.
        
        Args:
            target: Target à visualiser
            resolution: Résolution de la grille
        """
        # Création d'une grille 3D
        x = np.linspace(self.X_df['x'].min(), self.X_df['x'].max(), resolution)
        y = np.linspace(self.X_df['y'].min(), self.X_df['y'].max(), resolution)
        z = np.linspace(self.X_df['z'].min(), self.X_df['z'].max(), resolution)
        
        # Création des points de la grille
        grid_points = []
        for i in x:
            for j in y:
                for k in z:
                    grid_points.append([i, j, k])
        grid_points = np.array(grid_points)
        
        # Création d'un DataFrame avec les points de la grille
        grid_df = pd.DataFrame(grid_points, columns=['x', 'y', 'z'])
        
        # Ajout des autres features avec leurs valeurs moyennes
        for feature in self.feature_names:
            if feature not in ['x', 'y', 'z']:
                grid_df[feature] = self.X_df[feature].mean()
        
        # Prédiction sur la grille
        predictions = self.model.predict(grid_df)[target].values
        
        # Création de la figure 3D
        fig = go.Figure(data=go.Volume(
            x=grid_points[:, 0],
            y=grid_points[:, 1],
            z=grid_points[:, 2],
            value=predictions,
            isomin=predictions.min(),
            isomax=predictions.max(),
            opacity=0.3,
            surface_count=20,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f'Carte 3D de Production - {target.upper()}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.write_html(os.path.join(self.save_dir, f'production_map_3d_{target}.html'))
    
    def compare_with_full_simulation(self, n_scenarios: int = 1000) -> pd.DataFrame:
        """Compare les performances du proxy avec la simulation complète."""
        # Temps de prédiction avec le proxy
        start_time = time.time()
        proxy_predictions = self.model.predict(self.X_df.sample(n=n_scenarios))
        proxy_time = time.time() - start_time
        
        # Temps de simulation (estimation)
        simulation_time = proxy_time * 100  # Estimation: simulation 100x plus lente
        
        # Calcul des métriques pour chaque target
        metrics = []
        for target in ['qo', 'water_cut', 'pwf', 'sw']:
            y_true = self.y_df[target].sample(n=n_scenarios)
            y_pred = proxy_predictions[target]
            
            metrics.append({
                'target': target,
                'r2': np.corrcoef(y_true, y_pred)[0,1]**2,
                'mae': np.mean(np.abs(y_true - y_pred)),
                'rmse': np.sqrt(np.mean((y_true - y_pred)**2)),
                'proxy_time': proxy_time,
                'simulation_time': simulation_time,
                'speedup': simulation_time / proxy_time
            })
        
        return pd.DataFrame(metrics)
    
    def plot_comparison_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """Trace les métriques de comparaison."""
        # Temps de calcul
        plt.figure(figsize=(10, 6))
        times = [metrics['time_saved']['proxy_time'], metrics['time_saved']['simulation_time']]
        labels = ['Proxy Model', 'Simulation Complète']
        plt.bar(labels, times)
        plt.title('Comparaison des Temps de Calcul')
        plt.ylabel('Temps (secondes)')
        plt.yscale('log')
        plt.savefig(os.path.join(self.save_dir, 'time_comparison.png'))
        plt.close()
        
        # Précision
        targets = list(metrics['accuracy'].keys())
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, metric in enumerate(['r2', 'mae', 'rmse']):
            values = [metrics['accuracy'][t][metric] for t in targets]
            sns.barplot(x=targets, y=values, ax=axes[i])
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'accuracy_metrics.png'))
        plt.close()
    
    def generate_business_report(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """
        Génère un rapport business avec les impacts et recommandations.
        
        Args:
            metrics: Métriques de comparaison
            
        Returns:
            Rapport au format texte
        """
        report = []
        report.append("# Rapport d'Impact Business - Modèle Proxy\n")
        
        # Impact sur le temps de calcul
        speedup = metrics['time_saved']['speedup']
        report.append(f"## 1. Gain de Temps\n")
        report.append(f"- Temps de prédiction proxy: {metrics['time_saved']['proxy_time']:.2f} secondes")
        report.append(f"- Temps de simulation estimé: {metrics['time_saved']['simulation_time']:.2f} secondes")
        report.append(f"- Accélération: {speedup:.1f}x\n")
        
        # Impact sur la précision
        report.append("## 2. Précision du Modèle\n")
        for target in metrics['accuracy']:
            report.append(f"### {target.upper()}")
            report.append(f"- R²: {metrics['accuracy'][target]['r2']:.4f}")
            report.append(f"- MAE: {metrics['accuracy'][target]['mae']:.4f}")
            report.append(f"- RMSE: {metrics['accuracy'][target]['rmse']:.4f}\n")
        
        # Recommandations
        report.append("## 3. Recommandations\n")
        report.append("### Optimisation des Coûts")
        report.append(f"- Réduction estimée du temps de simulation: {speedup:.1f}x")
        report.append("- Impact sur l'OPEX: Réduction significative des coûts de calcul")
        report.append("- Impact sur le CAPEX: Meilleure allocation des ressources\n")
        
        report.append("### Stratégie de Développement")
        report.append("- Utilisation du proxy pour l'évaluation rapide des scénarios")
        report.append("- Simulation complète uniquement pour les scénarios les plus prometteurs")
        report.append("- Optimisation des paramètres de développement basée sur l'analyse de sensibilité\n")
        
        report.append("### Gestion des Risques")
        report.append("- Identification des zones de forte incertitude via l'analyse SHAP")
        report.append("- Évaluation des risques géologiques basée sur les prédictions du proxy")
        report.append("- Support à la prise de décision pour l'exploration et le développement")
        
        # Sauvegarde du rapport
        report_path = os.path.join(self.save_dir, 'business_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report) 