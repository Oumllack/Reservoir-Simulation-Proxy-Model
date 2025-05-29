"""
Module pour la génération de données synthétiques de simulation de réservoir.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class ReservoirSimulator:
    """Simulateur simplifié de réservoir pour générer des données synthétiques."""
    
    def __init__(self, 
                 n_cells: int = 100,
                 time_steps: int = 365,
                 dt: float = 1.0):
        """
        Initialise le simulateur de réservoir.
        
        Args:
            n_cells: Nombre de cellules dans le modèle
            time_steps: Nombre de pas de temps
            dt: Pas de temps en jours
        """
        self.n_cells = n_cells
        self.time_steps = time_steps
        self.dt = dt
        
    def generate_reservoir_parameters(self) -> Dict[str, float]:
        """Génère des paramètres aléatoires réalistes pour le réservoir."""
        return {
            'porosity': np.random.uniform(0.15, 0.25),  # Porosité moyenne
            'kh': np.random.uniform(100, 1000),         # Perméabilité horizontale (mD)
            'kv': np.random.uniform(10, 100),           # Perméabilité verticale (mD)
            'ntg': np.random.uniform(0.6, 0.9),         # Net-to-Gross ratio
            'owc': np.random.uniform(2000, 2500),       # Profondeur du contact eau-huile (m)
            'initial_pressure': np.random.uniform(250, 300)  # Pression initiale (bar)
        }
    
    def simulate_production(self, params: Dict[str, float]) -> pd.DataFrame:
        """
        Simule la production du réservoir avec les paramètres donnés.
        
        Args:
            params: Dictionnaire des paramètres du réservoir
            
        Returns:
            DataFrame contenant les résultats de simulation
        """
        # Initialisation
        time = np.arange(0, self.time_steps * self.dt, self.dt)
        qo = np.zeros_like(time, dtype=float)  # Débit d'huile
        wc = np.zeros_like(time, dtype=float)  # Water cut
        pwf = np.zeros_like(time, dtype=float) # Pression fond de trou
        sw = np.zeros_like(time, dtype=float)  # Saturation en eau
        
        # Paramètres de simulation
        qi = 1000  # Débit d'injection initial (bbl/j)
        pi = params['initial_pressure']
        kh_kv_ratio = params['kh'] / params['kv']
        
        # Simulation temporelle
        for i, t in enumerate(time):
            # Calcul du débit d'huile (modèle simplifié)
            qo[i] = qi * params['porosity'] * params['ntg'] * (1 - np.exp(-t/100))
            
            # Calcul du water cut (modèle simplifié)
            wc[i] = 1 - np.exp(-t/200) * (1 - params['ntg'])
            
            # Calcul de la pression fond de trou
            pwf[i] = pi - (qo[i] * t) / (params['kh'] * 1000)
            
            # Calcul de la saturation en eau
            sw[i] = wc[i] * params['porosity']
            
            # Ajout de bruit réaliste
            qo[i] += np.random.normal(0, qo[i] * 0.05)
            wc[i] += np.random.normal(0, 0.02)
            pwf[i] += np.random.normal(0, 2)
            sw[i] += np.random.normal(0, 0.01)
            
            # Contraintes physiques
            qo[i] = max(0, qo[i])
            wc[i] = np.clip(wc[i], 0, 1)
            sw[i] = np.clip(sw[i], 0, 1)
        
        # Création du DataFrame
        df = pd.DataFrame({
            'time': time,
            'qo': qo,
            'water_cut': wc,
            'pwf': pwf,
            'sw': sw
        })
        
        # Ajout des paramètres du réservoir
        for key, value in params.items():
            df[key] = value
            
        return df

def generate_dataset(n_simulations: int = 100) -> pd.DataFrame:
    """
    Génère un ensemble de données de simulation.
    
    Args:
        n_simulations: Nombre de simulations à générer
        
    Returns:
        DataFrame contenant toutes les simulations
    """
    simulator = ReservoirSimulator()
    all_simulations = []
    
    for i in range(n_simulations):
        params = simulator.generate_reservoir_parameters()
        df = simulator.simulate_production(params)
        df['simulation_id'] = i
        all_simulations.append(df)
    
    return pd.concat(all_simulations, ignore_index=True)

if __name__ == "__main__":
    # Génération d'un jeu de données de test
    df = generate_dataset(n_simulations=100)
    
    # Sauvegarde des données
    df.to_csv('data/raw/synthetic_reservoir_data.csv', index=False)
    print(f"Données générées et sauvegardées dans 'data/raw/synthetic_reservoir_data.csv'")
    print(f"Nombre total de simulations : {df['simulation_id'].nunique()}")
    print(f"Nombre total de points de données : {len(df)}") 