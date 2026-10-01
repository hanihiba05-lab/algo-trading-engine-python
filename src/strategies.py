import numpy as np
import pandas as pd

def generate_signals(df1: pd.DataFrame, df2: pd.DataFrame, lookback: int = 20, entry_z: float = 1.0) -> pd.DataFrame:
    """
    Génère des signaux quantitatifs de Statistical Arbitrage (Pair Trading).
    Mise en place d'un modèle de retour à la moyenne vectorisé avec contrôle du biais d'anticipation.
    """
    if isinstance(df1.columns, pd.MultiIndex):
        p1 = df1['Close'].iloc[:, 0]
        p2 = df2['Close'].iloc[:, 0]
    else:
        p1 = df1['Close']
        p2 = df2['Close']

    # Calcul du Spread logarithmique stationnaire
    spread = np.log(p1) - np.log(p2)
    rolling_mean = spread.rolling(window=lookback).mean()
    rolling_std = spread.rolling(window=lookback).std()

    z_score = (spread - rolling_mean) / (rolling_std + 1e-8)

    df = pd.DataFrame(index=p1.index)
    df['Close_Price'] = p1
    df['z_score'] = z_score

    # Génération des positions (-1, 0, 1)
    df['signal'] = 0.0
    df.loc[df['z_score'] < -entry_z, 'signal'] = 1.0   # Achat du spread
    df.loc[df['z_score'] > entry_z, 'signal'] = -1.0    # Vente du spread
    df.loc[df['z_score'].abs() < 0.2, 'signal'] = 0.0   # Take Profit

    df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0.0)
    
    # Décalage d'un jour obligatoire pour supprimer le Look-Ahead Bias
    df['position'] = df['signal'].shift(1).fillna(0.0)

    return df