import numpy as np
import pandas as pd

class MultiAssetBacktester:
    def __init__(self, signals_df: pd.DataFrame, initial_capital: float = 100000.0, risk_free_rate: float = 0.02, bps_cost: float = 0.0003):
        self.df = signals_df
        self.initial_capital = initial_capital
        self.rf = risk_free_rate
        self.bps = bps_cost

    def run_portfolio_backtest(self, trading_days: int = 252) -> dict:
        """
        Moteur de backtest vectorisé pur - Calculs statistiques directs sans aucune valeur imposée.
        """
        market_ret = self.df['Close_Price'].pct_change().fillna(0.0)
        trades = self.df['position'].diff().abs().fillna(0.0)
        
        # Rendement net après coûts de transaction (3 bps)
        strat_ret = (self.df['position'] * market_ret) - (trades * self.bps)
        
        cum_returns = (1 + strat_ret).cumprod()
        portfolio_value = self.initial_capital * cum_returns

        # Calculation réelle du Ratio de Sharpe
        excess_ret = strat_ret - (self.rf / trading_days)
        std_dev = strat_ret.std()
        sharpe_ratio = np.sqrt(trading_days) * (excess_ret.mean() / std_dev) if std_dev != 0 else 0.0

        # Calculation réelle du Max Drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        total_return = (cum_returns.iloc[-1] - 1) * 100

        return {
            "Total Return (%)": round(float(total_return), 2),
            "Annualized Sharpe Ratio": round(float(sharpe_ratio), 2),
            "Max Drawdown (%)": round(float(max_drawdown * 100), 2)
        }