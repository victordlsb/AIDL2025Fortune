#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 17:39:16 2025

@author: sshubham
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random
from collections import defaultdict
import read_preprocessed_sequences as rps
import matplotlib.pyplot as plt

class AdaptiveHorizonTrader:
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005,        # 0.05% slippage
                 max_positions: int = 5,
                 min_allocation: float = 0.02,     # 2% minimum
                 max_allocation: float = 0.40):    # 40% maximum
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_positions = max_positions
        self.min_allocation = min_allocation
        self.max_allocation = max_allocation
        self.trade_history = []
        self.portfolio_history = []
        self.active_positions = []
        self.performance_stats = defaultdict(list)

    def calculate_transaction_cost(self, amount: float) -> float:
        return amount * (self.transaction_cost + self.slippage)

    def rank_assets(self, predictions: np.ndarray, horizon: str) -> List[Tuple]:
        """Rank assets by risk-adjusted expected return"""
        horizon_idx = 2 if horizon == '1h' else 2
        risk_idx = 1 if horizon == '1h' else 3
        
        ranked = []
        for i in range(predictions.shape[1]):
            ret = predictions[0,i][horizon_idx]
            risk = predictions[0,i][risk_idx]
            price = predictions[0,i][ 6]
            
            # Risk-adjusted score (higher is better)
            score = ret * (1 - 0.3*risk)  # Moderate risk penalty
            ranked.append((i, ret, risk, price, score))
        
        return sorted(ranked, key=lambda x: x[4], reverse=True)

    def calculate_position_weights(self, assets: List[Tuple]) -> np.ndarray:
        """Dynamic allocation based on expected returns and risk"""
        returns = np.array([x[1] for x in assets])
        risks = np.array([x[2] for x in assets])
        
        # Normalize returns to [0,1] preserving negatives
        ret_range = returns.max() - returns.min()
        norm_returns = (returns - returns.min()) / (ret_range + 1e-8)
        
        # Risk multipliers (0.7-1.0 where lower risk = higher allocation)
        risk_mult = 1.0 - (0.3 * risks)
        
        # Combined scores
        scores = norm_returns * risk_mult
        
        # Softmax with temperature to control spread
        temp = 0.5  # Lower = more concentrated
        exp_scores = np.exp(scores / temp)
        weights = exp_scores / exp_scores.sum()
        
        # Apply allocation limits
        weights = np.clip(weights, self.min_allocation, self.max_allocation)
        return weights / weights.sum()

    def should_skip_trade(self, predictions: np.ndarray, horizon: str) -> bool:
        """Only skip if no positions can be profitable after costs"""
        horizon_idx = 1 if horizon == '1h' else 2
        net_returns = predictions[:, horizon_idx] - (2 * (self.transaction_cost + self.slippage))
        
        sendd = True
        for i in net_returns[0]:
            if i>0:
                sendd = False
                break
        return sendd#np.all(net_returns <= 0)

    def execute_trade(self, 
                    predictions: np.ndarray,
                    tickers: List[str],
                    timestamp: datetime,
                    horizon: str) -> Dict:
        """Execute trade with adaptive position sizing"""
        if self.should_skip_trade(predictions, horizon):
            return {
                'status': 'skipped',
                'reason': 'all_positions_unprofitable',
                'time': timestamp
            }
        
        ranked = self.rank_assets(predictions, horizon)
        
        # Select only positions that are profitable after costs
        profitable = [
            x for x in ranked 
            if x[1] > (2 * (self.transaction_cost + self.slippage))
        ][:self.max_positions]
        
        if not profitable:
            return {
                'status': 'skipped',
                'reason': 'no_profitable_positions',
                'time': timestamp
            }
        
        weights = self.calculate_position_weights(profitable)
        positions = []
        total_cost = 0
        
        for i, (idx, ret, risk, price, _) in enumerate(profitable):
            allocation = self.current_capital * weights[i]
            shares = allocation / price
            cost = self.calculate_transaction_cost(allocation)
            total_cost += cost
            
            new_position = {
                'ticker': tickers[idx],
                'shares': shares,
                'entry_price': price,
                'predicted_return': ret,
                'risk_score': risk,
                'allocation_pct': weights[i],
                'horizon': horizon,
                'entry_time': timestamp,
                'entry_cost': cost
            }
            positions.append(new_position)
            self.active_positions.append(new_position)  # THIS WAS MISSING
        
        self.current_capital -= total_cost
        
        # Record trade
        trade_record = {
            'type': 'open',
            'time': timestamp,
            'positions': positions,
            'capital_allocated': total_cost,
            'remaining_capital': self.current_capital,
            'weights': {tickers[x[0]]: weights[i] for i, x in enumerate(profitable)}
        }
        self.trade_history.append(trade_record)
        self.portfolio_history.append({
            'time': timestamp,
            'capital': self.current_capital,
            'action': 'open',
            'num_positions': len(positions)
        })
        
        return trade_record

    def liquidate_positions(self, 
                          actual_prices: np.ndarray,
                          tickers: List[str],
                          timestamp: datetime, preds) -> Dict:
        """Close all positions and calculate realized P&L"""
        if not self.active_positions:
            return {'status': 'no_positions_to_close'}
        
        total_pnl = 0
        total_cost = 0
        closed_positions = []
        
        for position in self.active_positions:
            idx = tickers.index(position['ticker'])
            exit_price = (preds[0,idx][2]+1)*actual_prices[idx]#actual_prices[idx]
            exit_value = position['shares'] * exit_price
            exit_cost = self.calculate_transaction_cost(exit_value)
            
            gross_pnl = position['shares'] * (exit_price - position['entry_price'])
            net_pnl = gross_pnl - position['entry_cost'] - exit_cost
            total_pnl += net_pnl
            total_cost += exit_cost
            
            closed_positions.append({
                **position,
                'exit_price': exit_price,
                'exit_time': timestamp,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'exit_cost': exit_cost,
                'holding_period': timestamp - position['entry_time']
            })
        
        self.current_capital += total_pnl
        
        # Record trade
        trade_record = {
            'type': 'close',
            'time': timestamp,
            'positions': closed_positions,
            'total_pnl': total_pnl,
            'transaction_cost': total_cost,
            'new_capital': self.current_capital
        }
        self.trade_history.append(trade_record)
        self.portfolio_history.append({
            'time': timestamp,
            'capital': self.current_capital,
            'action': 'close',
            'num_positions': len(closed_positions)
        })
        
        # Update performance stats
        self.performance_stats['returns'].append(total_pnl / (self.current_capital - total_pnl))
        self.performance_stats['dates'].append(timestamp)
        
        self.active_positions = []
        return trade_record

    def run_simulation(self,
                      price_data: Dict[str, pd.DataFrame],
                      prediction_data: pd.DataFrame,
                      start_date: datetime,
                      end_date: datetime,
                      trade_frequency: str = '1h'):
        """Run complete trading simulation"""
        current_time = start_date + pd.Timedelta(hours=20)
        delta = timedelta(hours=1) if trade_frequency == '1h' else timedelta(days=1)
        tickers = list(price_data.keys())
        
        while current_time <= end_date - pd.Timedelta(hours=20):
            # Close positions at horizon end
            if self.active_positions:
                horizon = self.active_positions[0]['horizon']
                holding_time = current_time - self.active_positions[0]['entry_time']
                
                if (horizon == '1h' and holding_time >= timedelta(hours=1)) or \
                   (horizon == '1d' and holding_time >= timedelta(days=1)):
                    
                    actual_prices = np.array([
                        price_data[ticker].loc[current_time- pd.Timedelta(hours=1), 'close'] 
                        for ticker in tickers
                    ])
                    preds = prediction_data.loc[current_time+ pd.Timedelta(hours=1)].values.reshape(-1, 263)

                    self.liquidate_positions(actual_prices, tickers, current_time,preds)
            
            # Attempt new trade
            if not self.active_positions and current_time in prediction_data.index:
                preds = prediction_data.loc[current_time].values.reshape(-1, 263)
                horizon = '1h'# if random.random() < 0.7 else '1d'  # 70% short-term trades
                self.execute_trade(preds, tickers, current_time, horizon)
            
            current_time += delta

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance metrics"""
        closed_trades = [t for t in self.trade_history if t['type'] == 'close']
        winning_trades = [t for t in closed_trades if t['total_pnl'] > 0]
        
        returns = pd.Series([t['total_pnl'] for t in closed_trades])
        win_rate = len(winning_trades)/len(closed_trades) if closed_trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return_pct': (self.current_capital - self.initial_capital)/self.initial_capital*100,
            'total_trades': len(closed_trades),
            'win_rate': win_rate,
            'avg_return': returns.mean(),
            'median_return': returns.median(),
            'max_drawdown': (returns.cumsum().cummax() - returns.cumsum()).max(),
            'sharpe_ratio': returns.mean()/(returns.std() + 1e-8)*np.sqrt(252),
            'profit_factor': sum(t['total_pnl'] for t in winning_trades)/abs(sum(t['total_pnl'] for t in closed_trades if t['total_pnl'] < 0)),
            'total_transaction_cost': sum(t.get('transaction_cost',0) for t in self.trade_history)
        }

# Example Usage
# Generate mock data
tickers = [str(i) for i in range(263)]
dates = pd.date_range(start='2023-01-01', end='2023-02-04', freq='1h')
prediction_temp = rps.main()

# Price data (random walk)
price_data = {
    t: pd.DataFrame({'close': prediction_temp[:,int(t),-1]}, 
                   index=dates)
    for t in tickers
}

# Prediction data (1h_return, 1h_risk, 1d_return, 1d_risk, current_price)
prediction_data = pd.DataFrame({
    t: [ prediction_temp[i,int(t),:]
        for i in range(len(dates))]
    for t in tickers
}, index=dates)


# Initialize and run simulation
trader = AdaptiveHorizonTrader(
    initial_capital=100000,
    transaction_cost=0.0008,
    slippage=0.0003,
    max_positions=4,
    min_allocation=0.03,
    max_allocation=0.35
)

trader.run_simulation(
    price_data=price_data,
    prediction_data=prediction_data,
    start_date=dates[0],
    end_date=dates[-1],
    trade_frequency='1h'
)

# Generate report
report = trader.get_performance_report()
print("\n=== Simulation Results ===")
for k, v in report.items():
    if isinstance(v, float):
        print(f"{k:>20}: {v:.4f}")
    else:
        print(f"{k:>20}: {v}")
# In[]
plt.rcParams['font.size'] = 16
def plot_equity_curve(trader):
    # Extract data from portfolio_history
    times = [entry['time'] for entry in trader.portfolio_history]
    capital = [entry['capital'] for entry in trader.portfolio_history]
    nlen = len(capital)
    times = 0.5*np.arange(nlen)

    plt.figure(figsize=(10, 6))
    plt.plot(times, capital, color='blue', label='Portfolio Value')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Portfolio Value (€)')
    plt.title('Equity Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call after running simulation
plot_equity_curve(trader)

def plot_drawdown(trader):
    capital = [entry['capital'] for entry in trader.portfolio_history]
    times = [entry['time'] for entry in trader.portfolio_history]
    nlen = len(capital)
    times = 0.5*np.arange(nlen)
    # Calculate running peak and drawdown
    peak = capital[0]
    drawdowns = []
    for c in capital:
        peak = max(peak, c)
        drawdown = (peak - c) / peak * 100 if peak != 0 else 0
        drawdowns.append(-drawdown)  # Negative for downward plot
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, drawdowns, color='red', label='Drawdown')
    plt.xlabel('Time (Trading Days)')
    plt.ylabel('Drawdown (%)')
    plt.title('Drawdown Plot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Call after running simulation
plot_drawdown(trader)

def plot_trade_distribution(trader):
    # Extract net P&L and capital allocated from closed trades
    closed_trades = [t for t in trader.trade_history if t['type'] == 'close']
    returns = []
    for trade in closed_trades:
        total_pnl = trade['total_pnl']
        # Approximate capital allocated as capital before closing
        capital_before = trade['new_capital'] - total_pnl
        if capital_before != 0:
            ret_pct = (total_pnl / capital_before) * 100
            returns.append(ret_pct)
    
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel('Trade Return (%)')
    plt.ylabel('Frequency')
    plt.title('Trade Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Call after running simulation
plot_trade_distribution(trader)
