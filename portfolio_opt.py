import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

NUM_TRADING_DAYS = 252
NUM_SIMULATIONS=100000
stocks=['RELIANCE.NS','TCS.NS','HINDUNILVR.NS','HDFCBANK.NS','ITC.NS','LT.NS','INFY.NS']
start_date="2018-01-01"
end_date="2024-09-01"

stock_data={}
for i in stocks:
  ticker=yf.Ticker(i)
  stock_data[i]=ticker.history(start=start_date,end=end_date,)['Close']
stock_prices=pd.DataFrame(stock_data)

log_returns=np.log(stock_prices/stock_prices.shift(1))
#log returns are additive, returns follow lognormal distribution
log_returns.dropna()

print("stocks annuliased returns")
print(log_returns.mean()  * NUM_TRADING_DAYS*100)


print("stocks annuliased volatility")
print(np.std(log_returns) * np.sqrt(NUM_TRADING_DAYS))

portfolio_return=[]
portfolio_risk=[]
portfolio_weight=[]


for i in range(NUM_SIMULATIONS):
  w=np.random.random(len(stocks))
  w/=np.sum(w)
  portfolio_weight.append(w)
  p_return=np.sum(log_returns.mean()*w)*NUM_TRADING_DAYS
  portfolio_return.append(p_return)
  p_risk=np.sqrt(np.dot(w.T,np.dot(log_returns.cov()*NUM_TRADING_DAYS,w)))
  portfolio_risk.append(p_risk)


portfolio_weight=np.array(portfolio_weight)
portfolio_risk=np.array(portfolio_risk)
portfolio_return=np.array(portfolio_return)


portfolios=pd.DataFrame({'Returns':portfolio_return,'Risk':portfolio_risk, "Sharpe":portfolio_return/portfolio_risk})


plt.figure(figsize=(10,6))
plt.scatter(portfolio_risk, portfolio_return, c=portfolio_return/portfolio_risk ,marker='o')
plt.grid(True)
plt.xlabel('Expected risk')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')
plt.title("expected risk vs expected return")
plt.tight_layout()
#monte carlo method- large number of values are randomly generated to reach an optimum and close approximation

ind=0
sharpe_ratio=portfolio_return/portfolio_risk
for i in range(len(sharpe_ratio)):
  if sharpe_ratio[i]==np.max(sharpe_ratio):
    ind=i


print("stocks with their corresponding weight")
for i in range(len(stocks)):
  print(stocks[i],"----->",np.round(portfolio_weight[ind][i],5))
  
plt.figure(figsize=(10,6))
plt.scatter(portfolio_risk, portfolio_return, c=portfolio_return/portfolio_risk ,marker='o')
plt.grid(True)
plt.xlabel('Expected risk')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')
plt.plot(portfolio_risk[ind], portfolio_return[ind], 'r*', markersize=10)
plt.title("expected risk vs expected return")
plt.tight_layout()