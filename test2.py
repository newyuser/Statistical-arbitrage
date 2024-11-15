import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
# 获取ticker（上证50）
import json
from timeit import default_timer as timer
t1 = timer()
with open('510050_20230214.json') as user_file:
  parsed_json = json.load(user_file)

tickers = []
for i in parsed_json["StockComponent"].keys():
  tickers.append(i+".SH")

import tushare as ts
ts.set_token('0742c95bee169abeccdf80bbdce1b2e0cf2ac4d83cbd932b608300fd')
pro = ts.pro_api()

def get_price(tickers, start, end):
    stocks = pro.daily(ts_code=tickers[0], start_date=start, end_date=end, fields=['trade_date'])
    for ticker in tickers:
        stock = pro.daily(ts_code=ticker, start_date=start, end_date=end, fields=['trade_date', 'close'])
        stock = stock.rename(columns={"close" : ticker})
        stocks = stocks.merge(stock,on=['trade_date'])
    stocks.set_index(['trade_date'], inplace=True)
    stocks = stocks.iloc[::-1]
    stocks = stocks.dropna()
    return stocks

def get_date_cv(tickers, date):
    stocks = pd.DataFrame(columns=['ts_code', 'close', 'vol'])
    for ticker in tickers:
        stock = pro.daily(ts_code=ticker, trade_date=date, fields=['ts_code', 'close', 'vol'])
        stocks = stocks.append(stock)
    return stocks


def get_weight(test_history):
    #算法主代码
    num_components = 3
    # 数据log平滑处理和中心化处理后做PCA
    # Sample data for PCA (smooth it using np.log function)
    sample = np.log(test_history)   #???取对数的效果更好吗?
    #sample.mean()对所有symbol的每个交易日求均值
    sample -= sample.mean() # Center it column-wise

    # Fit the PCA model for sample data
    model = PCA(n_components=num_components).fit(sample) #???初始设置k=3训练 和 k=n训练后取前3组 有什么差别?

    # 得到每个交易日的评分以反映市场(数值大小似乎没有意义,主要是为了之后与symbol做线性回归算残差)
    factors = np.dot(sample, model.components_.T)
    # Add 1's to fit the linear regression (intercept) 线性回归的必要处理
    factors = sm.add_constant(factors)

    # Train Ordinary Least Squares linear model for each stock sample[ticker]为y factors为x 做线性回归
    OLSmodels = {ticker: sm.OLS(sample[ticker], factors).fit() for ticker in sample.columns}

    # Get the residuals from the linear regression after PCA for each stock
    resids = pd.DataFrame({ticker: model.resid for ticker, model in OLSmodels.items()}) #???残差均值很小,计算机标准化过程中似乎会出现数值错误?

    #检测white noise并删除对应列
    white_noise = []
    for ticker in resids.columns:
        df = lb_test(resids[ticker], lags=20)
        if not df[df['lb_pvalue'] > 0.05].empty:
            white_noise.append(ticker)
    resids.drop(columns=white_noise, inplace=True)

    # 对残差标准化后取最近一个交易日的zscores  理解的是 zscores负得多,说明对应股票价值被低估(close < 模型认为的股票价值),应该加大下一天对其的投资占比
    # Get the Z scores by standarize the given pandas dataframe X  
    zscores = ((resids - resids.mean()) / resids.std()).iloc[-1].sort_values() # residuals of the most recent day

    # Get the stocks far from mean (for mean reversion) 
    selected = zscores[:3]
    
    # Return the weights for each selected stock
    weights = selected * (1 / selected.sum())
    for stock, weight in weights.items():
        print(stock+': %.4f'%weight, end='  ')
    return weights

start = '20200610'
end = '20230220'
num_pre_trade = 100
num_equities = 45
cum_ret = [0]
 
history = get_price(tickers, start, end)
ret = history.pct_change(1)
t2 = timer()
print("数据加载用时："+str(t2-t1))
while num_pre_trade < history.shape[0]:
    t3 = timer()

    #coarse selection
    date = history.index[num_pre_trade]
    print('date:'+date)
    coarse = get_date_cv(history.columns, date)
    # Sort the equities in DollarVolume decendingly
    symbols = list(coarse[coarse['close'] > 5].sort_values('vol', ascending=[False]).head(num_equities)['ts_code'])
    test_history = history[symbols].iloc[num_pre_trade-100:num_pre_trade]

    weights = get_weight(test_history)
    test_ret = ret[weights.index].iloc[num_pre_trade]
    cur_ret = sum(weights*test_ret)
    cum_ret.append((1+cum_ret[-1])*(1+cur_ret)-1)
    print('cur ret：%.4f'%cur_ret,end='   ')
    print('cum ret：%.4f'%cum_ret[-1],end='   ')
    num_pre_trade+=1

    t4 = timer()
    print("用时："+str(t4-t3))
