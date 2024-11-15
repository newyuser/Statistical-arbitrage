import numpy as np
import pandas as pd
import statsmodels.api as sm
import fastcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression
import math

# import json
# with open('510050_20230214.json') as user_file:
#   parsed_json = json.load(user_file)
# tickers = []
# for i in parsed_json["StockComponent"].keys():
#   tickers.append(i+".SH")

import tushare as ts
ts.set_token('0742c95bee169abeccdf80bbdce1b2e0cf2ac4d83cbd932b608300fd')
pro = ts.pro_api()
tickers = pro.index_weight(index_code='000300.SH', trade_date=20210630, 
                        fields=["con_code"])["con_code"].to_list()
                        
def get_returns(tickers, start, end):
    stocks = pro.daily(ts_code=tickers[0], start_date=start, end_date=end, fields=['trade_date'])
    for ticker in tickers:
        stock = pro.daily(ts_code=ticker, start_date=start, end_date=end, fields=['trade_date', 'close'])
        stock = stock.rename(columns={"close" : ticker})
        stocks = stocks.merge(stock,on=['trade_date'])
    stocks.set_index(['trade_date'], inplace=True)
    stocks = stocks.iloc[::-1]
    stocks = stocks.dropna()
    returns = stocks.pct_change().iloc[1:] # (pre_close - close)/pre_close
    return returns

start = '20200601'
end = '20230307'
history_returns = get_returns(tickers, start, end)

cur_ret_P = []
cur_ret_H = []
cur_ret__ = []
cur_ret_ = []
cum_ret_P = [0]
cum_ret_H = [0]
cum_ret__ = [0]
cum_ret_ = [0]

date = []
window_length = 200
nb_cluster = 19
n_component = 40
for cur_num in range(window_length, history_returns.shape[0]):
    date.append(history_returns.index[cur_num-1])
    print('date:', date[-1])
    # 获取标准化实验数据
    original_returns = history_returns.iloc[cur_num-window_length:cur_num]
    returns = (original_returns - original_returns.mean())/original_returns.std()
    sorted_correlations, HPCA_corr = correlations(returns,nb_cluster)
    sorted_returns = original_returns[sorted_correlations.columns] 
    eigenvals_P, eigenvecs_P = np.linalg.eig(sorted_correlations)
    idx_P = eigenvals_P.argsort()[::-1]   
    pca_eigenvecs = eigenvecs_P[:, idx_P][:, :n_component]  
    fct_P = np.dot(sorted_returns/sorted_returns.std(),pca_eigenvecs)
    OLSmodels_P = {stock: sm.OLS(sorted_returns[stock], fct_P).fit() for stock in sorted_returns.columns}
    resids_P = pd.DataFrame({stock: model_P.resid for stock, model_P in OLSmodels_P.items()})
    zscores_P = ((resids_P - resids_P.mean()) / resids_P.std()).iloc[-1]
    long_P = zscores_P[zscores_P < -1.5]
    short_P = zscores_P[zscores_P > 1.5]

    eigenvals_H, eigenvecs_H = np.linalg.eig(HPCA_corr)
    idx_H = eigenvals_H.argsort()[::-1]   
    hpca_eigenvecs = eigenvecs_H[:, idx_H][:, :n_component]  
    fct_H = np.dot(sorted_returns/sorted_returns.std(),hpca_eigenvecs)
    OLSmodels_H = {stock: sm.OLS(sorted_returns[stock], fct_H).fit() for stock in sorted_returns.columns}
    resids_H = pd.DataFrame({stock: model_H.resid for stock, model_H in OLSmodels_H.items()}) 
    zscores_H = ((resids_H - resids_H.mean()) / resids_H.std()).iloc[-1]
    long_H = zscores_H[zscores_H < -1.5]
    short_H = zscores_H[zscores_H > 1.5]
    ########################################################
    # HPCA选股
    print('HPCA：', end='')
    if len(long_H) != 0:
        weights_long_H = long_H * (1 / long_H.sum())
        print('long:', end=' ')
        for stock, weight in weights_long_H.items():
            print(stock+': %.4f'%weight, end='  ')
    else: weights_long_H = long_H
    if len(short_H) != 0:
        weights_short_H = short_H * (1 / short_H.sum())
        print('short:', end=' ')
        for stock, weight in weights_short_H.items():
            print(stock+': %.4f'%weight, end='  ')
    else: weights_short_H = short_H
    cur_ret_H.append((sum(weights_long_H*history_returns[weights_long_H.index].iloc[cur_num]) - sum(weights_short_H*history_returns[weights_short_H.index].iloc[cur_num]))/2 - history_returns.iloc[cur_num].mean())
    cum_ret_H.append((1+cum_ret_H[-1])*(1+cur_ret_H[-1])-1)
    print('cur ret：%.4f'%cur_ret_H[-1],end='   ')
    print('cum ret：%.4f'%cum_ret_H[-1])
    # PCA选股
    print('PCA：', end='')
    if len(long_P) != 0:
        weights_long_P =  long_P * (1 / long_P.sum())
        print('long:', end=' ')
        for stock, weight in weights_long_P.items():
            print(stock+': %.4f'%weight, end='  ')
    else: weights_long_P = long_P
    if len(short_P) != 0:
        weights_short_P =  short_P * (1 / short_P.sum())
        print('short:', end=' ')
        for stock, weight in weights_short_P.items():
            print(stock+': %.4f'%weight, end='  ')
    else: weights_short_P = short_P
    cur_ret_P.append((sum(weights_long_P*history_returns[weights_long_P.index].iloc[cur_num]) - sum(weights_short_P*history_returns[weights_short_P.index].iloc[cur_num]))/2 - history_returns.iloc[cur_num].mean())
    cum_ret_P.append((1+cum_ret_P[-1])*(1+cur_ret_P[-1])-1)
    cur_ret__.append(history_returns.iloc[cur_num].mean())
    cum_ret__.append((1+cum_ret__[-1])*(1+cur_ret__[-1])-1)
    print('cur ret：%.4f'%cur_ret_P[-1],end='   ')
    print('cum ret：%.4f'%cum_ret_P[-1])
    # 交叉选股
    print('hpca&pca', end=' ')
    if len(long_P.index.intersection(long_H.index)) == 0 and len(short_P.index.intersection(short_H.index)) == 0:
        cur_ret_.append(- history_returns.iloc[cur_num].mean())
    elif len(long_P.index.intersection(long_H.index)) == 0 and len(short_P.index.intersection(short_H.index)) != 0:
        cur_ret_.append(- history_returns[short_P.index.intersection(short_H.index)].iloc[cur_num].mean() - history_returns.iloc[cur_num].mean())
    elif len(long_P.index.intersection(long_H.index)) != 0 and len(short_P.index.intersection(short_H.index)) == 0:
        cur_ret_.append(history_returns[long_P.index.intersection(long_H.index)].iloc[cur_num].mean() - history_returns.iloc[cur_num].mean())
    else:
        cur_ret_.append((history_returns[long_P.index.intersection(long_H.index)].iloc[cur_num].mean() - history_returns[short_P.index.intersection(short_H.index)].iloc[cur_num].mean())/2 - history_returns.iloc[cur_num].mean())
    cum_ret_.append((1+cum_ret_[-1])*(1+cur_ret_[-1])-1)
    print('cur ret：%.4f'%cur_ret_[-1],end='   ')
    print('cum ret：%.4f'%cum_ret_[-1])
    ########################################################


df_cur = pd.DataFrame(np.array([cur_ret_P[:],cur_ret_H[:],cur_ret_[:],cur_ret__[:]]).T, columns=['PCA alpha','HPCA alpha','hpca&pca alpha','beta'], index=np.array(date))
df_cur.plot()
plt.title('cur ret')
plt.xticks(rotation=30, fontsize=8)
plt.show()
df_cum = pd.DataFrame(np.array([cum_ret_P[1:],cum_ret_H[1:],cum_ret_[1:],cum_ret__[1:]]).T, columns=['PCA alpha','HPCA alpha','hpca&pca alpha','beta'], index=np.array(date))
df_cum.plot()
plt.title('cum ret')
plt.xticks(rotation=30, fontsize=8)
plt.show()