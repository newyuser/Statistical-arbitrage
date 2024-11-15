import numpy as np
import pandas as pd
import statsmodels.api as sm
import fastcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression

# import json
# with open('510050_20230214.json') as user_file:
#   parsed_json = json.load(user_file)
# tickers = []
# for i in parsed_json["StockComponent"].keys():
#   tickers.append(i+".SH")

import tushare as ts
ts.set_token('0742c95bee169abeccdf80bbdce1b2e0cf2ac4d83cbd932b608300fd')
pro = ts.pro_api()
tickers = pro.index_weight(index_code='000016.SH', trade_date=20210630, 
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

start = '20200610'
end = '20210930'
# start = '20211001'
# end = '20230228'
original_returns = get_returns(tickers, start, end)

vars = np.array([0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
means = pd.DataFrame(index=vars,columns=['alpha','alpha+beta'])
stds = pd.DataFrame(index=vars,columns=['alpha','alpha+beta'])
cum_ret = pd.DataFrame(index=vars,columns=['alpha','alpha+beta'])
for var in vars:    
    print('cur var:',str(var))
    cur_ret_P = []
    cur_ret_P_ = []
    cum_ret_P = [0]
    cum_ret_P_ = [0]
    date = []
    time_length = 100 # time window
    t1 = timer()
    # num_long_short = pd.DataFrame(columns=['long','short'],index=original_returns.index)
    # z_df = pd.DataFrame(columns=original_returns.index,index=original_returns.columns)
    for cur_num in range(time_length, original_returns.shape[0]):
        date.append(original_returns.index[cur_num-1])
        # print('date:', date[-1])
        # 获取标准化实验数据
        returns = original_returns.iloc[cur_num-time_length:cur_num]
        returns = (returns - returns.mean())/returns.std()
        eigenvals, eigenvecs = np.linalg.eig(returns.corr())
        idx = eigenvals.argsort()[::-1]   
        pca_eigenvals = eigenvals[idx]/sum(eigenvals)
        for i in range(0, len(pca_eigenvals)+1):
            if sum(pca_eigenvals[:i]) >= 0.9:
                pca_eigenvecs = eigenvecs[:, idx][:, i]
                break
        factors_P = sm.add_constant(np.dot(returns, pca_eigenvecs))
        OLSmodels_P = {stock: sm.OLS(returns[stock], factors_P).fit() for stock in returns.columns}
        resids_P = pd.DataFrame({stock: model_P.resid for stock, model_P in OLSmodels_P.items()})
        zscores_P = ((resids_P - resids_P.mean()) / resids_P.std()).iloc[-1]
        long_P = zscores_P[zscores_P < -1.5]
        short_P = zscores_P[zscores_P > 1.5]
        # PCA选股
        if len(long_P) != 0:
            weights_long_P =  long_P * (1 / long_P.sum())
            # print('long:', end=' ')
            # for stock, weight in weights_long_P.items():
            #     print(stock+': %.4f'%weight, end='  ')
        else: weights_long_P = long_P
        if len(short_P) != 0:
            weights_short_P =  short_P * (1 / short_P.sum())
            # print('short:', end=' ')
            # for stock, weight in weights_short_P.items():
            #     print(stock+': %.4f'%weight, end='  ')
        else: weights_short_P = short_P
        cur_ret_P.append((sum(weights_long_P*original_returns[weights_long_P.index].iloc[cur_num]) - sum(weights_short_P*original_returns[weights_short_P.index].iloc[cur_num]))/2 - original_returns.iloc[cur_num].mean())
        cur_ret_P_.append((sum(weights_long_P*original_returns[weights_long_P.index].iloc[cur_num]) - sum(weights_short_P*original_returns[weights_short_P.index].iloc[cur_num]))/2)
        cum_ret_P.append((1+cum_ret_P[-1])*(1+cur_ret_P[-1])-1)
        cum_ret_P_.append((1+cum_ret_P_[-1])*(1+cur_ret_P_[-1])-1)
        # print('cur ret：%.4f'%cur_ret_P[-1],end='   ')
        # print('cum ret：%.4f'%cum_ret_P[-1])
        # num_long_short.loc[date[-1]] = [len(long_P),len(short_P)]
        # z_df[date[-1]] = zscores_P
    # num_long_short = num_long_short.dropna()
    # z_df = z_df.dropna(axis=1)
    df_cur = pd.DataFrame(np.array([cur_ret_P[:],cur_ret_P_[:]]).T, columns=['alpha','alpha+beta'], index=np.array(date))
    df_cur.plot()
    plt.title('cur ret--var:{}'.format(var))
    plt.xticks(rotation=30, fontsize=8)
    plt.show()
    dcb = df_cur.describe()
    means.loc[var] = dcb.loc['mean']
    stds.loc[var] = dcb.loc['std']
    cum_ret.at[var,'alpha'] = cum_ret_P[-1]
    cum_ret.at[var,'alpha+beta'] = cum_ret_P_[-1]
    df_cum = pd.DataFrame(np.array([cum_ret_P[1:],cum_ret_P_[1:]]).T, columns=['PCA','alpha+beta'], index=np.array(date))
    df_cum.plot()
    plt.title('cum ret--var:{}'.format(var))
    plt.xticks(rotation=30, fontsize=8)
    plt.show()
    t2 = timer()
    print('用时：', t2-t1)
means.plot()
plt.title('means')
plt.xticks(rotation=30, fontsize=8)
plt.show()
stds.plot()
plt.title('stds')
plt.xticks(rotation=30, fontsize=8)
plt.show()
cum_ret.plot()
plt.title('cum_ret')
plt.xticks(rotation=30, fontsize=8)
plt.show()
