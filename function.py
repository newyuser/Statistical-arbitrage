import numpy as np
import pandas as pd
import statsmodels.api as sm
import fastcluster
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from sklearn.decomposition import PCA

# tushare读数据
import tushare as ts

def get_returns(start, end):
    ts.set_token('0742c95bee169abeccdf80bbdce1b2e0cf2ac4d83cbd932b608300fd')
    pro = ts.pro_api()
    tickers = pro.index_weight(index_code='000016.SH', trade_date=20210630, 
                        fields=["con_code"])["con_code"].to_list()
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

# 函数correlations 输入标准化的returns, nb_clusters 返回sorted_correlations, HPCA_corr
def correlations(returns, nb_clusters):
    ### 分层聚类--使得相关系数大的股票离得更近
    corr = returns.corr(method='pearson')   #股票returns的相关系数
    dist = 1 - corr.values
    tri_a, tri_b = np.triu_indices(len(dist), k=1)
    linkage = fastcluster.linkage(dist[tri_a, tri_b], method='ward') # 最短最长平均法做层次聚类
    permutation = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage, dist[tri_a, tri_b])) # 重新排序切割树 得到叶节点
    sorted_stocks = returns.columns[permutation]
    sorted_corrs = corr.values[permutation, :][:, permutation]
    sorted_correlations = pd.DataFrame(sorted_corrs, index=sorted_stocks, columns=sorted_stocks) # 层次聚类排序后的股票相关系数 靠近斜对角线的系数更大，对应股票相关性更强 

    ### 根据sorted_correlations将股票分成若干簇  注意！如果没有明显的资产分组，HPCA会加强一个虚假的结构
    dist = 1 - sorted_correlations.values
    dim = len(dist)
    tri_a, tri_b = np.triu_indices(dim, k=1)
    linkage = fastcluster.linkage(dist[tri_a, tri_b], method='ward')
    clustering_inds = hierarchy.fcluster(linkage, nb_clusters,
                                        criterion='maxclust')
    clusters = {i: [] for i in range(min(clustering_inds),
                                    max(clustering_inds) + 1)}
    for i, v in enumerate(clustering_inds):
        clusters[v].append(i)

    permutation = sorted([(min(elems), c) for c, elems in clusters.items()],
                        key=lambda x: x[0], reverse=False)
    sorted_clusters = {}
    for cluster in clusters:
        sorted_clusters[cluster] = clusters[permutation[cluster - 1][1]]
    # 画每簇的成员股票累计收益率 簇内股票收益率相关性强 累计收益率曲线相似 注意！若相关系数不大，曲线不相似，可以增大簇的个数
    stock_to_cluster = {}
    for cluster in sorted_clusters:
        cluster_members = sorted_correlations.columns[sorted_clusters[cluster]].tolist()
        for stock in cluster_members:
            stock_to_cluster[stock] = cluster
    ### 簇内股票收益率矩阵 右乘 簇内相关系数矩阵的第一特征向量 除以 最大特征值的平方 得收益率降维后的值
    eigen_clusters = {}
    for cluster in clusters:
        cluster_members = sorted_correlations.columns[
            sorted_clusters[cluster]].tolist()
        corr_cluster = sorted_correlations.loc[
            cluster_members, cluster_members]
        cluster_returns = returns[cluster_members]
        eigenvals, eigenvecs = np.linalg.eig(corr_cluster.values) # 簇内相关系数的特征值、特征向量
        idx = eigenvals.argsort()[::-1]
        eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
        val1, vec1= eigenvals[0], eigenvecs[:, 0] # 簇内相关系数的最大特征值、第一特征向量
        F1 = (1 / np.sqrt(val1)) * np.dot(cluster_returns.values,vec1) # 得簇内收益率降维后的值 392*10 × 10*1 = 392*1  将股票收益率整合到方差最大的线性方向上
        eigen_clusters[cluster] = F1
    # 股票收益率和对应簇内降维收益率的线性回归斜率
    betas = {}
    for stock in returns.columns:
        stock_returns = returns[stock]
        cluster_F1 = eigen_clusters[stock_to_cluster[stock]]
        reg = LinearRegression(fit_intercept=False).fit(
            cluster_F1.reshape(-1, 1), stock_returns)
        beta = reg.coef_[0]
        betas[stock] = beta 
    ### 更新不同簇的股票之间的相关系数 体现出簇间相关性
    HPCA_corr = sorted_correlations.copy()
    for stock_1 in HPCA_corr.columns:
        beta_1 = betas[stock_1]
        F1_1 = eigen_clusters[stock_to_cluster[stock_1]]
        for stock_2 in HPCA_corr.columns:
            beta_2 = betas[stock_2]
            F1_2 = eigen_clusters[stock_to_cluster[stock_2]]
            if stock_to_cluster[stock_1] != stock_to_cluster[stock_2]: # 对不同簇股票的相关系数调整 体现簇间差异
                rho_sector = np.corrcoef(F1_1, F1_2)[0, 1] # 簇间相关性
                mod_rho = beta_1 * beta_2 * rho_sector
                HPCA_corr.at[stock_1, stock_2] = mod_rho
    return sorted_correlations, HPCA_corr

# KMO测度函数 输入相关系数矩阵 返回kmo_value测度值
def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value