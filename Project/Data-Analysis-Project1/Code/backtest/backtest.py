import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def get_daily_return():
    DATA_DIR=r"./data"
    volume = pd.read_hdf(f'{DATA_DIR}/volume.h5', key='df')
    amount = pd.read_hdf(f'{DATA_DIR}/amount.h5', key='df')
    volume = volume[volume.index>pd.to_datetime('2024-01-01')]
    amount = amount[amount.index>pd.to_datetime('2024-01-01')]
    backadj = pd.read_hdf(f'{DATA_DIR}/back_adj.h5', key='df')
    backadj = backadj[backadj.index>=pd.to_datetime("2024-01-01")]
    backadj = backadj.groupby(backadj.index.date).mean()
    def part_sum(df ):
            res = df.between_time('09:30',"09:40")
            res = res.groupby(res.index.date).sum()
            return res
    trade_vwap = part_sum(amount)/part_sum(volume) *backadj 
    daily_return = trade_vwap.shift(-1)/trade_vwap - 1
    daily_return.index = daily_return.index.astype(str).str.replace('-','')
    daily_return.columns = daily_return.columns.str.lower()
    daily_return.index = daily_return.index.astype(str)
    return daily_return

class Backtest:
    def __init__(self,factor_df, factor_name, transaction_rate,holding_days,start_date,end_date,ret):
        """
        factor_df: 因子DataFrame
        factor_name: 因子名
        transaction_rate:交易费率
        holding_days:换仓频率
        start_date:回测起始日期
        end_date:回测结束日期
        ret:日收益率DataFrame
        """
        # 处理因子
        self.factor_df = factor_df.shift(1).dropna(how='all', axis=0)
        self.factor_df.index = self.factor_df.index.astype(str)
        ret.columns = ret.columns.str.lower()
        ret.index = ret.index.astype(str)

        self.factor_df = self.factor_df[(self.factor_df.index>=start_date) &(self.factor_df.index<=end_date)]

        # self.factor_df = self.factor_df.where()
        # 根据调仓周期，同周期的因子值不变
        if holding_days>1:
            self.factor_df = self._adjust_holding_days(factor_df=self.factor_df,holding_days=holding_days)
        # self参数赋值
        self.factor_name = factor_name
        self.transaction_rate = transaction_rate
        self.holding_days = holding_days
        self.ret = ret
        self.groupNum = 10

    def run(self):
        """ 进行回测 """
        factor_new = self.factor_df
        # benchmark = self.benchmark.reindex(factor_new.index)
        # 获取超额
        ret_1d = self.ret.reindex_like(factor_new)
        print(ret_1d)
        ic = factor_new.corrwith(ret_1d, axis=1,method='spearman')
        tot_ic = ic.mean()
        tot_ir = tot_ic / ic.std()

        # 做多前50%，做空后50%，根据排序赋权，得到pnl
        pos_df = factor_new.rank(axis=1, pct=True) - 0.5
        pos_df[pos_df > 0] = pos_df[pos_df > 0].div(pos_df[pos_df > 0].sum(axis=1), axis=0)
        pos_df[pos_df < 0] = -pos_df[pos_df < 0].div(pos_df[pos_df < 0].sum(axis=1), axis=0)
        
        pnl_long = (pos_df[pos_df > 0] * ret_1d).sum(axis=1) 
        pnl_short = (pos_df[pos_df < 0] * ret_1d).sum(axis=1) 
        pnl = (pnl_long + pnl_short) / 2
        print(pnl)
        # 剔除换手成本
        pos_change = (pos_df - pos_df.shift(1).fillna(0)).abs()
        pos_volume = pos_df.abs().fillna(0)
        turnover = pos_change.sum(axis=1) / pos_volume.sum(axis=1).fillna(0)
        tot_turnover = turnover.mean()
        pnl = pnl - turnover*self.transaction_rate

        # 获取截面分组
        group_df = factor_new.apply(lambda x:self._quantile_calc(x, self.groupNum),axis=1)
        ones_df = pd.DataFrame().reindex_like(factor_new).fillna(1)
        groups_turnover = {}

        for i in range(1, self.groupNum+1):
            temp = ones_df.where(group_df==i)
            groups_turnover[i] = self._get_turnover(temp.fillna(0))
        groups_turnover_df = pd.DataFrame(groups_turnover)
        

        fig = plt.figure(figsize=(24, 12))
        # 累计收益率图
        ax0 = fig.add_subplot(221)
        pnl_long.cumsum().plot(label='longret', ax=ax0)
        pnl.cumsum().plot(label='ret', ax=ax0)
        ax0.set_xlabel('date')
        ax0.grid()
        ax0.legend()
        ax0.title.set_text(f"(long)ret")

        # IC图
        ax1 = fig.add_subplot(222)
        ic.plot(label='IC', ax=ax1)
        ax1.set_xlabel('date')
        ax1.grid()
        ax1.legend()
        ax1.title.set_text(f"IC:{np.round(tot_ic,3)} IR:{np.round(tot_ir,3)}")

        # 截面分组和return进行merge
        merged = pd.concat([group_df.unstack(),ret_1d.unstack()],axis=1).reset_index()
        merged.columns = ['code','date','group','return']

        # 分组平均收益图
        ax2 = fig.add_subplot(223)
        t = merged.groupby(['group'])['return'].mean()
        t.plot.bar(ax2)
        ax2.title.set_text(f"Mean of group return")

        # 分组累计收益图
        ax3 = fig.add_subplot(224)
        groups_pnl_df = merged.groupby(['date', 'group'])['return'].mean().unstack(level=1) 
        groups_cumsum_ret = (groups_pnl_df-groups_turnover_df*self.transaction_rate).cumsum()
        groups_cumsum_ret.plot(rot=-30, ax=ax3)
        ax3.grid()
        ax3.legend()
        ax3.title.set_text(f"Accumulation of group return")

        #收益率相关指标
        tot_ret = round(pnl.mean() * 242, 4)
        tot_long_ret = round(pnl_long.mean() * 242, 4)
        tot_short_ret = round(pnl_short.mean() * 242, 4)
        # sharpe
        tot_sharpe = round(pnl.mean() / pnl.std() * np.sqrt(242), 2)
        # 胜率
        tot_win_rate = len(pnl[pnl > 0]) / len(pnl)
        # 计算指标表格
        temp = pnl.to_frame('pnl')
        temp['turnover'] = turnover
        temp['year'] = [int(x[:4]) for x in temp.index.tolist()]
        temp['dateStr'] = temp.index
        # 总指标
        tot_info = pd.DataFrame(
            [tot_ret, tot_long_ret, tot_short_ret, tot_sharpe, tot_turnover, tot_win_rate, tot_ic, tot_ir], ["tot_ret", "tot_long_ret", "tot_short_ret", "tot_sharpe"
                , "tot_turnover", "tot_win_rate", "tot_ic", "tot_ir"])
        # 每年的指标
        yearly_info = pd.DataFrame(index=list(temp['year'].unique()),
                            columns=['return', 'pnl_mean', 'win_rate', 'sharpe', 'turnover', 'drawdown', 'dd_sdate',
                                        'dd_edate', 'trade_days'])
        yearly_info['return'] = temp.groupby('year')['pnl'].sum()
        yearly_info['pnl_mean'] = temp.groupby('year')['pnl'].mean()
        yearly_info['win_rate'] = temp[temp['pnl'] > 0].groupby('year')['pnl'].count() / temp.groupby('year')['pnl'].count()
        yearly_info['sharpe'] = temp.groupby('year')['pnl'].mean() / temp.groupby('year')['pnl'].std() * np.sqrt(242)
        yearly_info['turnover'] = temp.groupby('year')['turnover'].mean()
        yearly_info['trade_days'] = temp.groupby("year")['pnl'].count()
        yearly_info[['drawdown', 'dd_sdate', 'dd_edate']] = temp.groupby('year')['pnl'].apply(self._get_dd).apply(pd.Series)
        print(tot_info.T)
        print("-"*100)
        print(yearly_info)
        plt.show()
    def _get_dd(self,pnl):
        cum_pnl = pnl.cumsum() + 1
        max_drawdown = 0
        start_index = 0
        end_index = 0
        # 遍历净值数据，计算最大回撤
        for i in range(1, len(cum_pnl)):
            drawdown = (max(cum_pnl.iloc[:i]) - cum_pnl.iloc[i]) / max(cum_pnl.iloc[:i])
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                start_index = cum_pnl.index[np.argmax(cum_pnl[:i])]
                end_index = cum_pnl.index[i]
        return np.round(max_drawdown, 6), start_index, end_index
    
    def _quantile_calc(self,x, _quantiles=10):
        """计算截面分组"""
        quantiles = pd.qcut(x, _quantiles, labels=False, duplicates='drop') + 1
        return quantiles
    
    def _adjust_holding_days(self,factor_df,holding_days):
        """ 根据换仓频率处理因子值"""
        adjust_factor_df = factor_df[0::holding_days]
        adjust_factor_df = adjust_factor_df.reindex_like(factor_df)
        adjust_factor_df = adjust_factor_df.ffill(limit = holding_days - 1 )
        return adjust_factor_df
    
    def _get_turnover(self,pos_df):
        """根据持仓计算换手率"""
        pos_df = pos_df.div(pos_df.sum(axis=1),axis=0)
        pos_change = (pos_df - pos_df.shift(1).fillna(0)).abs()
        turnover = pos_change.sum(axis=1)
        return turnover

if __name__ =="__main__":
    daily_return= pd.read_csv(r'D:\vsCode\VScodeProject\pyProject\Pyproject\Courses\Fintech\Introduction-to-Fintech-DAPs\Data\backtest\daily_return.csv',index_col=0)
    
    factor = pd.read_csv(r"D:\vsCode\VScodeProject\pyProject\Pyproject\Courses\Fintech\Introduction-to-Fintech-DAPs\Data\Factor\20240102_20240529.csv",index_col=0)

    # 因子1 回测
    bt = Backtest(factor_df= factor, factor_name='test', transaction_rate=0 ,holding_days=1,start_date='20240102',end_date='20240529',ret=daily_return)
    bt.run()