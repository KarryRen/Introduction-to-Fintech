# M_dividend

## 因子

以下宏观因子均只有2002-2022年的数据（数据格式199012-200209不变）

- **M_dp**

- - 股息是在A股市场支付的12个月的移动股息
  - 股息价格比（Dividend Price Ratio）是中国A股市场总股息的对数与加权平均股价的对数之差

- **M_de**

- 派息率（Dividend Payout Ratio）是中国A股市场上所有上市股票的股息对数与收益对数之差



## 源数据

- 股息数据 dividend

  来自乐咕乐股-A 股个股指标: 市盈率, 市净率, 股息率，使用AKShare下载

  https://disk.pku.edu.cn:443/link/810665DC51E57064CDB723AC676ED5EF 有效期限：2026-09-30 23:59 访问密码：LGPX
  
  移动股息使用**股息率ttm**乘以**市值**计算
  
  
  
- 宏观因子M_ep（市盈率的倒数，盈利/市值）

  M_dp=股息/市值

  M_de=股息/盈利=M_dp/M_ep

  之后再取对数

  


## 使用

- 下载数据至`M_dividend/dividend`文件夹
- 在`M_dividend`文件夹下运行

```
python data_M_dividend.py [-p path]
```

其中path是输出目录。若省略则直接在`M_dividend`文件夹输出。

