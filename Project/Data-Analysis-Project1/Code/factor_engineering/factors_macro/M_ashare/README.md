# M_ashare

## 因子

以下宏观因子均只有2007-2022年的数据（数据格式199012-200209不变）

- M_ntis

  净股本扩张（Net Equity Expansion），A股净发行量/年末总市值

  注：使用发行市值的月度差值计算净发行量。可能存在偏差，例如未考虑退市。

- M_ep

  市盈率（Price Earnings Ratio）的倒数，总盈利/总市值

  使用上交所、深交所的市盈率加权计算

- M_mtr

  月交易额/总市值



## 源数据

- 上交所_深交所概况月度文件

  CME_Mstock2.xlsx

  


## 使用

- 在`M_ashare`文件夹下运行

```
python data_M_ashare.py [-p path]
```

其中path是输出目录。若省略则直接在`M_ashare`文件夹输出。

