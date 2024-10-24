# M_chinese_economy

## 因子

以下宏观因子均只有2007-2022年的数据（数据格式199012-200209不变）

- M_infl

  居民消费价格指数（CPI）

- M_m2gr

  M2货币同比增速

- M_itgr

  国际贸易量同比增速



## 源数据

注：以下数据来自CSMAR，原文来自国家统计局。经比对，两者相符。

- 居民消费价格分类指数月度文件CME_Mpi1

  M_infl（CPI）

- 货币供应量月度文件CME_Mfinamkt1

  M_m2gr

- 全国进出口情况月度文件CME_Mftrd1

  M_itgr

  


## 使用

- 在`M_chinese_economy`文件夹下运行

```
python data_M_chinese_economy.py [-p path]
```

其中path是输出目录。若省略则直接在`M_chinese_economy`文件夹输出。

