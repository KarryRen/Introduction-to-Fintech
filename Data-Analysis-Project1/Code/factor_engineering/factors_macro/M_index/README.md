# M_index

## 因子



- M_svar

  股票方差（Stock variance），上证综合指数日收益的平方和

  注：由于月交易日数量不同，实际计算为日收益平方和**月平均值**，似乎更加恰当

  


## 源数据

- 股票指数

  TRD_Index.xlsx

  


## 使用

- 在`M_index`文件夹下运行

```
python data_M_index.py [-p path]
```

其中path是输出目录。若省略则直接在`M_index`文件夹输出。

