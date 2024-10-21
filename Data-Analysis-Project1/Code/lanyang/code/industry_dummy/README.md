# industry_dummy

## 因子

- 行业隐变量

  使用2012证监会行业分类表。行业编号格式为“A01”，其中字母为大类行业，两位数字为细分行业。本因子采取后两位数字，共81个细分行业，为每支股票生成一个one-hot行业隐变量（所属行业位置上为1）。



## 源数据

- 2012证监会行业分类表

  TRD_Co.xlsx

  


## 使用

- 在`industry_dummy`文件夹下运行

```
python data_industry_dummy.py [-p path]
```

其中path是输出目录。若省略则直接在`industry_dummy`文件夹输出。

