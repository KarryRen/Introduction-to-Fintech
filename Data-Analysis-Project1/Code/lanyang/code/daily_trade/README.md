# daily_trade

## 因子

- 27 dolvol

  t-2月的交易额(Amount)取对数

- 85 turn

  t-3到t-1月交易量(Turnover)平均值除以t月流通股数

- 86 zerotrade

  t-1月零交易量的交易日天数

- 79 std_dolvol

  当月交易额(Amount)的标准差

- 80 std_turn

  当月交易量(Turnover)的标准差

  

## 源数据

- 日交易数据文件

  交易额(Amount)，交易量(Turnover)

  链接：https://disk.pku.edu.cn:443/link/4CD07EB3F0DA0EDC847FEACDA9FD386F 有效期限：2023-04-01 23:59 访问密码：YhHq       

- TRD_Mnth

  每月交易日数量

- 027nshra

  流通股数

  

## 使用

- 将数据下载到`daily_trade`文件夹下，并解压为`LT_Dailyinfo`文件夹

- 在`daily_trade`文件夹下运行

```
python data_daily_trade.py [-p path]
```

其中path是输出目录。若省略则直接在`daily_trade`文件夹输出。