# return

## 因子

- 40 maxret

  t-1月最大日收益率

- 41 mom12m

  t-11月到t-1月总收益

- 42 mom1m

  t-1月总收益

- 43 mom6m

  t-5月到t-1月总收益

- 44 mom36m

  t-36月到t-12月总收益

- 68 volatility

  t-1月日收益标准差

- **ground truth**

  t月总收益$r_{it}$

  

## 源数据

- 日收益率文件

  链接：https://disk.pku.edu.cn:443/link/07762E56DA60616E4F4B8846C20AC8A7 有效期限：2024-04-01 23:59 访问密码：RuGd

  

## 使用

- 将数据下载到`return`文件夹下，并解压为`TRD_Dalyr`文件夹

- 在`return`文件夹下运行

```
python data_return.py [-p path]
```

其中path是输出目录。若省略则直接在`return`文件夹输出。