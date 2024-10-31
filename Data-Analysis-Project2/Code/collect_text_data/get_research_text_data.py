# -*- coding: utf-8 -*-
# @Time    : 2024/10/19 17:00
# @Author  : YiMing Jiang

""" Get the research text data from the website.

https://data.eastmoney.com/report/stock.jshtml.

"""

import requests
import demjson3
import json
import pandas as pd
from bs4 import BeautifulSoup


msg = json.loads(requests.post("https://reportapi.eastmoney.com/report/list2",
                               data=demjson3.encode({"beginTime": '2019-01-01', "endTime": '2024-06-01', "pageNo": 1}),
                               headers={"content-type": "application/json"}).text)
df = pd.DataFrame(msg["data"])
for i in range(2, msg["TotalPage"] + 1):
    msg_ = json.loads(requests.post("https://reportapi.eastmoney.com/report/list2",
                                    data=demjson3.encode({"beginTime": '2019-01-01', "endTime": '2024-06-01', "pageNo": i}),
                                    headers={"content-type": "application/json"}).text)
    df = pd.concat([df, pd.DataFrame(msg_["data"])])
df["url"] = df["infoCode"].apply(lambda x: "https://data.eastmoney.com/report/info/" + str(x) + ".html")
for idx in df.index:
    df.loc[idx, "content"] = BeautifulSoup(requests.get(df.loc[idx, "url"]).text, features='lxml').select('div.newsContent')[0].get_text().replace(
        "\n", "").replace(" ", "").replace("\u3000", "").replace("\r", "")

df.to_csv("./个股研报列表(含内容).csv", index=False, encoding='utf-8-sig')
