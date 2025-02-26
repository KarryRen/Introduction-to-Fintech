import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from wordcloud import WordCloud
from snownlp import SnowNLP, sentiment#情感分析库
import networkx as nx
#中文字符设定
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

def loadData(file):
    df = pd.read_csv(file)
    df['update_time'] = pd.to_datetime(df['update_time'], format='%Y-%m-%d %H:%M:%S')
    return df

def wordcloudAnalyse(df,top_num=10):
    
    # 分词
    text = ' '.join(df['title_content'].tolist())
    words = list(jieba.cut(text))

    # 去除停用词
    stopwords = [' ','[',']','，', '。', '！', '？', '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们','今天','明天','中国','平安','都','资讯','2023']
#     stopwords = pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'],encoding='utf-8')  #quoting=3全部引用
    words = [w for w in words if w not in stopwords and len(w)>=2]
    word_count = {}
    for word in words:
        if word in word_count:
            word = re.escape(word)
            word_count[word] += df[df['title_content'].str.contains(word)]['read_num'].sum()
        else:
            word = re.escape(word)
            word_count[word] = df[df['title_content'].str.contains(word)]['read_num'].sum()

    # 绘制主题词词云图
    wordcloud = WordCloud(width=800,height=600,background_color="white",font_path="msyh.ttc")
    wordcloud.generate_from_frequencies(word_count)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return word_count


def investSentimentAnalyse(df, sentiment_thres=0.5):
    titles = df['title_content'].tolist()
    sentiments = []
    for title in titles:
        s = SnowNLP(title)
        sentiments.append(s.sentiments)

    # 统计情感分布
    positive_num = len(
        [sentiment for sentiment in sentiments if sentiment > sentiment_thres])
    negative_num = len(
        [sentiment for sentiment in sentiments if sentiment < sentiment_thres])
    neutral_num = len([
        sentiment for sentiment in sentiments if sentiment == sentiment_thres
    ])

    # 输出结果
    print(f'积极评论数：{positive_num}，占比：{positive_num/len(sentiments):.2%}')
    print(f'消极评论数：{negative_num}，占比：{negative_num/len(sentiments):.2%}')
    print(f'中性评论数：{neutral_num}，占比：{neutral_num/len(sentiments):.2%}')


    
def topicRelationAnalyse(df,top_num=10):
    # 将主题进行分词
    df['seg_title'] = df['title_content'].apply(lambda x: ' '.join(jieba.cut(x)))

    text = ' '.join(df['title_content'].tolist())
    words = list(jieba.cut(text))
    # 去除停用词
    stopwords = [' ','[',']','，', '。', '！', '？', '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们', '它们','今天','明天','中国','平安','都','资讯','2023']
    words = [w for w in words if w not in stopwords and len(w)>=2]

    word_count={}
    for word in words:
        if word in word_count:
            word_count[word]+= df[df['title_content'].str.contains(word)]['read_num'].sum()
        else:
            word_count[word] = df[df['title_content'].str.contains(word)]['read_num'].sum()
        
    # 取出出现次数最多的前top_num个词汇
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    top_words = [x[0] for x in sorted_word_count[:top_num]]
    print(f"出现次数最多的前{top_num}个词汇：")
    print(top_words)
    df['seg_title'] = df['title_content'].apply(lambda x: ' '.join(jieba.cut(x)))

    # 构建图
    G = nx.Graph()
    for text in df['seg_title']:
        words = set(text.split())
        for word1 in words:
            if word1 in top_words:
                for word2 in words:
                    if word1 != word2 and word2 in top_words:
                        if G.has_edge(word1, word2):
                            G[word1][word2]['weight'] += 1
                        else:
                            G.add_edge(word1, word2, weight=1)

    # 绘制图
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight']*0.1 for u, v in G.edges()])
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='Microsoft YaHei')
    plt.axis('off')
    plt.show()



