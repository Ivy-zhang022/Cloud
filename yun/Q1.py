import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import pyLDAvis.gensim_models
import pyLDAvis

# 1. 你的数据
texts = [
    "They finally caught his bitch ass!!!! Haha im happy :) #HappyAsFuck #ThugShit #USA #PrayForBoston",
    "So glad they caught both of the bombers...! Justice will prevail... #prayforboston",
    "@cnnbrk 2nd suspect in Boston bombing with white hat look far left he leaves after bomb went off/",
    "BREAKING NEWS: BOSTON POLICE GOT EM!!!!! 2nd Suspect in Custody!!! #boston",
    "Interesting post from 4chan... Suspect and backpack that exploded in red, 8-year old victim who died in blue.",
    "Don't need feds to solve the #bostonbombing when we have #4chan!!",
    "PIC: Comparison of #Boston suspect Sunil Tripathi's FBI-released images/video and his MISSING poster You decide.",
    "I'm not completely convinced that it's this Sunil Tripathi fellow—",
    "Brutal lo que se puede conseguir en colaboración. #4Chan analizando fotos de la maratón de #Boston atando cabos...",
    "4chan and the bombing. just throwing it out there:"
]

# 2. 数据清洗 + 分词 + 词形还原
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"http\S+", "", text)  # 去链接
    text = re.sub(r"[^a-zA-Z]", " ", text)  # 去除非字母
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

processed_texts = [preprocess(t) for t in texts]

# 3. 构建字典与语料
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# 4. 训练 LDA 模型
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20, random_state=42)

# 5. 输出每个主题词云图
for i in range(3):
    words = dict(lda_model.show_topic(i, 30))
    wc = WordCloud(width=800, height=400).generate_from_frequencies(words)
    wc.to_file(f"topic_{i+1}_wordcloud.png")
    print(f"已保存词云：topic_{i+1}_wordcloud.png")

# 6. 热力图（主题-文档分布）
topic_distributions = []
for doc in corpus:
    topic_probs = lda_model.get_document_topics(doc, minimum_probability=0)
    topic_distributions.append([prob for _, prob in topic_probs])

plt.figure(figsize=(10, 6))
sns.heatmap(topic_distributions, annot=True, cmap="YlGnBu", xticklabels=[f"Topic {i+1}" for i in range(3)])
plt.title("Document-Topic Heatmap")
plt.savefig("lda_heatmap.png")
plt.show()

# 7. pyLDAvis 可视化（生成HTML文件）
panel = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(panel, 'lda_vis.html')
print("✅ pyLDAvis 可视化已保存：lda_vis.html")
