import pandas as pd
import jieba
import re
from gensim.models import word2vec


# 分词
def tokenizer(text): 
    return [word for word in jieba.lcut(text) if word not in stop_words]


# 去停用词
def get_stop_words():
    file_object = open('data/stopwords.txt',encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words


traindata = pd.read_csv('data/train.tsv', sep='\t')
validata = pd.read_csv('data/validation.tsv', sep='\t')
traindata.head()

totaldata = pd.concat([traindata, validata])

totaldata.to_csv('data/text.tsv', columns=['text'], index=0)
text = pd.read_csv('data/text.tsv')

stop_words = get_stop_words()
stop_words[0:5]

text_cut = []
for row in text.itertuples():
    seg = tokenizer(row[1])
    text_cut.append(seg)
print(text_cut[0:5])

text_concat=[]
for seg in text_cut:
    seg_concat=[" ".join(word for word in seg)]
    text_concat.append(seg_concat)
print(text_concat[0:5])

corpus = pd.DataFrame(data=text_concat)

corpus.to_csv('data/corpus.tsv', header=0, index=0)

sentences = word2vec.LineSentence('data/corpus.tsv')
model = word2vec.Word2Vec(sentences, min_count=5)# 待考虑


# pairs = [
#     ('备胎', '硬伤'),
#     ('车', '用处'),
#     ('百万', '空调'),
# ]
# for w1, w2 in pairs:
#     print('%r\t%r\t%.2f' % (w1, w2,model.wv.similarity(w1, w2)))

# vector = model.wv['车']

word_vecors = model.wv

model.save("comment.model")

new_model = word2vec.Word2Vec.load('comment.model')
new_model.wv.save_word2vec_format('data/myvector.vector', binary=False)

