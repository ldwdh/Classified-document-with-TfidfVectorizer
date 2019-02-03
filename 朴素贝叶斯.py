from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import jieba
import os
#停用词表地址
stop_words_path ='text classification\\stop\\stopword.txt'
#训练集根目录
train_base_path='text classification\\train\\'
#测试集根目录
test_base_path='text classification\\test\\'
#文档类别
train_labels=['体育','女性','文学','校园']
test_labels=['体育','女性','文学','校园']
#读取文件夹中各个类别下的文档数据
def get_data(base_path,labels):
    contents=[]
    for label in labels:
     files={fileName for fileName in os.listdir(base_path+label)}
     try:
        for fileName in files:
            file=open(base_path+label+'\\'+fileName, encoding='gb18030')
            word=jieba.cut(file.read())
            contents.append(" ".join(word))
     except Exception:
        print(fileName+'文件读取失败')
    return contents
#1.对文档进行分词
#获取训练集与测试集
train_contents=get_data(train_base_path,train_labels)
test_contents=get_data(test_base_path,test_labels)
#2.加载停用词表
stop_words = [line.strip() for line in open(stop_words_path, encoding='utf-8-sig').readlines()]
#3.计算单词权重
tf=TfidfVectorizer(stop_words=stop_words,max_df=0.5)
train_features=tf.fit_transform(train_contents)
#4.生成多项式朴素贝叶斯分类器
train_labels=['体育']*1337+['女性']*954+['文学']*766+['校园']*249
clf=MultinomialNB(alpha=0.001).fit(train_features,train_labels)
#5.用生成的分类器做预测
test_tf=TfidfVectorizer(stop_words=stop_words,max_df=0.5,vocabulary=tf.vocabulary_)
test_features=test_tf.fit_transform(test_contents)
predicted_labels=clf.predict(test_features)
#6.计算准确率
test_labels=['体育']*115+['女性']*38+['文学']*31+['校园']*16
print('准确率',metrics.accuracy_score(test_labels,predicted_labels))