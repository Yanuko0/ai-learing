from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score

# 加載 20newsgroups 數據集 (訓練集以及測試集)
# 可以在(data_home="")指定數據所在的目錄
newsgroups_train = fetch_20newsgroups(subset="train")
newsgroups_test = fetch_20newsgroups(subset="test")

# 創建一個pipline, 用於文件特徵提取, 接著使用邏輯回歸,
# 添加(max_iter=3000)是最大回歸次數,並不是填3000就會跑到3000次可能會小於3000
pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000))

# 使用訓練集訓練模型
pipeline.fit(newsgroups_train.data, newsgroups_train.target)

# 在測試集上面預測
# newsgroups_test.data測試集的x拿到pipeline內
# 先會帶入到CountVectorizer進行特徵提取
# 最後數值化結果再交由邏輯回歸 LogisticRegression() 去做預測
y_pred = pipeline.predict(newsgroups_test.data)

# 輸出模型的準確率
print("Accuracy: %.2f" % accuracy_score(newsgroups_test.target, y_pred))