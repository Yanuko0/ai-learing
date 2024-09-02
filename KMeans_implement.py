'''
import numpy as np
from sklearn.cluster import KMeans

#定義幾個點
X = np.array([[1, 2],[1, 4],[1, 0],
              [4, 2],[4, 4], [4, 0]])

# 初始化KMeans聚類模型
kmeans = KMeans( n_clusters=2, random_state=0 )

# 使用數據訓練模型 因為Kmean是無監督的記憶學習,只需要X不用y
kmeans.fit(X)

# 輸出聚類結果
print(kmeans.labels_)

#輸出模型的聚類中心點
print(kmeans.cluster_centers_)
'''

# 導入科學計算numpy
import numpy as np
# 繪圖
import matplotlib.pyplot as plt

#我們自己去實現KMeans算法,不依賴任何機器學習的庫
def kmeans(X, k, max_iter=100):
    #隨機初始化K個中心點
    centers = X[np.random.choice(X.shape[0], k, replace=False)]
    #這個labels集合我們就存放KMeans給樣本打的標籤
    labels = np.zeros(X.shape[0])

    for i in range(max_iter):
        #分配樣本到最近的中心點
        # 求的是歐式距離
        distances = np.sqrt(((X- centers[:, np.newaxis])**2).sum(axis=2))
        # 看每一個樣本離哪個中心點更近,也就是距離更小
        new_labels = np.argmin(distances, axis= 0)

        # 更新中心點
        for j in range(k):
            centers[j] = X[new_labels == j].mean(axis=0)

        # 如果聚類的結果沒有變化, 則提前結束迭代任務
        if (new_labels == labels).all():
            break
        else:
            labels = new_labels
    return labels, centers

#生成數據集, 生成了三個組的數據
X = np.vstack((np.random.randn(100, 2) * 0.75 + np.array([1, 0]),
               np.random.randn(100, 2) * 0.25 + np.array([-0.5, 0.5]),
               np.random.randn(100, 2) * 0.5 + np.array([-0.5, -0.5])))

#運行KMeans聚類方法
labels, centers = kmeans(X, k=3)

#可視化聚類結果
plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centers[:,0], centers[:,1], marker= 'x', s=200, linewidths=3, color='r')
plt.show()