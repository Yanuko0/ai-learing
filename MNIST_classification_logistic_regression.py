import pandas
from sklearn.datasets import fetch_openml
import numpy as np
# 導入繪圖模塊plt是起的別名
import matplotlib.pyplot as plt
# 導入預處理
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from iris_logistic_regression import y_pred


# 1.載入MNIST手寫數字圖片數據集
mnist = fetch_openml("mnist_784")

img0 = np.array(mnist.data)[0]
# print(np.array(mnist.target)[0])
# # 把圖片變成寬28高28的
# img0 = img0.reshape(28, 28)
# # 圖片繪製
# plt.imshow(img0, cmap='gray')
# # 展示彈出來
# plt.show()

# 2.數據預處理: 標準歸一化
scaler = StandardScaler()
# 3.fit進行計算均值和標準差,transform就是應用這個均值和標準差
X = scaler.fit_transform(mnist.data)

# 4.劃分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, mnist.target, test_size=0.2, random_state=42)

# 不用歸一化的話
# X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# 5.創建邏輯回歸模型(max_iter=1000為最大迭代次數, 也可以不限制)
model = LogisticRegression(max_iter=1000)

# 6.在訓練集上訓練模型
model.fit(X_train, y_train)

#在測試集上進行預測
y_pred = model.predict(X_test)

#輸出模型的準確率
print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

print(model.predict([img0]))

#去Anaconda Powershell Prompt安裝模塊
# pip install pandas


# print(img0.shape)
#圖片繪製出來安裝 pip install matplotlib

