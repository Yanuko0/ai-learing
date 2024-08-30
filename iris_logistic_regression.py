from sklearn import  datasets
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score

#加載鳶尾花數據集
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(y)

#將數據拆分為訓練集和測試集 ,random_state=42如果寫了每次切分出來的會固定
# 這樣機率就會一樣 沒有寫得話數值就會是隨機數
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42)

#創建一個邏輯回歸對象,這裡邏輯回歸會根據數據決定試用二分類還是多分類
# 邏輯回歸到底是把多分類轉換成多個二分類, 還是說使用的是Softmax回歸
lr = LogisticRegression() #沒有輸入的話這邊是用ovr可右建進去看 默認的是auto ()內一般情況下不會去寫
# lr = LogisticRegression(multi_class='ovr') #多分類轉成多個二分類
# lr = LogisticRegression(multi_class='multinomial') #Softmax回歸做多分類

#使用訓練集訓練模型
lr.fit(X_train, y_train)

#對測試集進行預測
y_pred = lr.predict(X_test)

#打印模型的準確率
print("準確率: %.2f" % accuracy_score(y_test, y_pred))
