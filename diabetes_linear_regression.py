from sklearn import  datasets
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from  sklearn.metrics import  mean_squared_error

#加載糖尿病數據集
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

#將數據集拆分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 創建一個多元現性回歸算法對象
lr = LinearRegression()

# 使用訓練集
lr.fit(X_train, y_train)

# 使用測試集進行預測
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

#打印模型的均方差
print("均方誤差: %.2f" % mean_squared_error(y_train, y_pred_train))
print("均方誤差: %.2f" % mean_squared_error(y_test, y_pred_test))