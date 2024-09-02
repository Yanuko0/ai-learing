import numpy as np
import scipy
from pyexpat import features
# 5.導入模塊用來讀取wavfile的聲音文件
from scipy.io.wavfile import read
# 7.導入剛剛做的函數
from speaker_features import  extract_features
#17.引入高斯混合模型
from sklearn.mixture import GaussianMixture
#20.導入保存模塊
import  pickle

# 1.拿到聲音文件數據訓練用跟測試用(裡面放每一個語音文件位置對應文件夾下面的聲音文件)
source = '數據路徑(放聲音文件)'
train_file = '數據路徑(放聲音文件)'

#19.保存到當前目錄下面
dest = './speaker_models'

# 2.通過open打開文件
file_paths = open(train_file, 'r')

# 9.生成一個多維數組,把提取好的特徵放進去
features = np.asanyarray(())

# 15.首先先設定為1
count = 1

# 3.拿到每一個文件的路徑
for path in file_paths:
    path = path.strip()
    # 4.打印路徑
    print(path)

    # 6.讀取一個個地讀取原聲音文件 會返為sr(採樣率) + 聲音文件本身
    sr, audio = read(source + path)

    # 8.提取40個維度特徵(MFCC + △MFCC) 傳入聲音文件+採樣率 得到vector向量
    vector = extract_features(audio, sr)

#10.判斷因為一開始features是空的,所以可以看裡面裝向量的數量 如果是一開始沒裝
    if features.size == 0:
        # 11.可以把features = vector
        features = vector
        # 12.如果已經往裡面裝了(如果不為空)
    else:
        # 13.放入features跟新來的vector 做成向量
        features = np.vstack((features, vector))

    # 14.對於訓練集是每x個文件對應一個人(這邊的x是5因為例題是每5個對應一個人)
    if count == 5:
        # 16.這裡相當於是用一個人的五條語音文件, 對應來使用一個GMM模型來進行建模
        # 加上()生成算法的實例對象,這邊的16相當於1個人的5調語音文件用16個高斯分布進行擬合
        # 這邊的n_components越大, 代表擬合的越精細.max_iter代表擬合了多少次
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag')
        gmm.fit(features)

        #18. 訓練好後保存 落地保存每一個人對應的GMM模型
        picklefile = path.split('-')[0] + '.gmm'
        #21.把對象落地把存gmm模型本身,打開一個文件讓它保存在這個文件裡面
        pickle.dump(gmm, open(dest+picklefile, 'wb'))

        #17
        features = np.asanyarray(())
        count = 0
    count = count + 1