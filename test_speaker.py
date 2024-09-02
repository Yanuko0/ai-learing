#音頻數據所存放的目錄
import time
import numpy as np
from sklearn.covariance import log_likelihood
from speaker_features import extract_features
from scipy.io.wavfile import read
import  pickle

source = '數據路徑(放聲音文件)'
# 模型文件存放的目錄
modelpath = './speaker_models'
# 測試集文件的名稱
test_file = 'D:/data/speaker-identification/development_set_test.txt'
# 打開測試集文件
file_paths = open(test_file, 'r')
# 得到每一個模型文件的路徑
gmm_files = [os.path.join(modelpath, fname) for fname in
             os.listdir(modelpath) if fname.endswith('.gmm')]
# 通過路徑加載每一個模型文件
models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
# 獲取每一個人的人名
speakers = [fname.splist("/")[-1].split(".gmm")[0] for fname in gmm_files]
# 遍歷測試集的file_paths文件
for path in file_paths:
    path = path.strip()
    print(path)
    # 讀取每一個測試音頻
    sr, audio = read(source + path)
    # 進行特徵提取,每一個對應40個維度
    vector = extract_features(audio, sr)

    log_likelihood = np.zeros(len(models))

    # 使用模型
    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)