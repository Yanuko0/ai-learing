import numpy as np
import python_speech_features as mfcc
from fontTools.merge.util import first
from mkl import second
from  sklearn import preprocessing

# 開啟Anaconda Powershell Prompt
# 執行conda activate pythonProject激活當前環境
# 安裝pip install python_speech_features模塊

# 5.封裝一個函數(傳入數據)
def calculate_delta(array):
    # 7.實現邏輯
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i + j
            index.append((second, first))
            j += 1
            deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]])))
    # 6.最終傳回結果
    return deltas

# 1.封裝一個函數(傳入語音文本, rate提取特徵)
def extract_features(audio, rate):
    # 2.mfcc結果通過變量接收 = 調用上面mfcc函數(傳入聲音文件,20相當於生成20個特徵維度)
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy=True)
    # 3.數據歸一化scale對數據縮放
    mfcc_feat = preprocessing.scale(mfcc_feat)
    # 4.計算  8.調用calculate_delta傳入mfcc_feat 得到delta
    delta = calculate_delta(mfcc_feat)
    # 9.把mfcc_feat跟delta拼接在一起
    combined = np.hstack((mfcc_feat, delta))
    return  combined

