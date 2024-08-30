from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is a sample text.',
    'This text is another example text.',
    'This is just another text.'
]

# 聲明一個對象
vectorizer = CountVectorizer()
# 計算完後把結果轉換出來fit_transform
X = vectorizer.fit_transform(corpus)

print(X.toarray())

'''
[[0 0 1 0 1 1 1 1]
 [1 1 1 0 0 0 2 1]
 [1 0 1 1 0 0 1 1]]
這邊CountVectorizer是把上面每一句話變成一個項量,使得每句話項量長度為一樣的
(這樣做好處是雖然每一句話長度是不一樣的, 
但通過讓項量變成一樣,方便記憶學習算法來去做)

這邊CountVectorizer Count是計數,text 再第二行出現兩次所以變成2
'''
