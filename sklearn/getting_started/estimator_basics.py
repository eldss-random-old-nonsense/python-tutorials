from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

# Data could be viewed as:
# 1, 2, 3 => 0
# 11, 12, 13 => 1
X = [[1, 2, 3],
     [11, 12, 13]]
y = [0, 1]

clf.fit(X, y)
print(clf.predict(X))
print(clf.predict([[4, 5, 6], [14, 15, 16]]))
