from sklearn.preprocessing import StandardScaler

X = [[0, 15],
     [1, -10]]

print(StandardScaler().fit(X).transform(X))
