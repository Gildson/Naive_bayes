import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

base = pd.read_csv('spam.csv')

print(base)

X = base.iloc[:, [1]].values
Y = base.iloc[:, 0].values

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)

prediction = naive_bayes.predict(X_test)

print(accuracy_score(Y_test, prediction))