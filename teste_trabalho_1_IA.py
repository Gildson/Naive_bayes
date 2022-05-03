import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#método para lê o arquivo csv
base = pd.read_csv('spam.csv')

#para mostra os dados
print(base)

#transforma tudo em valores
X = base.iloc[:, [1]].values
Y = base.iloc[:, 0].values

labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, Y_train)

prediction = naive_bayes.predict(X_test)

#Acurácia dos dados
print(accuracy_score(Y_test, prediction))
