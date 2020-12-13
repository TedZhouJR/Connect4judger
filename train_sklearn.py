from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

lookup_dict = {"x":0, "o":2, "b":1, "-":0, "+":1, "=":2}

def load_data():
    with open("data/database.txt", "r") as f:
        inputs = f.readlines()
    X = [ip[:-2] for ip in inputs]
    Y = [ip[-2] for ip in inputs]
    X = np.array([[lookup_dict[xx] for xx in x] for x in X])
    Y = np.array([lookup_dict[y] for y in Y])
    X1 = (X == 0).astype(int)
    X2 = (X == 2).astype(int)
    X = np.concatenate((X1, X2), axis=1)
    return X, Y

def get_model(depth, model="dt"):
    if model == "dt":
        return DecisionTreeClassifier(max_depth=depth,
                                 min_samples_split=3,
                                 min_samples_leaf=1,
                                 random_state=0)
    elif model == "rf":
        return RandomForestClassifier(max_depth=depth,
                                 min_samples_split=3,
                                 min_samples_leaf=1,
                                 random_state=0)

if __name__ == "__main__":
    for i in [5,10,15,20,25,40]:
        X, Y = load_data()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=0.8)
        clf = get_model(depth=i)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(acc)
