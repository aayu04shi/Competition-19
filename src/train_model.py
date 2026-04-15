import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    df = pd.get_dummies(df)

    X = df.drop("Crime Domain", axis=1)
    y = df["Crime Domain"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    pickle.dump(model, open("models/trained_model.pkl", "wb"))

    return model, X_test, y_test
