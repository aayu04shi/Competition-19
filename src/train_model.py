import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    # Drop high-cardinality / useless columns
    df = df.drop([
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Crime Description",
        "Date Case Closed"
    ], axis=1)

    # Target
    y = df["Crime Domain"]
    X = df.drop("Crime Domain", axis=1)

    # Encode only useful categorical columns
    X = pd.get_dummies(X, drop_first=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Save model
    pickle.dump(model, open("models/trained_model.pkl", "wb"))

    return model, X_test, y_test
