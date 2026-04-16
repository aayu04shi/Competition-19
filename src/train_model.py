import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):

    df = df.drop([
        "Report Number",
        "Date Reported",
        "Date of Occurrence",
        "Time of Occurrence",
        "Crime Description",
        "Date Case Closed"
    ], axis=1)

    y = df["Crime Domain"]
    X = df.drop("Crime Domain", axis=1)

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # ✅ SAVE MODEL (COMPRESSED)
    joblib.dump(model, "models/trained_model.pkl", compress=3)

    return model, X_test, y_test
