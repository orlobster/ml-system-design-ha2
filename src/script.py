from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

data = pd.read_csv('data.csv')
data_no_dup = data.drop_duplicates(ignore_index=True)
X, y = data_no_dup.drop(columns='NObeyesdad'), data_no_dup['NObeyesdad']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, stratify=y, random_state=42, test_size=0.25)

def custom_transform(X):
    X = X.copy()
    X['Gender'] = 1 * (X['Gender'] == 'Male')
    X['family_history_with_overweight'] = 1 * (X['family_history_with_overweight'] == 'yes')
    X['FAVC'] = 1 * (X['FAVC'] == 'yes')
    X['SMOKE'] = 1 * (X['SMOKE'] == 'yes')
    X['SCC'] = 1 * (X['SCC'] == 'yes')
    X['TUE_DIV_FAF'] = X['TUE'] / (X['FAF'] + 1e-4)
    X = X.drop(columns=['Weight'])
    return X

preprocessor = ColumnTransformer(
    transformers=[
        ("one_hot", OneHotEncoder(sparse_output=False), ['CAEC', 'CALC', 'MTRANS']),
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("custom_transform", FunctionTransformer(custom_transform)),
    ("preprocessor", preprocessor),
    ("scaler", MinMaxScaler()),
    ("classifier", RandomForestClassifier(random_state=42, criterion='entropy', max_depth=24, n_estimators=245))
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "../model.pkl")
