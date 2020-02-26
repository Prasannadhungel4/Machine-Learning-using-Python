from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import  OneHotEncoder
from sklearn.linear_model import Perceptron
import tensorflow as tf
import numpy as np

dforigin = pd.read_csv("train.csv")
df = dforigin.drop("Cabin", axis=1)
df = df.drop("Survived", axis = 1)

#background: #092756; <h1>Titanic Passenger Survival</h1>




class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names]

from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
    ("standarad_scaler", StandardScaler())
])

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent = pd.Series([X[c].value_counts().index[0] for c in X], index = X.columns)
        return self
    
    def transform(self,X, y=None):
        return X.fillna(self.most_frequent)


cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),  
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

from sklearn.pipeline import FeatureUnion

preprocess_full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])



X_train = preprocess_full_pipeline.fit_transform(df)
y_train = dforigin["Survived"]



from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train,y_train)


'''
from sklearn.svm import SVC
svm = SVC(kernel='linear', C= 1.0, random_state=0)
svm.fit(X_train, y_train)
'''
'''
from sklearn.linear_model import LogisticRegression
percep = LogisticRegression(C=1000, random_state=42, penalty='l2')
percep.fit(X_train, y_train)
'''

from sklearn.model_selection import cross_val_score
print(cross_val_score(forest, X_train, y_train, cv=3, scoring='accuracy'))
'''
import pickle
pickle.dump(forest, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

'''

x_test = pd.read_csv("test.csv")
x_test = preprocess_full_pipeline.fit_transform(x_test)
y_pred = forest.predict(x_test)

print(y_pred)

pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv("gender_submission.csv")
datasets = pd.concat([sub_df['PassengerId'], pred], axis = 1)
datasets.columns=['PassengerId', 'Survived']
datasets.to_csv("submission.csv", index=False)







'''
df = num_pipeline.fit_transform(df)
df = pd.DataFrame({'Age': df[:, 0], 'Fare': df[:, 1]})
dforigin["Age"] = df["Age"]
print(df.info())
print(dforigin.info())
'''

'''
df= df["Age"]
imp = SimpleImputer(missing_values = 'NaN', strategy = 'median')
imp = imp.fit(df)
df = imp.transform(df.values)

'''