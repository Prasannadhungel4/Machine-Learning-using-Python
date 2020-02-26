'''
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'PassengerId':892, 'Pclass':3, 'Name':'Kelly, Mr. James', 'Sex':'male', 'Age':47, 'SibSp':0, 'Parch':0, 'Ticket':'A/4 48871', 'Fare':7.8292, 'Embarked':'Q' })

print(r.json())
'''
import numpy as np
from sklearn.preprocessing import OneHotEncoder

print(np.array(OneHotEncoder("male")))
