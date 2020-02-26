import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from titanic import *

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [str(x) for x in request.form.values()]
    
    '''
    pclass = StandardScaler(int(int_features[0]))
    age= StandardScaler(int(int_features[3]))
    sibsp=StandardScaler(int(int_features[4]))
    parch=StandardScaler(int(int_features[5]))
    fare = StandardScaler(int_features[7])
    
    int_features[1] = OneHotEncoder(int_features[1])
    int_features[2] = OneHotEncoder(int_features[2])
    int_features[6] = OneHotEncoder(int_features[6])
    int_features[8] = OneHotEncoder(int_features[8])
    '''

    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    if output == 1:
        toprint = "Survived"
    else:
        toprint = "Died"

    return render_template('index.html', prediction_text='This passenger should have $ {}'.format(toprint))

@app.route("/predict_api", methods = ['POST'])
def predict_api():
    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    if output == 1:
        toprint = "Survived"
    else:
        toprint = "Died"

    return jsonify(toprint)

if __name__ == "__main__":
    app.run(debug=True)


