import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)
model = pickle.load(open('final_prediction.pickle', 'rb'))
df_new = pd.read_csv('column.csv').drop(['Unnamed: 0'],axis=1)

'''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
    
'''


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json()
    #print(type(data[0]['a']))
    #print(data['a'])
    df = pd.DataFrame.from_dict (data, orient='columns')
    #print(df)
    dff = pd.get_dummies (df)
    #print(dff)
    X=dff.reindex(columns=df_new.columns,fill_value=0)
    #print(X.shape)
    y_pred = model.predict(X)
    y_pred=y_pred.tolist()
    #print(y_pred)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    return jsonify({'prediction':y_pred})

if __name__ == "__main__":
    app.run(debug=True)