import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app2 = Flask(__name__)
model2 = pickle.load(open('model2.pkl', 'rb'))

@app2.route('/')
def home():
    return render_template('index2.html')

@app2.route('/predict',methods=['POST'])
def predict():
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model2.predict(final_features)
    
    output = round(prediction[0], 2)

    return render_template('index2.html', prediction_text='predicted sheep census to be million {}'.format(output))

if __name__ == "__main__":
    app2.run(debug=True)    
    