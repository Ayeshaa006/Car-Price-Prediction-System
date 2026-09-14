from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model  = pickle.load(open('model.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))

fuel_map         = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
seller_map       = {'Dealer': 0, 'Individual': 1}
transmission_map = {'Manual': 0, 'Automatic': 1}

@app.route('/')
def index():
    return render_template('index.html', prediction=None)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        year           = int(request.form['year'])
        present_price  = float(request.form['present_price'])
        kms_driven     = int(request.form['kms_driven'])
        fuel_type      = fuel_map[request.form['fuel_type']]
        seller_type    = seller_map[request.form['seller_type']]
        transmission   = transmission_map[request.form['transmission']]
        owner          = int(request.form['owner'])

        input_data = np.array([[year, present_price, kms_driven,
                                 fuel_type, seller_type, transmission, owner]])

        result = model.predict(input_data)[0]
        prediction = labels[result]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', prediction="Error: " + str(e))



if __name__ == '__main__':
    app.run(debug=True)