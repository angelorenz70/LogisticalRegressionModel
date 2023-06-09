from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    data = np.array(list(data.values())).astype(float).reshape(1, 8)

    with open('HeartDiseaseorAttack.pickle', 'rb') as f:
        model = pickle.load(f)

    # make a prediction using the input values and the loaded model
    y_prediction = model.predict(data)

    # Make the prediction
    prediction = y_prediction[0]

    # # Get the proba
    proba = model.predict_proba(data).tolist()[0]
    probability = float(proba[1]) * 100

    # Return the prediction as a JSON response
    return jsonify({'prediction': float(prediction), 'probability': "{:.2f}".format(probability)})

if __name__ == '__main__':
    app.run(debug=True)


