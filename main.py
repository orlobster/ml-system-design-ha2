from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

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

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_path = request.json.get('input_path')
    output_path = request.json.get('output_path')
    
    if not os.path.exists(input_path):
        return jsonify({'error': f'input file {input_path} not found'}), 400
    
    try:
        input_data = pd.read_csv(input_path)
    except Exception as e:
        return jsonify({'error': f'failed to convert input file {input_path} to csv: {str(e)}'}), 400
    
    try:
        y_pred = model.predict(input_data)
    except Exception as e:
        return jsonify({'error': f'failed to inference model: {str(e)}'}), 500
    
    input_data['NObeyesdad'] = y_pred
    try:
        input_data.to_csv(output_path, index=False)
    except Exception as e:
        return jsonify({'error': f'failed to save data to csv: {str(e)}'}), 500
    
    return jsonify({'success': f'saved data to {output_path}'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5132)
