# Importa las librerías necesarias y carga tu modelo predictivo
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carga tu modelo predictivo entrenado previamente
model_path = 'modelo_vector.pkl'
print(model_path)  # Verificar que la ruta sea correcta
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Realiza la predicción utilizando el modelo cargado
        prediction = model.predict([data['feature']])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)