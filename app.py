from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('model/salary_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Obtener datos del formulario [cite: 36]
        age = float(request.form['age'])
        gender = request.form['gender']
        education = request.form['education']
        job_title = request.form['job_title']
        experience = float(request.form['experience'])

        # Crear DataFrame con los inputs (nombres de columnas deben coincidir con el entrenamiento)
        input_data = pd.DataFrame([[age, gender, education, job_title, experience]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        # Realizar predicción
        prediction = model.predict(input_data)[0]

        # Formatear resultado
        prediction_text = f"${prediction:,.2f}"

        # Renderizar resultado eco de los inputs [cite: 43]
        return render_template('result.html', 
                               prediction=prediction_text,
                               age=age,
                               gender=gender,
                               education=education,
                               job_title=job_title,
                               experience=experience)

if __name__ == '__main__':
    app.run(debug=True)
