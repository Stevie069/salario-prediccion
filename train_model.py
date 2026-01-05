import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Cargar el dataset [cite: 16]
df = pd.read_csv('salarydataset.csv')

# Limpieza básica: Eliminar filas vacías o duplicados
df.dropna(inplace=True)

# 2. Definir Features (X) y Target (y) [cite: 17, 23]
features = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
target = 'Salary'

X = df[features]
y = df[target]

# 3. Preprocesamiento [cite: 25, 27]
# Variables numéricas: Age, Years of Experience
numeric_features = ['Age', 'Years of Experience']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Variables categóricas: Gender, Education Level, Job Title
categorical_features = ['Gender', 'Education Level', 'Job Title']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Crear el Pipeline con el Modelo (Regresión Lineal) [cite: 31]
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 5. Split de datos (80% train / 20% test) [cite: 28]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Entrenar el modelo [cite: 11]
print("Entrenando el modelo...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluar el modelo 
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Resultados de Evaluación:")
print(f"MAE (Mean Absolute Error): {mae:.2f}")
print(f"R2 Score: {r2:.4f}")

# 8. Guardar el modelo en la carpeta 'model/' 
joblib.dump(model_pipeline, 'model/salary_model.pkl')
print("Modelo guardado exitosamente en model/salary_model.pkl")
