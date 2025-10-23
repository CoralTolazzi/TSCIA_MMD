import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

print("=" * 60)
print("INICIANDO CARGA DE DATOS")
print("=" * 60)

# 1. VERIFICAR Y CARGAR EL DATAFRAME
ruta_archivo = "C:/Users/DELL/Desktop/TSCIA_MMD/proyecto_2/Mini_Proyecto_Clientes_Promociones_from_Python.xlsx"

# Verificar si el archivo existe
if not os.path.exists(ruta_archivo):
    print(f"ERROR: El archivo no existe en la ruta: {ruta_archivo}")
    print("Buscando archivos Excel en la carpeta Downloads...")
    
    downloads_path = "C:/Users/47-01/Downloads/"
    if os.path.exists(downloads_path):
        archivos_excel = [f for f in os.listdir(downloads_path) if f.endswith(('.xlsx', '.xls'))]
        if archivos_excel:
            print("Archivos Excel encontrados:")
            for archivo in archivos_excel:
                print(f"   - {archivo}")
            # Usar el primer archivo Excel encontrado
            ruta_archivo = os.path.join(downloads_path, archivos_excel[0])
            print(f"Usando archivo: {ruta_archivo}")
        else:
            print("No se encontraron archivos Excel en Downloads")
            exit()
    else:
        print("No se puede acceder a la carpeta Downloads")
        exit()

# Cargar el archivo con manejo de errores
try:
    df = pd.read_excel(ruta_archivo)
    print(f"ARCHIVO CARGADO EXITOSAMENTE")
    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"Nombre del archivo: {os.path.basename(ruta_archivo)}")
    
except Exception as e:
    print(f"ERROR AL LEER EL ARCHIVO: {e}")
    print("Posibles soluciones:")
    print("   1. Verifica que el archivo no esté abierto en Excel")
    print("   2. Instala openpyxl: pip install openpyxl")
    print("   3. Verifica que el archivo no esté corrupto")
    exit()

# 2. MOSTRAR INFORMACIÓN DEL DATAFRAME
print("\n" + "=" * 60)
print("INFORMACIÓN DEL DATASET")
print("=" * 60)

print("PRIMERAS 5 FILAS:")
print(df.head())

print("\n NOMBRES DE COLUMNAS:")
print(df.columns.tolist())

print("\n INFORMACIÓN GENERAL:")
print(df.info())

print("\n VALORES NULOS POR COLUMNA:")
print(df.isnull().sum())

print("\n DISTRIBUCIÓN DE DATOS:")
print(df.describe())

# 3. VERIFICAR SI EXISTEN LAS COLUMNAS NECESARIAS
columnas_requeridas = ['Cliente_ID', 'Recompra']
columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]

if columnas_faltantes:
    print(f"\n COLUMNAS FALTANTES: {columnas_faltantes}")
    print("COLUMNAS DISPONIBLES:")
    for col in df.columns:
        print(f"   - {col}")
    exit()
else:
    print(f"\n TODAS LAS COLUMNAS REQUERIDAS ESTÁN PRESENTES")

# 4. PROCESAMIENTO DE DATOS (solo si df se cargó correctamente)
print("\n" + "=" * 60)
print("PROCESAMIENTO DE DATOS")
print("=" * 60)

# Verificar y mapear columnas categóricas
if 'Genero' in df.columns:
    print(" Valores únicos en Género:", df['Genero'].unique())
    df['Genero'] = df['Genero'].map({'F': 0, 'M': 1, 'Femenino': 0, 'Masculino': 1})
    
if 'Recibio_Promo' in df.columns or 'Recibió_Promo' in df.columns:
    col_name = 'Recibio_Promo' if 'Recibio_Promo' in df.columns else 'Recibió_Promo'
    print(f" Valores únicos en {col_name}:", df[col_name].unique())
    df[col_name] = df[col_name].map({'Si': 1, 'No': 0, 'Sí': 1})
    
if 'Recompra' in df.columns:
    print(" Valores únicos en Recompra:", df['Recompra'].unique())
    df['Recompra'] = df['Recompra'].map({'Si': 1, 'No': 0, 'Sí': 1})

# 5. VISUALIZACIÓN DE DATOS
print("\n" + "=" * 60)
print("VISUALIZACIÓN DE DATOS")
print("=" * 60)

# Visualización de relaciones clave
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x="Recompra", y="Monto_Promo", data=df)
plt.title("Recompra según el Monto Promocional")

plt.subplot(2, 2, 2)
sns.countplot(x='Recompra', data=df)
plt.title("Distribución de Recompra")

plt.subplot(2, 2, 3)
sns.countplot(x='Genero', hue='Recompra', data=df)
plt.title("Recompra por Género")

plt.subplot(2, 2, 4)
sns.countplot(x='Recibio_Promo', hue='Recompra', data=df)
plt.title("Recompra por Recepción de Promoción")

plt.tight_layout()
plt.show()

# 6. MODELADO PREDICTIVO
print("\n" + "=" * 60)
print("MODELADO PREDICTIVO")
print("=" * 60)

# Preparar variables
X = df.drop(['Cliente_ID', 'Recompra'], axis=1)
y = df['Recompra']

print(f" X shape: {X.shape}")
print(f" y shape: {y.shape}")
print(f" Variables predictoras: {X.columns.tolist()}")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Entrenar modelo
modelo = DecisionTreeClassifier(random_state=42, max_depth=3)
modelo.fit(X_train, y_train)

# Predecir y evaluar
y_pred = modelo.predict(X_test)

print("\n MATRIZ DE CONFUSIÓN:")
print(confusion_matrix(y_test, y_pred))

print("\n REPORTE DE CLASIFICACIÓN:")
print(classification_report(y_test, y_pred))

# 7. VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN
print("\n" + "=" * 60)
print("VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN")
print("=" * 60)

plt.figure(figsize=(20, 10))
plot_tree(modelo, 
          feature_names=X.columns.tolist(),
          class_names=['No Recompra', 'Recompra'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Árbol de Decisión - Predicción de Recompra", fontsize=16)
plt.show()

# 8. IMPORTANCIA DE LAS VARIABLES
print("\n" + "=" * 60)
print("IMPORTANCIA DE LAS VARIABLES")
print("=" * 60)

importancias = modelo.feature_importances_
features_importancia = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': importancias
}).sort_values('Importancia', ascending=False)

print(features_importancia)
36
# 9. ANÁLISIS ADICIONAL DEL MODELO
print("\n" + "=" * 60)
print("ANÁLISIS ADICIONAL DEL MODELO")
print("=" * 60)

print(f"Profundidad del árbol: {modelo.get_depth()}")
print(f"Número de hojas: {modelo.get_n_leaves()}")
print(f"Score de entrenamiento: {modelo.score(X_train, y_train):.4f}")
print(f"Score de prueba: {modelo.score(X_test, y_test):.4f}")

# 10. PREDICCIONES DE EJEMPLO
print("\n" + "=" * 60)
print("PREDICCIONES DE EJEMPLO")
print("=" * 60)

# Mostrar algunas predicciones vs valores reales
resultados = pd.DataFrame({
    'Real': y_test.values,
    'Predicho': y_pred
}).head(10)

print("Primeras 10 predicciones:")
print(resultados)

print("=" * 60)
print("PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 60)