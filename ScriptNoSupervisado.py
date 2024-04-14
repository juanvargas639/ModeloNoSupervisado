import pandas as pd
from sklearn.ensemble import IsolationForest

# Cargar el dataset desde el archivo CSV
data = pd.read_csv('dataSet2.csv')

# Detectar anomalías utilizando Isolation Forest
model = IsolationForest(contamination=0.1)
model.fit(data[['usuarios_diarios', 'promedio_tiempos_espera']])
anomalies = model.predict(data[['usuarios_diarios', 'promedio_tiempos_espera']])

# Agregar las predicciones de anomalías al DataFrame
data['anomalia'] = anomalies

# Filtrar las estaciones anómalas
anomalous_stations = data[data['anomalia'] == -1]

# Imprimir información detallada sobre las estaciones anómalas
print("Estaciones anómalas:")
print(anomalous_stations)
