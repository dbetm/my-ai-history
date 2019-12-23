# Training and running a linear model using Scikit-Learn
# Esto es aprendizaje usando el modelo, no por instancia
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

"""
This function just merges the OECD's life satisfaction data and the IMF's
GDP per capita data. It's a bit too long and boring and it's not specific
to Machine Learning.
"""
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
        left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

## LO IMPORTANTE ##
# Definimos la ruta que es donde encontraremos nuestros datasets
datapath = os.path.join("../datasets", "lifesat", "")
# Leemos con Pandas el primer dataset
oecd_bli_dataset = pd.read_csv(datapath + "oecd_bli_2015.csv", thousands=',')
# Ahora se lee el otro dataset
gdb_per_capita_dataset = pd.read_csv(datapath + "gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')

# Preparar los datos
country_stats = prepare_country_stats(oecd_bli_dataset, gdb_per_capita_dataset)
# Obtener la variable independiente, con pandas seleccionamos la columna deseada
X = np.c_[country_stats["GDP per capita"]]
# Obtener la variable dependiente
y = np.c_[country_stats["Life satisfaction"]]

# Visualizar los datos
country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show()

# Seleccionar un modelo lineal, en este caso regresión lineal
# model = sklearn.linear_model.LinearRegression()
# Si queremos que la predicción sea por modelo
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3) # => [[5.76666667]]


# Entrenamos el modelo
model.fit(X, y)

# Hacemos la predicción para el país Chipre - Cyprus
X_new = [[22587]] # Cyprus' GDP per capita
print(model.predict(X_new)) # output [[ 5.96242338]]
