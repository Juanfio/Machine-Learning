'''
    Comentarios globales sobre el código:
    Al modificar las variables "flor" y "filtra" se puede ver que:
    a) El modelo permite distinguir perfectamente entre: Setosa-Versicolor;
    Setosa-Virginica.
    b) Esto se corrobora no solamente por los valores de las métricas utilizadas
    (MAE, Accuracy, Precision, Recall, F1-Score) sino también por los plots de
    clusters.
    c) El modelo disminuye levemente su eficacia cuando tiene que clasificar entre
    Versicolor-Virginica; corroborado por la cercanía de estos en los plots de 
    clusters y las métricas.
'''

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# %% Defino las funciones a utilizar.


def filtro(iris_data, flor):
    # Para filtrar algún tipo de flor.
    if flor not in ['Setosa', 'Versicolor', 'Virginica']:
        print('ATENCIÓN: El dataset no fue filtrado.')
        print('Flor debe ser igual a: "Setosa", "Versicolor", "Virginica".\n')
        data_filtrada = iris_data
    else:
        filtro = iris_data['variety'] != flor
        data_filtrada = iris_data[filtro]
    return data_filtrada


def plot_data(iris_data, modo, filtra, flor):
    # Para plotear los datos.
    col_names = iris_data.columns
    if filtra == 'on':
        iris_data = filtro(iris_data, flor)
    elif filtra == 'off':
        pass
    else:
        print('ATENCIÓN - filtra debe ser igual a: "on", "off" \n')

    #########################
    if modo == 'boxplot':
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=[8, 8])
        l = 0
        for i in range(2):
            for j in range(2):
                sns.boxplot(
                    x='variety', y=col_names[l], data=iris_data, ax=axs[i, j])
                l += 1

        fig.suptitle(
            'Distribución de parámetros \n característicos para cada flor')

    #########################
    elif modo == 'clusters':
        fig = plt.figure(figsize=(13, 13))

        ###############
        plt.subplot(331)
        sns.scatterplot(x=col_names[0], y=col_names[1],
                        hue='variety', data=iris_data)

        ###############
        plt.subplot(332)
        sns.scatterplot(x=col_names[0], y=col_names[2],
                        hue='variety', data=iris_data)

        ###############
        plt.subplot(333)
        sns.scatterplot(x=col_names[0], y=col_names[3],
                        hue='variety', data=iris_data)

        ###############
        plt.subplot(334)
        sns.scatterplot(x=col_names[1], y=col_names[2],
                        hue='variety', data=iris_data)

        ###############
        plt.subplot(335)
        sns.scatterplot(x=col_names[1], y=col_names[3],
                        hue='variety', data=iris_data)

        ###############
        plt.subplot(337)
        sns.scatterplot(x=col_names[2], y=col_names[3],
                        hue='variety', data=iris_data)

        fig.suptitle(
            'Agrupamiento de las flores en función \n de sus parámetros característicos')
        plt.show()
    else:
        print('ATENCIÓN - modo debe ser igual a: "boxplot", "clusters". \n')
    return


def modelo(iris_data, filtra, flor):
    # Defino el modelo.
    if filtra == 'on':
        iris_data = filtro(iris_data, flor)
    elif filtra == 'off':
        pass
    else:
        print('ATENCIÓN - filtra debe ser igual a: "on", "off" \n')

    # Transformo Variables Categóricas en Variables Numéricas.
    # Referencias Label: 0 = Setosa, 1 = Versicolor, 2 = Virginica.
    label_iris_data = iris_data.copy()
    label_encoder = LabelEncoder()
    label_iris_data['variety'] = label_encoder.fit_transform(
        iris_data['variety'])

    # Parto los datos en: Train, Test.
    # Separo en X (variables) e y (predicción).
    y = label_iris_data.variety
    X = label_iris_data.drop(['variety'], axis=1)

    # Train Data = 60%, Test Data = 40%.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, test_size=0.4, random_state=0)

    # Fiteo el modelo con los parámetros óptimos obtenidos.
    modelo_knn = KNeighborsClassifier()
    modelo_knn.fit(X_train, y_train)
    y_pred = modelo_knn.predict(X_test)

    # Performance del modelo
    MAE = mean_absolute_error(y_test, y_pred)
    Accuracy = accuracy_score(y_test, y_pred)
    Precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
    Recall = recall_score(y_test, y_pred, average='weighted')
    F1_Score = round(f1_score(y_test, y_pred, average='weighted'), 2)

    return MAE, Accuracy, Precision, Recall, F1_Score


# %% Importo los datos.
iris_data = pd.read_csv('iris.csv')

###########################
# Descomentar para ver los datos.

# pd.set_option('display.max_rows', None)
# print(iris_data)
# print(iris_data.describe())

###########################
# Modificar la variable flor (Setosa, Versicolor, Virginica).
# Modificar la variable filtra (on/off).
# Modificar la variable modo (boxplot/clusters)
flor = 'Setosa'
filtra = 'off'
modo = 'clusters'

###########################
plot_data(iris_data, modo, filtra, flor)
MAE, accuracy, precision, recall, F1_score = modelo(iris_data, filtra, flor)

###########################
# Veo la performance del modelo.
print('El error medio absoluto: ', MAE)
print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1-Score: ', F1_score)
# %%
