import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso


def load_data(file_path):
    df = pd.read_csv("bikes_rent.csv")
    return df


def simple_linear_regression(df, predictor, target):
    X = df[[predictor]]
    y = df[target]

    model = LinearRegression()
    model.fit(X, y)

    plt.scatter(X, y, color='blue')
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel(predictor)
    plt.ylabel(target)
    plt.title('Линейная регрессия')
    plt.show()

    return model

def predict_cnt(model, predictor_value):
    return model.predict([[predictor_value]])

def reduce_dimensionality_and_plot(df, predictors, target):
    X = df[predictors]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.xlabel('1 компонент')
    plt.ylabel('2 компонент')
    plt.title('Предсказание')
    plt.colorbar(label=target)
    plt.show()

def lasso_regression(df, predictors, target):
    X = df[predictors]
    y = df[target]

    model = Lasso()
    model.fit(X, y)

    feature_importance = pd.Series(model.coef_, index=predictors)
    most_influential_feature = feature_importance.idxmax()

    return feature_importance, most_influential_feature


file_path = "bikes_rent.csv"
df = load_data(file_path)

predictor = 'weathersit'
target = 'cnt'
model = simple_linear_regression(df, predictor, target)


predictor_value = 2
predicted_cnt = predict_cnt(model, predictor_value)
print("Предпологаемое значение :", predicted_cnt[0])

predictors = ['temp', 'hum', 'windspeed(ms)']
reduce_dimensionality_and_plot(df, predictors, target)


predictors = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed(ms)']
feature_importance, most_influential_feature = lasso_regression(df, predictors, target)
print("Важность функции:")
print(feature_importance)
print("Самая влиятельная особенность:", most_influential_feature)
