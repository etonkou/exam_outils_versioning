import pandas as pd
import seaborn as sb
import matplotlib as plt
import numpy as np
import plotly.express as px

# load dataset
data = pd.read_csv('data/california_housing_train.csv')

print('*'*500)

print(data)

# tris sur les colonnes utiles pour notre analyse
data = data[['total_rooms','total_bedrooms','population','households']]

print('*'*500)
print('tris sur les colonnes utiles pour notre analyse')
print(data)

# Verification des valeurs manquantes
print('*'*500)
print('Verification des valeurs manquantes')
d = data.isnull().sum() 
print(d)

# Verification des valeurs doubles
print('*'*500)
data.duplicated().sum()
print('Verification des valeurs doubles')
d = data.duplicated().sum() 
print(d)


# Analyse bivariee
print('*'*500)
print("Analyse bivariee")
sb.scatterplot(data['total_rooms'],data['total_bedrooms'])
sb.scatterplot(data['households'],data['total_bedrooms']) 
sb.scatterplot(data['households'],data['population']) 

print('*'*500)
# Matrice de correlation
cor_matrice = data.corr()
print(cor_matrice)

print('*'*500)
print("Traitement des valeurs extremes")
continuous_features = data.select_dtypes('float64').columns
for col in continuous_features:
 print(data[col].quantile(0.10))
 print(data[col].quantile(0.90))
