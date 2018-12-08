#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 20:54:24 2018
Descrição: Tentei ficar milionário e não deu certo.
@author: victor
"""

import csv
import pandas as pd
import numpy as np
import keras.layers as kl
import keras.models as km
import keras.callbacks as kc
import matplotlib.pyplot as plt


df = pd.read_csv('sorteios.csv')
#print(df)

print('dtypes:')
print(df.dtypes)
print('index:')
print(df.index)
print('columns:')
print(df.columns)

sorteios = []  # Não há itens repetidos
for i in range(len(df)):
    linha=[]
    for dezena in range(1,7):
        nome = str(dezena) + ' Dezena'
        linha.append(df[nome][i])
    if linha not in sorteios:
        sorteios.append(linha)

serie = np.array(sorteios)
serie = serie.flatten()

ultimos = serie[-6:]
serie = serie[:-6]  # 42

qtd_pts = 60  # quantidade de pontos no passado que serão fornecidos para a rede
#qtd_amostra = 500  # quantas amostras serão usadas para treinar
x_treino,y_treino = [],[]

ind_y = [len(serie)-6,len(serie)]  # Posições de corte na lista 'serie'. Segundo índice está incrementado em 1.
ind_x = [ind_y[0]-qtd_pts, ind_y[0]]  # Posições de corte da lista 'serie'. Segundo índice está incrementado em 1 porque o corte é exlcusivo.
while True:
    x_treino.append(serie[ind_x[0]:ind_x[1]])
    y_treino.append(serie[ind_y[0]:ind_y[1]])
    
    if ind_x[0]==0:
        break
    
    ind_y[0]-=1; ind_y[1]-=1
    ind_x[0]-=1; ind_x[1]-=1
    
x_treino = np.array(x_treino)
y_treino = np.array(y_treino)

model = km.Sequential()
model.add(kl.Dense(256, input_dim=qtd_pts, activation='relu'))
model.add(kl.Dense(256, activation='relu'))
model.add(kl.Dense(256, activation='relu'))
model.add(kl.Dense(256, activation='relu'))
model.add(kl.Dense(256, activation='relu'))
model.add(kl.Dense(6, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['acc'])

monitor = kc.EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')
H = model.fit(x_treino, y_treino, epochs=1500, batch_size=1200, callbacks=[monitor])

plt.plot(H.history['loss'])

x_teste = np.array([serie[len(serie)-60:]])
result = model.predict(x_teste)
