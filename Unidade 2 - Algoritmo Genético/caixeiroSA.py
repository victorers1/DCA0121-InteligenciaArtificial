#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 16:14:12 2018

@author: victor

Para ver funcionando, executar todo o código.
"""
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

def testaCaminho(grafo, caminho):
    '''Recebe um grafo e um caminho. Testa se um caminho é válido.'''
    for i in range(1, len(caminho)-1):
        if not grafo.has_edge(caminho[i],caminho[i+1]):  # Se não é adjacêcia entre dos nós
            return False
    return True  # Se saiu do FOR ,significa que os nós vizinhos têm adjacêcia

def geraCaminho(grafo, nos):
    '''Recebe um grafo e a lista usada para criar os nós. Gera um caminho adjacente, que é uma lista.'''
    caminhoVal = False  # CaminhoValido
    while(not caminhoVal):
        caminho = random.sample(nos[1:], grafo.number_of_nodes()-1)
        caminho.append(1)
        caminho.insert(0, 1)
        caminhoVal = testaCaminho(grafo, caminho)
    return caminho

def caminhoAdjacente(grafo, caminho):
    '''Gera um caminho 'adjacente' ao dado. Retorna o mesmo caminho com duas cidades trocadas de lugar.'''
    pos1 = 0
    pos2 = 0
    adj = caminho.copy()
    caminhoVal = False
    while(not caminhoVal or adj==caminho):  
        while(pos1==pos2):
            pos1 = random.randint(1, len(caminho)-2)
            pos2 = random.randint(1, len(caminho)-2)
        adj[pos1],adj[pos2] = adj[pos2],adj[pos1]  # Realiza troca
        caminhoVal = testaCaminho(grafo, adj)
    return adj
    
def custo(grafo, caminho):
    '''Calcula custo de um caminho'''
    soma = 0
    for i in range(len(caminho)-1):
        soma += grafo[caminho[i]][caminho[i+1]]['weight']
    return soma


def SA(grafo, cidades, T):
    '''Implementa Simulated Anneling'''
    xAtual = geraCaminho(grafo, cidades)  # Gera primeiro caminho aleatório
    yAtual = custo(grafo, xAtual)  # E seu custo
    hist_X = [xAtual]  # Histórico de
    hist_Y = [yAtual]  # soluções
    for i in range(50000):
        xProx = caminhoAdjacente(grafo, xAtual)  # Gera os novos dados para
        yProx = custo(grafo, xProx)              # comparar com o atual
        print('i: '+str(i)+', yA: '+str(yAtual)+', yP: '+str(yProx)+', T: '+str(T))
        if np.random.rand() > 1 - np.e**-((yAtual-yProx)/T):  # Caso verdade, NÃO aceito novo valor
            xProx = xAtual.copy()  # Valor de xProx é descartado
        
        xAtual = xProx.copy()         # Atualiza valores de x e y
        yAtual = custo(grafo, xAtual) #
        
        if  i%2000:
            T = 0.99*T
        if T<0.5:  # Para o loop quando T fica bem pequeno
            break;
        hist_X.append(xAtual)  # Adiciona no fim da lista o valor de xAtual (que também é uma lista)
        hist_Y.append(yAtual)
        
    custoFinal = custo(grafo, xAtual)  # Calcula custo da última solução encontrada
    return xAtual, custoFinal, hist_X, hist_Y


#nx.draw_circular(Mapa,with_labels=True,font_weight='bold')
#nx.draw_networkx_edge_labels(Mapa, pos=nx.circular_layout(Mapa))
#Mapa.number_of_edges()
#Consultar documentação completa do draw(): nx.draw_networkx()

cidades = []
qtd_cidade = 7
for i in range(1, qtd_cidade+1):
    cidades.append(i)

Mapa = nx.Graph()
Mapa.add_nodes_from(cidades)
for i in range(1, qtd_cidade+1):
    for j in range(1, qtd_cidade+1):
        if i > j:  # Pra não percorrer a mesma aresta duas vezes
            Mapa.add_edge(i,j, weight=random.randint(0, 50))


print('Gerando grafo aleatório')
plt.figure(1)
nx.draw_circular(Mapa,with_labels=True,font_weight='bold')
nx.draw_networkx_edge_labels(Mapa, pos=nx.circular_layout(Mapa))
plt.show()

print('Executando SA várias vezes')
resultados = []
for i in range(50):
    x,c,hx,hy = SA(Mapa, cidades,19999)
    resultados.append(c)

plt.figure(2)
plt.plot(resultados)
plt.xlabel('Iteração')
plt.ylabel('Custo Calculado')
plt.show()

'''
cidades = []
qtd_cidade = 6
for i in range(1, qtd_cidade):
    cidades.append(i)

 
Mapa = nx.Graph()  # Cria grafo
Mapa.add_nodes_from(cidades)  # Adiciona nós 

# Adiciona arestas 
e = [(1,2,99), (2,3,99), (3,4,1),(4,5,99),(5,1, 1),(1,4,99), (2,4,1),
     (2,5,1),(1,3,1), (3,5,99)]
Mapa.add_weighted_edges_from(e)


Mapa.add_edges_from([(1,2, {'peso': random.randint(0, 100)}), (2,3, {'peso': random.randint(0, 100)}),
                     (3,4, {'peso': random.randint(0, 100)}),(4,5, {'peso': random.randint(0, 100)}),
                     (5,1, {'peso': random.randint(0, 100)}),(1,4, {'peso': random.randint(0, 100)}),
                     (2,4, {'peso': random.randint(0, 100)}),(2,5, {'peso': random.randint(0, 100)}),
                     (1,3, {'peso':}))])


#Mapa = nx.complete_graph(10)
Mapa = nx.Graph()
Mapa.add_nodes_from(cidades)
for i in range(1, qtd_cidade):
    for j in range(1, qtd_cidade):
        if i > j:  # Pra não percorrer a mesma aresta duas vezes
            Mapa.add_edge(i,j, peso=random.randint(0, 100))

'''
