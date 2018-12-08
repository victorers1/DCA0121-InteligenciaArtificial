#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:55:02 2018
@author: victor
"""

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

taxa_crossover = 0.85  # Taxa de cruzamento
taxa_mutacao = 0.0015  # Taxa de mutacao

'''# Modelando  o grafo da questao
Mapa = nx.Graph()
Mapa.add_nodes_from([1,2,3,4,5])
Mapa.add_weighted_edges_from([(1,2,2),(2,3,4),(3,4,7),(4,5,3),(5,1,6),(1,3,9),
                              (2,4,3),(2,5,8),(1,4,3),(3,5,3)])
'''
cidades = []
qtd_cidade = 10
for i in range(1, qtd_cidade+1):
    cidades.append(i)

Mapa = nx.Graph()
Mapa.add_nodes_from(cidades)
for i in range(1, qtd_cidade+1):
    for j in range(1, qtd_cidade+1):
        if i > j:  # Pra não percorrer a mesma aresta duas vezes
            Mapa.add_edge(i,j, weight=random.randint(0, 50))

nx.draw_circular(Mapa,with_labels=True,font_weight='bold')
nx.draw_networkx_edge_labels(Mapa, pos=nx.circular_layout(Mapa))
plt.show()


def pop_inic(tam_pop, tam_cromo):
    '''Gera populacao inical. Recebe o tamanho da populacao e do cromossomo.'''
    cromossomos = []  # Lista de caminhos
    cidades = []  # Lista com nomes das cidades
    for i in range(1,tam_cromo+1):  # Nomeia cidades e 1 a tam_cromo+1
        cidades.append(i)
        
    for i in range(tam_pop):  # Cria lista de cromossomos
        caminho = random.sample(cidades, tam_cromo)
        while(caminho in cromossomos):  # while garante que nao ha dois caminhos iguais na populacao
            caminho = random.sample(cidades, tam_cromo)
        cromossomos.append(caminho)  # Adiciona caminho ao fim da lista        
    return cromossomos


def aptidao(grafo, caminho):
    '''Retorna o custo de percorrer no grafo o caminho dado. Para nosso 
    problema quanto menor o custo, melhor. Por isso definimos que 
    Aptidao = 1/Custo'''
    soma = 0
    for i in range(len(caminho)-1):
        soma += grafo[caminho[i]][caminho[i+1]]['weight']
    return 1/soma



def cria_roleta(grafo, caminhos):
    '''Recebe o grafo e uma lista de caminhos. Primeiro, cria uma lista na qual cada 
    posicao i contem a probabilidade do elemento i na lista de caminhos ser 
    selecionado. Depois retorna a prob. cumulativa.'''
    custos = []  # Lista com custos de todos caminhos dados
    for c in caminhos:
        custos.append(aptidao(grafo, c))
    soma_custos = np.sum(custos)  # Somatorio dos custos
    prob = [x/soma_custos for x in custos]  # prob eh UMA LISTA DE x/soma_custos PARA TODO x EM custos
    return list(np.cumsum(prob))  # Retorna uma lista comum


def selec_pais(prob_cumul, cromossomos):
    '''Recebe a probabilidade cumulativa e uma lista de cromossomos. Retorna
    uma lista com os cromossomos selecionados para serem pais.'''
    pais = []  # lista com os pais da geracao atual
    for cromo in cromossomos:
        val = random.uniform(0,1)  # Gera um aleatorio no intervalo [0,1]
        for i in range(len(prob_cumul)):  # Percorre a lista de prob. cumulativa procurando o primeiro maior que val
            if prob_cumul[i] > val:  # Se encontrar...
                pais.append(cromossomos[i])  # Adiciona na lista de pais
                break
    return pais



def cria_filhos(pai, mae):
    '''Aplica o algoritmo de cruzamente entre dois caminhos. Recebe dois
    indivoduos de mesmo tamanho e retorna seus 2 filhos. Esta funcao eh necessaria para que 
    crossover() funcione.'''
    pos_max = len(pai)
    p1,p2 = 0,0
    while(p1>=p2):
        p1,p2 = random.randint(0, pos_max),random.randint(0, pos_max)
    
    m1,m2 = 0,0
    while(m1>=m2):
        m1,m2 = random.randint(0, pos_max),random.randint(0, pos_max)
    
    filho1,filho2 = [],[]  # Crio lista preenchidas com -1
    for i in pai:
        filho1.append(-1)
        filho2.append(-1)
    
    # Primeiro passo: parte dos pais entre os cortes sao transmitidos integralmente aos filhos
    for i in range(len(pai)):  
        if i >= p1 and i < p2:  # Preenchendo filho 1 com parte do pai
            filho1[i] = pai[i]
        if i >= m1 and i < m2:  # Preenchendo filho 2 com parte da mae
            filho2[i] = mae[i]

    # Segundo passo: preenchemos filho1 com o resto da mae
    for i in range(len(mae)-1, -1, -1):  # Percorre 'mae' em sentido contrario a leitura
        if mae[i] not in filho1:
            #print(str(mae[i])+' not in filho1')
            #print('filho1['+str(ind)+'] = '+str(mae[i]))
            for j in range(len(filho1)):
                if j not in range(p1,p2) and filho1[j]==-1:
                    filho1[j] = mae[i]
                    break
        
    # Terceiro passo: preenchemos filho2 com o resto do pai
    for i in range(len(pai)-1, -1, -1):
        if pai[i] not in filho2:
            for j in range(len(filho2)):
                if j not in range(m1,m2) and filho2[j]==-1:
                    filho2[j] = pai[i]
                    break
    return filho1, filho2    


def crossover(pais, taxa_crossover):
    ''' Recebe uma lista com os pais e a taxa de cruzamento. Retorna os pais
    da proxima geracao. A realizacao do cruzamento ou nao, eh decidido pela taxa
    de crossover'''
    filhos = []
    sobreviventes = []  # pais que continuarao na proxima geracao
    for i in range(0, len(pais)-1, 2):
        if random.uniform(0,1) < taxa_crossover:  # Aceito fazer cruzamento
            f1,f2 = cria_filhos(pais[i],pais[i+1])
            filhos.append(f1); filhos.append(f2)  # Guarda resultados em 'filhos'
        else:  # Recuso cruzamento
            sobreviventes.append(pais[i]); sobreviventes.append(pais[i+1])  # Guardo os pais em 'sobreviventes'
    return filhos, sobreviventes


def trocaCidades(caminho):
    '''Retorna o mesmo caminho com duas cidades trocadas de lugar. Essa funcao
    eh necessaria para que mutacao() dê certo.'''
    pos1 = 0
    pos2 = 0
    adj = caminho.copy()
    while(pos1==pos2):
        pos1 = random.randint(1,len(caminho)-1)
        pos2 = random.randint(1,len(caminho)-1)
    adj[pos1],adj[pos2] = adj[pos2],adj[pos1]  # Realiza troca
    return adj


def mutacao(filhos, taxa_mutacao):
    ''' Aplica uma alteracao aleatoria em alguns dos filhos, que sao 
    selecionados aleatoriamente. A quantidade de filhos a ser mutados eh 
    decidida pela taxa de mutacao.'''
    filhos_novo = filhos.copy()
    for f in filhos_novo:
        if random.uniform(0,1) < taxa_mutacao:
            f = trocaCidades(f)
    return filhos_novo


def geracao(grafo, cromossomos):
    '''Retorna informacoes sobre a geracao atual. Informacoes como: roleta de 
    selecao, indivoduo de melhor aptidao, aptidao media.'''
    roleta = cria_roleta(grafo, cromossomos)
    aptidoes = []
    for c in cromossomos:
        aptidoes.append(aptidao(grafo, c))
    
    pos_melhor = np.argmax(aptidoes)  #Melhor = maior = menor custo
    melhor_apt = aptidoes[pos_melhor]
    melhor_cromo = cromossomos[pos_melhor]
    media = np.mean(aptidoes)
    return roleta, melhor_apt, melhor_cromo, media


def GA(tam_pop,tam_cromo,iter_max,taxa_crossover,taxa_mutacao):
    i=0  # Conta quantas geracoes foram geradas
    cromossomos = pop_inic(tam_pop, tam_cromo)
    roleta, tmp_melhor_apt, tmp_melhor_caminho, tmp_media = geracao(Mapa, cromossomos)
    melhor_caminho = [tmp_melhor_caminho]  # Guarda melhor caminho da geracao  | historico
    melhor_apt = [tmp_melhor_apt]          # Guarda melhor aptidao da geracao  | das
    media = [tmp_media]                    # Guarda a media das aptidoes       | geracoes
    while True:
        prob_cumul = cria_roleta(Mapa, cromossomos)  # Crio lista que servira de roleta
        pais = selec_pais(prob_cumul, cromossomos)  # Dentre os indivoduos atuais, seleciono quais serao pais
        filhos, sobreviventes = crossover(pais, taxa_crossover)  # Criacao de filhos
        filhos = mutacao(filhos,taxa_mutacao)  # Somente os recem chegados sao mutados
        cromossomos = list(map(list,filhos))  # Nova populacao
        for s in sobreviventes:  # como 'sobreviventes' eh uma lista de listas(caminhos), temos de fazer um FOR para incluo-lo em cormossomos
            cromossomos.append(s)
        i+=1
        if i > iter_max: # Verifico se ja basta de iteracoes
            break
        roleta, tmp_melhor_apt, tmp_melhor_caminho, tmp_media = geracao(Mapa, cromossomos)  # Faco inferencias sobre a nova geracao 
        melhor_caminho.append(tmp_melhor_caminho) # Atualizacao
        melhor_apt.append(tmp_melhor_apt)         # dos
        media.append(tmp_media)                   # historicos
    ind = np.argmax(melhor_apt)  # Recupera em qual geracao ocorreu o melhor indivoduo de todos
    return ind, melhor_caminho[ind], melhor_apt[ind], media


custos = []
for i in range(25):  # Testando 25 vezes para ver a precisao do resultado
    ind,caminho, apt, media = GA(20, 5, 20, taxa_crossover, taxa_mutacao)
    custos.append(1/apt)  # O custo de percorre o caminho eh o inverso da aptidao

plt.plot(custos)
plt.title('Analise de Desempenho')
plt.xlabel('Iteracao')
plt.ylabel('Custo calculado')
plt.show()