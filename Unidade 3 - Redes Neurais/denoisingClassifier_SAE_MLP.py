"""
Created on Fri Dec  7 03:44:19 2018

@author: Guillherme
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras import callbacks
import itertools
from sklearn import model_selection as sk
from sklearn.metrics import confusion_matrix
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo estimado')
    plt.tight_layout()
    
    
def read_img():
    '''
    Essa função lê as imagens das letras 'a, b, c, d, e, f, i, o, u' contidas 
    na pasta 'letras'. 
    301 imagens de cada letra são utilizadas.
    Esta função retorna os dados (imagens) que serão utilizados para 
    treinamento e de teste.
    '''
    x = []
    y = []
    x_treino = []
    y_treino = []
    x_teste = []
    y_teste = []
    
    ind = 0
    for letra in ['a','b','c','d','e','f','i','o','u']:
        for i in range(301):
            nome = 'letras/'+letra+str(i)+'.jpg'
            img = cv.imread(nome, cv.IMREAD_GRAYSCALE)
            x.append(img)
            y.append(ind)
        ind+=1
    x = np.array(x); y = np.array(y)
    x_treino, x_teste, y_treino, y_teste = sk.train_test_split(x,y, 
                                            test_size=0.20, random_state=42)
    return x_treino, x_teste, y_treino, y_teste

def pre_processamento(x_treino, x_teste, ruidos='g'):
    '''
    Esta função primeiro escala os dados que originalmente estão entre 
    [0, 255] para o intervalo [0, 1].
    
    Depois são aplicados ruídos indicados no param 'ruidos'. 'ruidos' é uma 
    string que pode conter as letras 'g', 'n', 'l', 'i' em qualquer ordem e em qualquer quatidade
    
    Valores que ficarem fora do intervalo [0, 1] são 'truncados' para 
    obedecerem o intervalo.
    
    Em seguida, a imagem que é de dimensão nxn se transforma em um vetor de 
    tamanho n² (n*n).
    
    Esta função retorna os dados normalizados assim como suas versões com 
    ruído.
    '''
    max_value = float(x_treino.max())
    x_treino = x_treino.astype('float32') / max_value
    x_teste = x_teste.astype('float32') / max_value
    noise_factor = 0.5  # Desvio padrão da distribuição normal
    
    x_train_noisy = x_treino.copy()
    x_test_noisy = x_teste.copy()
    
    
    if 'n' in ruidos:
        x_train_noisy = np.ones((x_treino.shape[1], x_treino.shape[2])) - x_train_noisy
        x_test_noisy = np.ones((x_treino.shape[1], x_treino.shape[2])) - x_test_noisy
    
    if 'l' in ruidos:  # adiciona linha na frente da letra
        for img in x_train_noisy:      
            ind1, ind2 = np.random.randint(0, 25, 2)
            cv.line(img, (0, ind1), (24, ind2), 0, 5)
        
        for img in x_test_noisy:
            ind1, ind2 = np.random.randint(0, 25, 2)
            cv.line(img, (0, ind1), (24, ind2), 0, 5)
    
    if 'i' in ruidos:  # cria iluminação artificial
        dim = x_treino.shape[1:]  # dimensão das imagens
        a = np.ones((dim[0], dim[1]))
        
        for i in range(len(x_train_noisy)):
            m_x, m_y = np.random.randint(0, dim[0], 2) # coordenadas da média
            desvio = np.random.randint(8,13)  # valores nos argumentos são arbitrários
            for lin in range(dim[0]):
                for col in range(dim[1]):
                    a[lin][col] = np.exp(-((lin-m_x)**2/(2*desvio**2)+(col-m_y)**2/(2*desvio**2)))
            x_train_noisy[i] = np.multiply(x_train_noisy[i], a)
            
        for i in range(len(x_test_noisy)):
            m_x, m_y = np.random.randint(0, dim[0], 2) 
            desvio = np.random.randint(8,13)
            for lin in range(dim[0]):
                for col in range(dim[1]):
                    a[lin][col] = np.exp(-((lin-m_x)**2/(2*desvio**2)+(col-m_y)**2/(2*desvio**2)))
            x_test_noisy[i] = np.multiply(x_test_noisy[i], a)
            
    if 'g' in ruidos:  # adiciona ruído gaussiano
        x_train_noisy = x_train_noisy + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_treino.shape) 
        x_test_noisy = x_test_noisy + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_teste.shape) 
    
    
    
    x_treino = x_treino.reshape((len(x_treino), np.prod(x_treino.shape[1:])))
    x_teste = x_teste.reshape((len(x_teste), np.prod(x_teste.shape[1:])))
    x_train_noisy = x_train_noisy.reshape((len(x_treino), np.prod(x_treino.shape[1:])))
    x_test_noisy = x_test_noisy.reshape((len(x_teste), np.prod(x_teste.shape[1:])))
    return x_treino, x_teste, x_train_noisy, x_test_noisy


def plt1():
    #Print das figuras sem ruído (linha 1) e com ruído (linha 2)
    n = 10  # Quantidade de letras impressas
    plt.figure(figsize=(15,3))
    for i in range(n):
        #Imagem original
        plt.subplot(2, n, i + 1)
        plt.imshow(x_teste[i].reshape(img_shape[0], img_shape[1]))
        plt.gray(), plt.axis('off')
        
        #Imagem com ruído
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(img_shape[0], img_shape[1]))
        plt.gray(), plt.axis('off')
        
    plt.show()
    
    
def plt2():
    n = 10  # Quantidade de letras impressas
    plt.figure(figsize=(15,6))
    for i in range(n):
        #Imagem original
        plt.subplot(4, n, i + 1)
        plt.imshow(x_teste[i].reshape(img_shape[0], img_shape[1]))
        plt.gray(), plt.axis('off')
        
        #Imagem com ruído
        plt.subplot(4, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(img_shape[0], img_shape[1]))
        plt.gray(), plt.axis('off')
        
        #Imagem codificada
        plt.subplot(4, n, i + 1 + 2*n)
        plt.imshow(encoded_imgs[i].reshape(enc_dis[0], enc_dis[1]))
        plt.gray(), plt.axis('off')
        
        #Imagem reconstruída
        plt.subplot(4, n, i + 1 + 3*n)
        plt.imshow(decoded_imgs[i].reshape(img_shape[0], img_shape[1]))
        plt.gray(), plt.axis('off')
    
    plt.show()


def SAE_Model():
    encoding_dim = 25 #Menor tamanho da representação da imagem
    #enc_dis[0]*enc_dis [1] = encoding_dim
    enc_dis = (5,5) #Tamanho da imagem codificada. 
    
    #input_dim  é o tamanho da camada de entrada (qtd de pixels da imagem)
    input_dim = x_treino.shape[1]
    
    #Rede Neural SAE
    autoencoder = Sequential()
    
    #Encoder Layers
    autoencoder.add(Dense(4 * encoding_dim, input_shape=(input_dim,), 
                          activation='relu')) #625 -> 100
    autoencoder.add(Dense(2 * encoding_dim, activation='relu')) #100 -> 50
    autoencoder.add(Dense(encoding_dim, activation='relu')) #50 -> 25
    
    #Decoder Layers
    autoencoder.add(Dense(2 * encoding_dim, activation='relu')) #25 -> 50
    autoencoder.add(Dense(4 * encoding_dim, activation='relu')) #50 -> 100
    autoencoder.add(Dense(input_dim, activation='sigmoid')) #100 -> 625
    
    autoencoder.summary() #Mostra no console a arquitetura da rede 
    
    #Com a finalidade de mostrar como a imagem fica em sua forma codificada
    input_img = Input(shape=(input_dim,))
    encoder_layer1 = autoencoder.layers[0]
    encoder_layer2 = autoencoder.layers[1]
    encoder_layer3 = autoencoder.layers[2]
    encoder = Model(input_img, 
                    encoder_layer3(encoder_layer2(encoder_layer1(input_img))))
    
    #Mostra no console a arquitetura da rede (Parte do Encoder)
    encoder.summary()
    
    return autoencoder, encoder, input_dim, encoding_dim, enc_dis 


def predict():
    #Predição dos dados sem ruído
    rotulos = model.predict_classes(x_teste)
    cm = confusion_matrix(y_teste, rotulos)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9,9))
    title = 'Predição dos dados sem ruído\n'+('Precisão de: {}%'.format(
            np.trace(cm)/len(y_teste)*100))
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], 
                          title=title)
    plt.show()
    
    #Predição dos dados ruidosos que NÃO foram submetidos a rede SAE
    rotulos = model.predict_classes(x_test_noisy)
    cm = confusion_matrix(y_teste, rotulos)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9,9))
    title = 'Predição dos dados ruidosos que NÃO foram submetidos a rede SAE\nPrecisão de: {}%'.format(np.trace(cm)/len(y_teste)*100)
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], 
                          title=title)
    plt.show()
        
    #Predição dos dados ruidosos que foram submetidos a rede SAE
    rotulos = model.predict_classes(decoded_imgs)
    cm = confusion_matrix(y_teste, rotulos)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9,9))
    title = 'Predição dos dados ruidosos que foram submetidos a rede SAE\nPrecisão de: {}%'.format(np.trace(cm)/len(y_teste)*100)
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], 
                          title=title)
    plt.show()
   
     
def save_var():
    np.save('x_treino.npy', x_treino)
    np.save('x_teste.npy', x_teste)
    np.save('y_treino.npy', y_treino)
    np.save('y_teste.npy', y_teste)
    np.save('x_train_noisy.npy', x_train_noisy)
    np.save('x_test_noisy.npy', x_test_noisy)
    np.save('x_test_noisy.npy', x_test_noisy)
    np.save('encoding_dim.npy', encoding_dim)
    np.save('enc_dis.npy', enc_dis)
    np.save('encoded_imgs.npy', encoded_imgs)
    np.save('decoded_imgs.npy', decoded_imgs)
    np.save('pMode.npy', pMode)
    
    W_SAE = autoencoder.get_weights()
    np.save('W_SAE.npy', W_SAE)
    
    W_MLP = model.get_weights()
    np.save('W_MLP.npy', W_MLP)
    
    
def boot():
    x_treino = np.load('x_treino.npy')
    x_teste = np.load('x_teste.npy')
    y_treino = np.load('y_treino.npy')
    y_teste = np.load('y_teste.npy')
    x_train_noisy = np.load('x_train_noisy.npy')
    x_test_noisy = np.load('x_test_noisy.npy')
    encoding_dim = np.load('encoding_dim.npy')
    enc_dis = np.load('enc_dis.npy')
    encoded_imgs = np.load('encoded_imgs.npy')
    decoded_imgs = np.load('decoded_imgs.npy')
    pMode = np.load('pMode.npy')
    
    W_SAE = np.load('W_SAE.npy')
    W_MLP = np.load('W_MLP.npy')
    return (x_treino, x_teste, y_treino, y_teste, x_train_noisy, 
            x_test_noisy, encoded_imgs, decoded_imgs, pMode, W_SAE, W_MLP, 
            encoding_dim, enc_dis)

def SAE_2D():
    encoding_dim = 2 #Menor tamanho da representação da imagem
    
    #input_dim  é o tamanho da camada de entrada (qtd de pixels da imagem)
    input_dim2D = x_treino.shape[1]
    
    #Rede Neural SAE
    autoencoder2D = Sequential()
    
    #Encoder Layers
    autoencoder2D.add(Dense(64 * encoding_dim, input_shape=(input_dim,), 
                          activation='relu')) #625 -> 100
    autoencoder2D.add(Dense(32 * encoding_dim, activation='relu')) #100 -> 50
    autoencoder2D.add(Dense(12 * encoding_dim, activation='relu')) #100 -> 50
    autoencoder2D.add(Dense(4 * encoding_dim, activation='relu')) #100 -> 50
    autoencoder2D.add(Dense(encoding_dim, activation='relu')) #50 -> 25
    
    #Decoder Layers
    autoencoder2D.add(Dense(4 * encoding_dim, activation='relu')) #25 -> 50
    autoencoder2D.add(Dense(12 * encoding_dim, activation='relu')) #25 -> 50
    autoencoder2D.add(Dense(32 * encoding_dim, activation='relu')) #25 -> 50
    autoencoder2D.add(Dense(64 * encoding_dim, activation='relu')) #50 -> 100
    autoencoder2D.add(Dense(input_dim, activation='sigmoid')) #100 -> 625
    
    autoencoder2D.summary() #Mostra no console a arquitetura da rede 
    
    #Com a finalidade de plotar a imagem codificada no plano cartesiano
    #ENCODER
    input_img2D = Input(shape=(input_dim2D,))
    encoder_layer1 = autoencoder2D.layers[0]
    encoder_layer2 = autoencoder2D.layers[1]
    encoder_layer3 = autoencoder2D.layers[2]
    encoder_layer4 = autoencoder2D.layers[3]
    encoder_layer5 = autoencoder2D.layers[4]
    encoder2D = Model(input_img2D, encoder_layer5(encoder_layer4(encoder_layer3(encoder_layer2(encoder_layer1(input_img2D))))))
    
    #Mostra no console a arquitetura da rede (Parte do Encoder)
    encoder2D.summary()
    
    #Com o objetivo de reconstruir a imagem
    #DECODER
    input_img2D = Input(shape=(2,))
    decoder_layer5 = autoencoder2D.layers[5]
    decoder_layer6 = autoencoder2D.layers[6]
    decoder_layer7 = autoencoder2D.layers[7]
    decoder_layer8 = autoencoder2D.layers[8]
    decoder_layer9 = autoencoder2D.layers[9]
    decoder2D = Model(input_img2D, decoder_layer9(decoder_layer8(decoder_layer7(decoder_layer6(decoder_layer5(input_img2D))))))
    
    #Mostra no console a arquitetura da rede (Parte do Encoder)
    decoder2D.summary()
    
    autoencoder2D.compile(optimizer='adadelta', loss='binary_crossentropy', 
                        metrics=['accuracy'])
    monitor = callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, 
                                      patience=2, verbose=2, mode='auto')
    #Treina a rede SAE
    SAE2D = autoencoder2D.fit(x_train_noisy, x_treino,
                    epochs=70,
                    batch_size=20,
                    validation_data=(x_test_noisy, x_teste), 
                    callbacks=[monitor], verbose=2)
    
    #Valor da função custo e precisão da rede durante o treinamento
    plt.plot(SAE2D.history['loss'])
    plt.plot(SAE2D.history['acc'])
    plt.legend(('Custo','Precisão'), loc='upper right')
    plt.title('Históricos')
    plt.xlabel('Época'); plt.show()
    
    return encoder2D, decoder2D, autoencoder2D
    
    
def plt3(data, label, title):
    '''
    Esta função plota a imagem codificada em 2D no plano cartesiano.
    Cada classe (letra) tem uma respectiva cor associada.
    '''
    #Predição da imagem (com modificação a.k.a ruído) codificada
    encoded_imgs2D = encoder2D.predict(data)
    cores = ['red', 'green', 'blue', 'cyan', 'purple', 'magenta', 
             'black', 'yellow', 'orange']
    legenda = ['a', 'b', 'c', 'd', 'e', 'f', 'i', 'o', 'u']
    res = []
    for i in range(len(set(label))):
        res.append(encoded_imgs2D[label == i])
    for i in range(len(res)):
        plt.scatter(res[i][:,0], res[i][:,1], color = cores[i], 
                    label = legenda[i])
    plt.title(title)
    plt.legend(legenda, loc='center left', bbox_to_anchor=(1, 0.69), 
               fancybox=True, shadow=True)
    plt.show()
    
    return res
    
    
def features(res, label, title):   
    '''
    res: Lista com os pares (x,y) de cada letra.
    label: O rótulo dos dados que foram codificados.
    title: título da imagem a ser plotada.
    
    Essa função extrai a estatística dos dados plotados com a função plt3().
    slope: Coeficiente angular da reta.
    intercept: Coeficiente linear da reta.
    r_value: Correlação dos dados.
    p_value: p valor para teste de hipótese.
    std_err: Desvio padrão
    '''    
    index = ['a', 'b', 'c', 'd', 'e', 'f', 'i', 'o', 'u']
    columns = ['slope', 'intercept', 'r_value', 'p_value', 'std_err']
    cores = ['red', 'green', 'blue', 'cyan', 'purple', 'magenta', 
             'black', 'yellow', 'orange']
    estatisticas = {}
    slope = []
    intercept = []
    r_value = []
    p_value = []
    std_err = []
    for i in range(len(res)):
        temp = stats.linregress(res[i][:,0], res[i][:,1])
        slope.append(temp[0])
        intercept.append(temp[1])
        r_value.append(temp[2])
        p_value.append(temp[3])
        std_err.append(temp[4])
    est = [slope, intercept, r_value, p_value, std_err]
    for i in range(len(columns)):
        estatisticas[columns[i]] = est[i]
    
    estFrame = pd.DataFrame(data = estatisticas, index = index)
    print(estFrame)
    
    #Plotar as retas da regressão linear
    x = np.linspace(0, 40, 100)
    for i in range(len(res)):
        for j in range(len(x)):
            plt.plot(x, x*slope[i] + intercept[i], color=cores[i])
        plt.scatter(res[i][:,0], res[i][:,1], color = cores[i])
    plt.title(title)
    plt.show()
    
    return estFrame

def unzip(x, frame, letra):
    '''
    x: inteiro ou float que dá uma posição no plano cartesiano.
    frame: DataFrama do pandas com as estatísticas dos dados.
    letra: Uma das letras do dataset (a, b, c, d, e, f, i, o, u).
    
    Essa função, a partir de um valor de x consegue decodificar algo em 2D 
    para a dimensão original (25,25).
    
    O valor de y é calculado através da equação da reta, que depende de cada 
    letra.
    '''
    #Predição da imagem com ruído (espera-se que a rede tenha eliminado o ruído)
    #decoded_imgs2D_teste = autoencoder2D.predict(x_test_noisy)
    y = x*frame['slope'][letra] + frame['intercept'][letra]
    _2Dimg = np.array([[x, y],])
    decoded_img = decoder2D.predict(_2Dimg)
    decoded_img = decoded_img.reshape(25,25)
    plt.imshow(decoded_img)
    

#MODE pode ser 'START' ou 'BOOT'
MODE = 'START'
_2D = 'FALSE'
if MODE == 'START':
    x_treino, x_teste, y_treino, y_teste = read_img()
    img_shape = x_treino.shape[1:]
    pMode='g' #Forma com que a imagem será modificada
    x_treino, x_teste, x_train_noisy, x_test_noisy = pre_processamento(x_treino, x_teste, pMode)
    
    #Print das figuras sem ruído (linha 1) e com ruído (linha 2)
    plt1()
    
    #Construção da rede neural SAE (Stacked Autoencoder)
    #A rede terá a arquitetura (625, 100, 50, 25, 50, 100, 625)
    autoencoder, encoder, input_dim, encoding_dim, enc_dis = SAE_Model()
    #autoencoder2D, encoder2D, input_dim2D, encoding_dim2D = SAE_2D()
    #Parâmetros para o treinamento da rede neural
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', 
                        metrics=['accuracy'])
    monitor = callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, 
                                      patience=2, verbose=2, mode='auto')
    #Treina a rede SAE
    SAE = autoencoder.fit(x_train_noisy, x_treino,
                    epochs=50,
                    batch_size=10,
                    validation_data=(x_test_noisy, x_teste), 
                    callbacks=[monitor], verbose=2)
    
    #Valor da função custo e precisão da rede durante o treinamento
    plt.plot(SAE.history['loss'])
    plt.plot(SAE.history['acc'])
    plt.legend(('Custo','Precisão'), loc='upper right')
    plt.title('Históricos')
    plt.xlabel('Época'); plt.show()
    
    #Predição da imagem codificada
    encoded_imgs = encoder.predict(x_test_noisy)
    #Predição da imagem com ruído (espera-se que a rede tenha eliminado o ruído)
    decoded_imgs = autoencoder.predict(x_test_noisy)
    
    #Plot dos resultados
    plt2()
    
    ###############################################################################
    #A rede neural abaixo é uma MLP que irá classificar as imagens obtidas na 
    #saída da SAE.
    model = Sequential()
    model.add(Dense(10, input_shape=(input_dim,), activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    
    model.summary() #Mostra no console a arquitetura da rede (Parte do Encoder)
    
    #Parâmetros para o treinamento da rede neural
    model.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    monitor = callbacks.EarlyStopping(monitor='loss', min_delta=1e-3,
                                      patience=2, verbose=2, mode='auto')
    
    #Treina a rede MLP
    H = model.fit(x_treino, y_treino, epochs=40, batch_size=20, 
                  validation_data=(x_teste, y_teste), 
                  callbacks=[monitor], verbose=2) 
    
    #Valor da função custo e precisão da rede durante o treinamento
    plt.plot(H.history['loss'])
    plt.plot(H.history['acc'])
    plt.legend(('Custo','Precisão'), loc='upper right')
    plt.title('Históricos')
    plt.xlabel('Época'); plt.show()
    
    predict()
    
    if _2D == 'TRUE':
        encoder2D, decoder2D, autoencoder2D = SAE_2D()
        W_SAE2D = autoencoder.get_weights()
        np.save('W_SAE2D.npy', W_SAE2D)
        
        x_test_noisy2D_title = 'Representação em 2D dos DADOS DE TESTE (imagens modificadas)'
        res_x_test_noisy = plt3(x_test_noisy, y_teste, x_test_noisy2D_title)
        
        x_treino2D_title = 'Representação em 2D dos DADOS DE TREINO (imagens originais)'
        res_x_train_noisy = plt3(x_train_noisy, y_treino, x_treino2D_title)
        
        title = 'Dados de Teste com Ruído codificados em 2D e Regressão Linear'
        frameTest_noisy = features(res_x_test_noisy, y_teste, title)
        
        title = 'Dados de Treino codificados em 2D e Regressão Linear'
        frameTreino = features(res_x_train_noisy, y_treino, title)
        
        frame = frameTest_noisy 
        letra = 'o'
        x = 1
        unzip(x, frame, letra)
        
        
elif MODE == 'BOOT':
    (x_treino, x_teste, y_treino, y_teste, x_train_noisy, 
            x_test_noisy, encoded_imgs, decoded_imgs, pMode, W_SAE, W_MLP, 
            encoding_dim, enc_dis) = boot()
    autoencoder, encoder, input_dim, encoding_dim, enc_dis = SAE_Model()
    autoencoder.set_weights(W_SAE)
    model = Sequential()
    model.add(Dense(10, input_shape=(input_dim,), activation='sigmoid'))
    model.add(Dense(9, activation='softmax'))
    model.set_weights(W_MLP)
    plt2()
    predict()
    
else:
    print('NÃO EXISTE ESTE MODO. TENDE O MODO <START> OU <BOOT>')
    
save_var()