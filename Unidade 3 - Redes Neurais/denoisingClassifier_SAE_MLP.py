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
        print('Matriz de confusão normalizada')
    else:
        print('Matriz de confusão sem normalização')


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

def pre_processamento(x_treino, x_teste):
    '''
    Esta função primeiro escala os dados que originalmente estão entre 
    [0, 255] para o intervalo [0, 1].
    Em seguida, a imagem que é de dimensão nxn se transforma em um vetor de 
    tamanho n² (n*n).
    Depois é aplicado um ruído gaussiano a imagem. Valores que ficarem fora do 
    intervalo [0, 1] são 'saturados' para obedecerem o intervalo.
    Esta função retorna os dados normalizados assim como suas versões com 
    ruído.
    '''
    max_value = float(x_treino.max())
    x_treino = x_treino.astype('float32') / max_value
    x_teste = x_teste.astype('float32') / max_value
    
    x_treino = x_treino.reshape((len(x_treino), np.prod(x_treino.shape[1:])))
    x_teste = x_teste.reshape((len(x_teste), np.prod(x_teste.shape[1:])))
    
    noise_factor = 0.7 #Desvio padrão da distribuição normal
    x_train_noisy = x_treino + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=x_treino.shape) 
    
    x_test_noisy = x_teste + noise_factor * np.random.normal(
            loc=0.0, scale=1.0, size=x_teste.shape) 
    
    '''
    Clipando (np.clip()) ou não, o resultado final é praticamente o mesmo
    Clipar mantém o fundo predominantemente branco, não clipar deixa o fundo 
    com tons de cinza.
    '''
    #x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    #x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    return x_treino, x_teste, x_train_noisy, x_test_noisy

def plt1():
    #Print das figuras sem ruído (linha 1) e com ruído (linha 2)
    n = 10  # how many digits we will display
    plt.figure(figsize=(15,3))
    for i in range(n):
        #Imagem original
        plt.subplot(2, n, i + 1)
        plt.imshow(x_teste[i].reshape(img_shape[0], img_shape[1]))
        plt.gray()
        plt.axis('off')
    
        #Imagem com ruído
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(img_shape[0], img_shape[1]))
        plt.gray()
        plt.axis('off')
        
    plt.show()
    
def plt2():
    n = 10  #Quantidade de dígitos que serão mostrados
    plt.figure(figsize=(15,6))
    for i in range(n):
        #Imagem original
        plt.subplot(4, n, i + 1)
        plt.imshow(x_teste[i].reshape(img_shape[0], img_shape[1]))
        plt.gray()
        plt.axis('off')
        
        #Imagem com ruído
        plt.subplot(4, n, i + 1 + n)
        plt.imshow(x_test_noisy[i].reshape(img_shape[0], img_shape[1]))
        plt.gray()
        plt.axis('off')
        
        #Imagem codificada
        plt.subplot(4, n, i + 1 + 2*n)
        plt.imshow(encoded_imgs[i].reshape(enc_dis[0], enc_dis[1]))
        plt.gray()
        plt.axis('off')
        
        #Imagem reconstruída
        plt.subplot(4, n, i + 1 + 3*n)
        plt.imshow(decoded_imgs[i].reshape(img_shape[0], img_shape[1]))
        plt.gray()
        plt.axis('off')
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
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], title='Predição dos dados sem ruído')
    plt.show()
        
    print('Precisão de: {}%'.format(np.trace(cm)/len(y_teste)*100))
    
    #Predição dos dados ruidosos que NÃO foram submetidos a rede SAE
    rotulos = model.predict_classes(x_test_noisy)
    cm = confusion_matrix(y_teste, rotulos)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9,9))
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], title='Predição dos dados ruidosos que foram submetidos a rede SAE')
    plt.show()
        
    print('Precisão de: {}%'.format(np.trace(cm)/len(y_teste)*100))
        
    #Predição dos dados ruidosos que foram submetidos a rede SAE
    rotulos = model.predict_classes(decoded_imgs)
    cm = confusion_matrix(y_teste, rotulos)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(9,9))
    plot_confusion_matrix(cm, ['a','b','c','d','e','f','i','o','u'], title='Predição dos dados ruidosos que foram submetidos a rede SAE')
    plt.show()
        
    print('Precisão de: {}%'.format(np.trace(cm)/len(y_teste)*100))

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
    W_SAE = np.load('W_SAE.npy')
    W_MLP = np.load('W_MLP.npy')
    return (x_treino, x_teste, y_treino, y_teste, x_train_noisy, 
            x_test_noisy, encoded_imgs, decoded_imgs, W_SAE, W_MLP, 
            encoding_dim, enc_dis)

#MODE pode ser 'START' ou 'BOOT'
MODE = 'START'
if MODE == 'START':
    x_treino, x_teste, y_treino, y_teste = read_img()
    img_shape = x_treino.shape[1:]
    x_treino, x_teste, x_train_noisy, x_test_noisy = pre_processamento(
            x_treino, x_teste)
    
    #Print das figuras sem ruído (linha 1) e com ruído (linha 2)
    plt1()
    
    #Construção da rede neural SAE (Stacked Autoencoder)
    #A rede terá a arquitetura (625, 100, 50, 25, 50, 100, 625)
    autoencoder, encoder, input_dim, encoding_dim, enc_dis = SAE_Model()
    
    #Parâmetros para o treinamento da rede neural
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', 
                        metrics=['accuracy'])
    monitor = callbacks.EarlyStopping(monitor='loss', min_delta=1e-5, 
                                      patience=2, verbose=2, mode='auto')
    #Treina a rede SAE
    SAE = autoencoder.fit(x_train_noisy, x_treino,
                    epochs=100,
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
                  callbacks=[monitor], verbose=2) 
    
    #Valor da função custo e precisão da rede durante o treinamento
    plt.plot(H.history['loss'])
    plt.plot(H.history['acc'])
    plt.legend(('Custo','Precisão'), loc='upper right')
    plt.title('Históricos')
    plt.xlabel('Época'); plt.show()
    
    predict()

elif MODE == 'BOOT':
    (x_treino, x_teste, y_treino, y_teste, x_train_noisy, 
            x_test_noisy, encoded_imgs, decoded_imgs, W_SAE, W_MLP, 
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
