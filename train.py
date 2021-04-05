import os
from re import T
import cv2  # OpenCV ou cv2 para tratamento de imagens;
import numpy as np  # Numpy para trabalharmos com matrizes n-dimensionais
import pandas as pd # Pandas para tratar data frames de dados gerados em algumas parte do treinamento

"""Fiz alterações nas importações das funçoes e classes e 
bibliotecas ultilizadas pois as que inclui conrespondia 
melhor ao meu amibiente virtual de desenvovimento"""

# Retornos para respectivamente paradas antecipadas e redução de platonaria entre epocas de treino
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# Para o agupamento lienar das camadas de treinamento do modelo 
from tensorflow.keras.models import Sequential
# Camada da função de ativação, flatten, entre outros
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam  # optimizador Adam
# Função de conversão da imagem para um vetor
from tensorflow.keras.preprocessing.image import img_to_array
# Função utilizada para categorizar listas de treino e teste
from tensorflow.keras.utils import to_categorical
# Classe para ajudar na variação de amostras de treinamento
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Classe utilizada para acompanhamento durante o treinamento onde definimos os atributos que serão considerados para avaliação
from tensorflow.python.keras.callbacks import ModelCheckpoint
#Camada de convolução 2D
from tensorflow.python.keras.layers.convolutional import Conv2D
# Aplica um pooling máximo 2D em um sinal de entrada composto de vários planos de entrada.
from tensorflow.python.keras.layers.pooling import MaxPooling2D
# Classe para normalização de lotes
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization


def get_data_x_and_y(data_path, width, height, depth, classes):
    """
    Essa função itera pelo data_path para separar dados como rotulos e
        os dados que serão utilizados para o treinamento e teste

    Args:
        data_path: O diretório com os dados
        width: Largura das matriz esperada pelo modelo
        height: Altura das matriz esperada pelo modelo
        classes: Numero de classes que o modelo utilizará

    Returns:
        Uma tupla onde na primeira posição você tem o eixo X e na segunda posição o eixo Y
    """

    labels = []
    data = []

    # itera pelo diretório
    for filename in os.listdir(data_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # faz a leitura de cada imagem
            image = cv2.imread(os.path.join(data_path, filename))
            if depth == 1:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # redimensiona a imagem
            image = cv2.resize(image, (width, height))
            # converte a imagem para um vetor
            image = img_to_array(image)
            # concatena a imagem a lista de dados que serão utilizados pelo treinamento
            data.append(image)
            # concatena a lista de rotulos a classe da imagem
            labels.append(int(filename[5])-1)
    # Normaliza os dados de treinamento
    X = np.array(data, dtype="float32") / 255.0
    # Categoriza os rotulos
    Y = to_categorical(labels, num_classes=classes)
    return (X, Y)

def create_lenet(input_data):
    """
    Cria uma mini arquitetura lenet

    Args:
        input_shape: Uma lista de três valores inteiros que definem a forma de\
                entrada da rede. Exemplo: [100, 100, 3]

    Returns:
        Um modelo sequencial, seguindo a arquitetura lenet
    """
    # Definimos que estamos criando um modelo sequencial
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_data))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    return model

if __name__ == '__main__':
    # Adicione aqui o caminho para chegar no diretório que contém as imagens de treino na sua maquina
    train_path = './Data/train'
    # Adicione aqui o caminho para chegar no diretório que contém as imagens de teste na sua maquina
    test_path = './Data/test'
    models_path = "models"  # Defina aqui onde serão salvos os modelos na sua maquina
    width = 128  # Tamanho da largura da janela que será utilizada pelo modelo
    height = 128  # Tamanho da altura da janela que será utilizada pelo modelo
    depth = 3  # Profundidade das janelas utilizadas pelo modelo, caso seja RGB use 3, caso escala de cinza 1
    classes = 2  # Quantidade de classes que o modelo utilizará
    epochs = 10 # Quantidade de épocas (a quantidade de iterações que o modelo realizará durante o treinamento)
    init_lr = 1e-3  # Taxa de aprendizado a ser utilizado pelo optimizador
    batch_size = 32  # Tamanho dos lotes utilizados por cada epoca
    input_shape = (height, width, depth)  # entrada do modelo
    
    #optei por não usar este comando por que apresentou varios erros de incompatibilidade com meu sistema!
    #save_mode = os.path.join(models_path, "lenet-{epoch:02d}-{acc:.3f}-{val_acc:.3f}.csv"), os.makedirs(models_path, exist_ok=True)

    (trainX, trainY) = get_data_x_and_y(
        train_path, width, height, depth, classes)
    (testX, testY) = get_data_x_and_y(test_path, width, height, depth, classes)

    # Gerador de imagens randomicas para testes
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2,
        fill_mode="nearest"
    )

    # Criando o modelo
    model = create_lenet(input_shape)

    # Definindo o otimizador Adam para o modelo
    opt = Adam(lr=init_lr, decay=init_lr / epochs)

    # Compilando o modelo
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=["accuracy"])

    model.summary()

    print("\n training network")

    # Retorno de chamada para salvar o modelo Keras ou pesos do modelo com alguma frequência.
    checkpoint1 = ModelCheckpoint(models_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(models_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    
    # Um model.fit()loop de treinamento verificará ao final de cada
    # época se a perda não está mais diminuindo, 
    # considerando o min_deltae patiencese aplicável. 
    # Uma vez que não esteja mais diminuindo, model.stop_trainingé 
    # marcado como Verdadeiro e o treinamento termina.
    earlystop = EarlyStopping(patience=10)

    # Pointer Este retorno de chamada monitora uma quantidade 
    # e se nenhuma melhoria for vista para um número de 'paciência' 
    # de épocas, a taxa de aprendizagem é reduzida.
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=2,
        verbose=1,
        factor=0.5,
        min_lr=init_lr
    )
    
    callbacks_list = [earlystop, learning_rate_reduction]

    # Treinamento do modelo
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                            validation_data=(testX, testY), steps_per_epoch=len(
                                trainX) // batch_size,
                            epochs=epochs, verbose=1, callbacks=callbacks_list)

    
    # Epresão que espôen valores de precisão e perca do modelo após o treinamento
    valid_loss, valid_accuracy = model.evaluate_generator(testX)

    #Salvando o modelo através de tipagem Data Frame
    hist_df = pd.DataFrame(H.history)
    print(hist_df)
    hist_csv_file = './Csvs/csv_epochs_10/reduce_plato.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
