import os
import cv2 # OpenCV ou cv2 para tratamento de imagens;
import numpy as np # Numpy para trabalharmos com matrizes n-dimensionais
import pandas as pd # Pandas para tratar data frames de dados gerados em algumas parte do treinamento
import tensorflow as tf

# optimizador Adam
from tensorflow.keras.optimizers import Adam  
# Função de conversão da imagem para um vetor
from tensorflow.keras.preprocessing.image import img_to_array
# Função utilizada para categorizar listas de treino e teste
from tensorflow.keras.utils import to_categorical
# Classe para ajudar na variação de amostras de treinamento
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Classe utilizada para acompanhamento durante o treinamento onde definimos os atributos que serão considerados para avaliação
from tensorflow.python.keras.callbacks import ModelCheckpoint



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

def create_model_arquitecture(input_data):
    """
    Em seguida, 
    carregamos a arquitetura do modelo e os pesos
    já treinados do
    imagenet para as redes.
    """

    img_shape = (128, 128, 3)
   
    model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape,
            include_top=False,
            weights="imagenet"
        )

    return model

if __name__== '__main__':

    train_path = './Data/train' # Diretorio para os arquivos de treinamento
    test_path = './Data/test' # Diretorio para os arquivos de treinamento
    models_path = "transfer_model" # Diretorio para os modelos treinados
    width = 128  # Tamanho da largura da janela que será utilizada pelo modelo com alteração para uma arquitetura maior
    height = 128 # Tamanho da altura da janela que será utilizada pelo modelo tambem com o valor alterado para a atual arquitetura
    depth = 3 # Profundidade das janelas utilizadas pelo modelo, caso seja RGB use 3, caso escala de cinza 1
    classes = 2 # Quantidade de classes que o modelo utilizará 
    epochs = 10 # Quantidade de épocas (a quantidade de iterações que o modelo realizará durante o treinamento)
    init_lr = 1e-3 # Taxa de aprendizado a ser utilizado pelo optimizador
    batch_size = 32 # Tamanho dos lotes utilizados por cada epoca
    input_shape = (height, width, depth) # entrada do modelo

    #optei por não usar este comando por que apresentou varios erros de incompatibilidade com meu sistema!
    #save_mode = os.path.join(models_path, "lenet-{epoch:02d}-{acc:.3f}-{val_acc:.3f}.csv"), os.makedirs(models_path, exist_ok=True)
    
    (trainX, trainY) = get_data_x_and_y(train_path, width, height, depth, classes)
    (testX, testY) = get_data_x_and_y(test_path, width, height, depth, classes)
    
    # Gerador de imagens randomicas para testes
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2, zoom_range=0.2,
        fill_mode="nearest"
    )
    
    
    model = create_model_arquitecture(input_shape) # Intanciando arquitetura pré-executada
    
    # Definindo otimizador do modelo
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    
    # Congelamento de treino das primeiras camadas 
    model.trainable = False 
    # Definindo o cabeçalho personalizado da rede neural
    average_layer = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    predict_layers = tf.keras.layers.Dense(2, activation='softmax')(average_layer)

    # Definindo o modelo
    model = tf.keras.models.Model(inputs = model.input, outputs = predict_layers)
    
    # Compilando o modelo
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr = init_lr ),
              loss="categorical_crossentropy", metrics = ["accuracy"])

    model.summary()
    
    print("\n Treinando e classificando !")
    print("\n Comecei!")
    
    # Definindo pointers de pesos que irão armasenar a avariavei de treinamento
    checkpoint1 = ModelCheckpoint(models_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint2 = ModelCheckpoint(models_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')  
    callbacks_list = [checkpoint1, checkpoint2]

    # Treinando o modelo
    history = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size),
                            validation_data=(testX, testY), steps_per_epoch=len(
                                trainX) // batch_size,
                            epochs=epochs, verbose=1, callbacks=callbacks_list)

    # Epresão que espôen valores de precisão e perca do modelo após o treinamento
    valid_loss, valid_accuracy = model.evaluate_generator(testX)

    #Salvando o modelo através de tipagem Data Frame
    hist_df = pd.DataFrame(history.history)
    print(hist_df)
    hist_csv_file = './Csvs/Cvs_epochs_10/Transfer_data.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    