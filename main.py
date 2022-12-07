# -*- coding: utf-8 -*-


import tensorflow, pandas, numpy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def draw_graphics(training_hist):
    # Getting metrics
    loss = training_hist.history['loss']
    val_loss = training_hist.history['val_loss']
    acc = training_hist.history['acc']
    val_acc = training_hist.history['val_acc']
    epochs = range(1, len(loss) + 1)

    # Drawing loss graphic
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Drawing accuracy graphic
    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def create_model():
    # Reading a csv-file with data set and preparing data (training input data - X and outputs - Y)
    try:
        data_frame = pandas.read_csv('./stats.csv')
    except BaseException:
        print('ERROR: Data set csv-file was deleted, damaged or renamed!')
        return
    data_set = data_frame.values
    X = data_set[:, 0:11].astype(float)
    Y = data_set[:, 11].astype(float)
    encoder = LabelEncoder()
    Y = tensorflow.keras.utils.to_categorical(encoder.fit_transform(Y))

    # Creating the NN-model
    nn_model = tensorflow.keras.models.Sequential()
    nn_model.add(tensorflow.keras.layers.Dense(X.shape[1], activation='relu', input_dim=X.shape[1]))
    nn_model.add(tensorflow.keras.layers.Dense(X.shape[1], activation='relu'))
    nn_model.add(tensorflow.keras.layers.Dense(X.shape[1], activation='relu'))
    nn_model.add(tensorflow.keras.layers.Dense(X.shape[1], activation='relu'))
    nn_model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

    # Compiling and training the model
    nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    training_hist = nn_model.fit(X, Y, epochs=30, batch_size=5, validation_split=0.1)
    nn_model.save('./model.nn')

    # Drawing graphics and printing training results
    draw_graphics(training_hist)
    results = nn_model.evaluate(X, Y)
    print(results)


def test_model(input_x):
    try:
        nn_model = tensorflow.keras.models.load_model('./model.nn')
    except BaseException:
        print('ERROR: Saved NN-model file was deleted, damaged or renamed!')
        return
    overtime_chance = nn_model.predict(input_x)[0][1]
    print('Predicted overtime chance: ', round(overtime_chance * 100, 2), '%')


# Program processing ...
if input('Do You want to create a new NN-model (Yes / Other word)? ') == 'Yes':
    create_model()
if input('Do You want to test saved NN-model (Yes / Other word)? ') == 'Yes':
    input_data = input('Enter an input match data: ')
    test = numpy.array([float(i) for i in input_data.split()])
    test_model(test.reshape(1, test.shape[0]))
