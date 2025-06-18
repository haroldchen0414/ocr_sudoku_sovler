# -*- coding: utf-8 -*-
# author: haroldchen0414

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

class SudokuNet:
    def __init__(self):
        self.batchSize = 128
        self.epochs = 20

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), padding="same", input_shape=(28, 28, 1)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(10))
        model.add(Activation("softmax"))

        return model        

    def train(self):
        ((trainX, trainY), (testX, testY)) = mnist.load_data()
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype("float32") / 255.0
        testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype("float32") / 255.0

        le = LabelBinarizer()
        trainY = le.fit_transform(trainY)
        testY = le.transform(testY)

        opt = Adam(learning_rate=1e-3)
        model = self.build()
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=self.batchSize, epochs=self.epochs, verbose=1)

        preds = model.predict(testX)
        print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

        model.save("sudokuNet.h5", save_format="h5")

if __name__ == "__main__":
    solver = SudokuNet()
    solver.train()