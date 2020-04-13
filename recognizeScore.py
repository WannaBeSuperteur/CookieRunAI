# PURPOSE OF THIS FILE #
# to make image(number of score) recognization model

import random
import math
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Model, model_from_json
from keras import backend as K
from PIL import Image

import deepLearning_GPU as DL
import deepLearning_GPU_helper as DLH

## return array that represents each image in location
## each element of array represent each image with width 'width' and height 'height'
# location   : location where train and test images exist
# width      : width of each element in the array
# height     : height of each element in the array
def loadImgs(location, width, height):
    file_list = os.listdir(testImgLoc)

    imgArray = [] # array to return
    labels = [] # labels to return

    for file in range(len(file_list)):
        im = Image.open(testImgLoc + file_list[file])
        im = im.resize((width, height))
        
        imArray = np.array(im) # [height][width][3] numpy array of this image

        array = [[0]*width for _ in range(height)] # [height][width] array of this image

        for i in range(len(array[0])):
            for j in range(len(array)):
                array[j][i] = int(sum(imArray[j][i])/77)-4 # int range -4 ~ 5 (based on brightness : sum of R, G and B values)

        # append this array to imgArray
        imgArray.append(array)
        labelArray = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # index of that number = 1, other = 0
        labelArray[int(file_list[file][0])] = 1
        labels.append(labelArray) # label is ALWAYS equal to first character of file name

    return [imgArray, labels, file_list]

## training
# imgArray   : array that contains all the images FOR TRAIN: in the form of [imgs][height][width]
# labels     : array that contains all the labels for images FOR TRAIN: in the form of [imgs]
# drop       : dropout
# lr         : learning rate
# epoch      : epoch
# height     : height of image
# width      : width of image
# modelName  : name of deep learning model
# deviceName : name of device
def train(imgArray, labels, modelName, drop, lr, epoch, height, width, deviceName):

    # Neural Network
    NN = [tf.keras.layers.Reshape((height, width, 1), input_shape=(height, width,)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(height, width, 1), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(drop),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.Dropout(drop),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dropout(drop),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(40, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')]

    # Optimizer
    op = tf.keras.optimizers.Adam(lr)

    # learning
    DL.deepLearning(NN, op, 'mean_squared_error', imgArray, labels, modelName, epoch, False, True, deviceName)
    
## test
# imgArray    : array that contains all the images FOR TEST: in the form of [imgs][height][width]
# answerArray : array of correct answer: in the form of [answer0, answer1, ...]
# testFileList: list of test files
# modelName   : name of deep learning model
def test(imgArray, answerArray, testFileList, modelName):

    count = len(imgArray) # number of images
    correct = 0 # number of correctly classified images
    
    newModel = DL.deepLearningModel(modelName, True)
    testOutput = DL.modelOutput(newModel, imgArray) # deep learning output for test data (imgArray)

    # test result
    for i in range(count):
        outputLayer = testOutput[len(testOutput)-1][i] # output layer array of test result

        # find index of maximum value in outputLayer
        maxIndex = 0 # index of maximum value
        maxVal = 0
        for j in range(10):
            if outputLayer[j] > maxVal:
                maxIndex = j
                maxVal = outputLayer[j]

        # compare maxIndex with correct answer
        for j in range(10): outputLayer[j] = round(max(0.000499, outputLayer[j]-0.0005), 3) # to print
        print(testFileList[i] + ' / out: ' + str(outputLayer) + ' / ans: ' + str(answerArray[i]))
        if answerArray[i][maxIndex] == 1: correct += 1

    print('correct rate: ' + str(correct) + ' / ' + str(count) + ', ' + str(round(100*correct/count, 2)) + '%')

## default training and test function
# drop       : dropout
# lr         : learning rate
# height     : height of image
# width      : width of image
# modelName  : name of deep learning model
# deviceName : name of device
# testImgLoc : location where test images exist
# trainProb  : probability that each data is designated as data for training
# testing    : do test?
def defaultTrainAndTest(drop, lr, epoch, height, width, modelName, deviceName, testImgLoc, trainProb, testing):
    a = loadImgs(testImgLoc, 16, 24)
    imgArray = a[0] # in the form of [height][width]
    labels = a[1] # in the form of [0, 0, ..., 0] (10 elements), only (label)-th element is 1
    file_list = a[2] # list of image file name

    # extract train and test images from 'imgArray' and 'labels'
    trainImgArray = [] # image for train
    trainLabels = [] # label for train
    testFileList = [] # file list for test
    testImgArray = [] # image for test
    testLabels = [] # label for test

    for i in range(len(imgArray)):
        if random.random() < trainProb: # each image is for train with probability (trainProb)
            trainImgArray.append(imgArray[i])
            trainLabels.append(labels[i])
        else: # each image is for test with probability (1-trainProb)
            testFileList.append(file_list[i])
            testImgArray.append(imgArray[i])
            testLabels.append(labels[i])

    # train and test
    train(trainImgArray, trainLabels, modelName, drop, lr, epoch, height, width, deviceName)

    if testing == True:
        test(testImgArray, testLabels, testFileList, modelName)
        print('\ntrained using ' + str(len(trainLabels)) + ' images:\n' + str(list(set(file_list) - set(testFileList))) + '\n')

# MAIN FUNCTION
if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    testImgLoc = 'images/test/' # location where test images exist

    # train and test
    defaultTrainAndTest(drop=0, lr=0.0001, epoch=1500, height=24, width=16, modelName='scoreRecognize',
                        deviceName=deviceName, testImgLoc=testImgLoc, trainProb=0.75, testing=True)
