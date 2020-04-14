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

## return shape-changed image array([height][width]) of original iamge array([height][width][3])
# imArray    : original image array
def imArrayToArray(imArray):

    # height and width
    height = len(imArray)
    width = len(imArray[0])

    # [height][width] array of this image
    array = [[0]*width for _ in range(height)]

    # shape change: [height][width][3] -> [height][width]
    # value (-4.0 ~ +5.0) based on brightness : sum of R, G and B values
    # (-4.0 : BLACK, 0.0 : GRAY, +5.0 : WHITE)
    # [Note: discrete input value is better for training than continuous input value]
    for i in range(len(array[0])):
        for j in range(len(array)):
            array[j][i] = int(sum(imArray[j][i]) / 77) - 4

    return array

## return array that represents each image in location
## each element of array represent each image with width 'width' and height 'height'
# location   : location where train and test images exist
# width      : width of each element in the array
# height     : height of each element in the array
def loadImgs(location, width, height):
    file_list = os.listdir(location)

    imgArray = [] # array to return
    labels = [] # labels to return

    for file in range(len(file_list)):
        im = Image.open(location + file_list[file])
        im = im.resize((width, height))
        
        imArray = np.array(im) # [height][width][3] numpy array of this image
        array = imArrayToArray(imArray)

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
        print(testFileList[i] + ' '*(16-len(testFileList[i])) + ' / out: ' + str(outputLayer) + ' / ans: ' + str(answerArray[i]))
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

## test function for image of numeric value (ex: 1,312,500)
# img        : image of numeric value
# modelName  : name of deep learning model
# w          : width of resized image to input to the model
# h          : height of resized image to input to the model
def testNumeric(img, modelName, w, h):

    # find height and width of image
    width = img.size[0] # width of image
    height = img.size[1] # height of image
    
    print('height=' + str(height) + ', width=' + str(width))

    # initialize top, bottom, left and right
    top = 0
    bottom = height
    left = width
    right = width

    recognized = ''
    model = DL.deepLearningModel(modelName, True)

    # crop and recognize number
    while left >= 0:

        # resize cropped area of image
        # width of cropped image >= 0.45*(bottom-top)
        # -> resize vertically (remove top 2px and bottom 2px rows)
        if right-left >= 0.45*(bottom-top) and bottom-top >= h:
            top += 2
            bottom -= 2
        # otherwise -> resize horizontally
        # -> expand cropped image area to 2 pixels(columns) left 
        else: left -= 2

        # width of cropped image must be at least the value of w
        if right-left < w: continue

        # deep learning output for test data (imgArray)
        croppedImage = img.crop((left, top, right, bottom)) # crop
        tempFileName = 'temp.png'
        croppedImage.save(tempFileName) # temporarily save resized image file
        croppedImage = Image.open(tempFileName) # open the temporarily saved file
        resizedImage = croppedImage.resize((w, h)) # resize
        imArray = np.array(resizedImage) # [height][width][3] numpy array of resized image
        imArray_ = imArrayToArray(imArray) # change the shape([height][width][3]) into [height][width]
        os.remove(tempFileName) # delete the file
        
        testOutput = DL.modelOutput(model, [imArray_])

        # output layer array of test result
        outputLayer = testOutput[len(testOutput)-1][0]

        # find index of maximum value in outputLayer
        maxIndex = 0 # index of maximum value
        maxVal = max(outputLayer)
        for i in range(10):
            if outputLayer[i] == maxVal: maxIndex = i

        # print test result
        print('top,bottom,left,right = ' + str(top) + ' ' + str(bottom) + ' ' + str(left) + ' ' + str(right) +
              ' / maxIndex,maxVal = ' + str(maxIndex) + ' ' + str(round(maxVal, 6)))

        # recognize as number if max(one-hot output) > 0.75 and > 0.75 * sum(one-hot output)
        if maxVal > 0.75 and maxVal > 0.75 * sum(outputLayer):
            print('recognized as ' + str(maxIndex))
            recognized = str(maxIndex) + recognized
            right = left # move right boundary of cropped area to left boundary (so that cropped area is 0px horizontally)

            # 'jump' for the ',' token (move 10 pixels left)
            if len(recognized) % 3 == 0:
                left -= 10
                right -= 10

    # return result
    return recognized

## test function for image of numeric value in a specific location
## suppose: file name indicate the numeric value (ex: 1312500.png -> 1,312,500)
# location   : location where numeric value images exist
# modelName  : name of deep learning model
# w          : width of resized image to input to the model
# h          : height of resized image to input to the model
def testNumericInLocation(location, modelName, w, h):
    file_list = os.listdir(location)

    testAnswer = [] # answer for test

    for file in range(len(file_list)): # for each image file
        im = Image.open(location + file_list[file])

        # answer for the image (compare it with correct answer)
        testAnswer.append(testNumeric(im, modelName, w, h))

    # print result
    for i in range(len(testAnswer)):
        print('answer of model:' + testAnswer[i] + ', right answer:' + str(file_list[i].split('.')[0]))

# MAIN FUNCTION
if __name__ == '__main__':
    deviceName = input('device name (for example, cpu:0 or gpu:0)')
    testImgLoc = 'images/test/' # location where test images exist
    testNumericLoc = 'images/result/' # location where numeric value images exist

    # train and test
    try: # check if scoreRecognize.h5 file exists
        open('scoreRecognize.h5')
    except: # do train and test if the file does not exist
        defaultTrainAndTest(drop=0, lr=0.0001, epoch=200, height=24, width=16, modelName='scoreRecognize',
                            deviceName=deviceName, testImgLoc=testImgLoc, trainProb=0.75, testing=True)

    # test for numeric value
    testNumericInLocation(testNumericLoc, 'scoreRecognize', 16, 24)
