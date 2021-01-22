from chainer import Sequential
import chainer.functions as F
import chainer.links as L
import chainer
from chainer import Chain, optimizers, Variable, serializers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math

from . import path

class CNN(Chain):#出力数を受け取ってcnnを作成する
    def __init__(self):
        output_num = 2
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, 32, 5), # filter 5
            conv2 = L.Convolution2D(32, 64, 5), # filter 5
            l1 = L.Linear(256, 200),
            l2 = L.Linear(200, 300),
            l3 = L.Linear(300, output_num, initialW=np.zeros((output_num, 300), dtype=np.float32))
        )
    def forward(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h


class Recognizer():
    @staticmethod
    #rectangleを受け取ってlabelとconfidenceを返す
    def predict(rec):
        rec = rec.reshape((1,1,20,20)).astype(np.float32)
        rec /= 255
        x = Variable(np.array(rec, dtype=np.float32))
        model = CNN()
        chainer.serializers.load_npz(path.ROOTPATH+"cnn.net",model)
        y = model.forward(x)
        label = np.argmax(y.data[0])
        confidence = abs(y.data[0][0])*100

        return label,confidence
    
    