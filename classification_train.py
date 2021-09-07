from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from keras.callbacks import CSVLogger
from datetime import datetime
from matplotlib import pyplot as plt
import requests
from keras import backend as K
import os
import multiprocessing
import tqdm
import pickle
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.models import Model
from keras.applications import vgg16
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, Reshape, Average, Conv1D, MaxPooling1D, concatenate,Conv2D,Activation,Dropout,BatchNormalization,MaxPooling2D,GlobalAveragePooling1D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import layers, models, optimizers,Sequential
import datetime


class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     

        y_true = K.argmax(y_true, axis=-1)
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.flatten(y_true)

        true_poss = K.sum(K.cast((K.equal(y_true, y_pred)), dtype=tf.float32))

        self.cat_true_positives.assign_add(true_poss)

    def result(self):

        return self.cat_true_positives

def create_model():
    model=tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=(608, 416, 3),
        pooling=None, classes=20, classifier_activation='sigmoid')
    x = model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(4096, activation='relu')(x)
    #x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    #x = layers.GlobalAveragePooling2D()(x)
    model = models.Model(inputs=model.input, outputs=x)
    return model
###datagen.flow_from_directory
batch=10
BATCH_SIZE=batch
NUM_CLASSES=20
#tensorboard --logdir C:\Users\Student240914\Desktop\PycharmProjects\keylogger\Praca\log\20210403-140900
os.environ["CUDA_VISIBLE_DEVICES"]="0"
dataframe=pd.read_csv(r'C:\Users\PatrykB\Desktop\Dane\indeksacja\klasyfikacja_dane_wszystkie.csv',sep=',')
train,test=train_test_split(dataframe,random_state=24)
gen= ImageDataGenerator(rescale=1./255,horizontal_flip=True, featurewise_center=True)
traingen=gen.flow_from_dataframe(train, directory=None, x_col='filename', y_col='class', target_size=(608, 416), color_mode='rgb',classes=["dok_in","otop", "nie dotyczy", "kbud","obl","dz","mapa","mpow","mw","ppodz","prgr","dz_gps","spis","spr","styt","szk","wsp","zmew","zmgr","zwr"], class_mode='categorical', batch_size=batch, shuffle=True,validate_filenames=True)
testgen=gen.flow_from_dataframe(test, directory=None, x_col='filename', y_col='class', target_size=(608, 416), color_mode='rgb',classes=["dok_in","otop", "nie dotyczy", "kbud","obl","dz","mapa","mpow","mw","ppodz","prgr","dz_gps","spis","spr","styt","szk","wsp","zmew","zmgr","zwr"], class_mode='categorical', batch_size=batch, shuffle=True,validate_filenames=True)

METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  CategoricalTruePositives(NUM_CLASSES, BATCH_SIZE),
]
####conda update -n tf --all
vgg=create_model()
x = layers.Dense(20, activation='sigmoid')(vgg.output)#(x)
#checkpoint_path=r'C:\Dane\Pandora2\result\logs\20210210-062028\best_model.h5'
#model=tf.keras.models.load_model(checkpoint_path,compile=False)
model = Model(inputs=vgg.input, outputs=x)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=METRICS)#
print(model.summary())

earlystopper = EarlyStopping(monitor='acc', patience=7, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,verbose=1, mode='max', min_lr=0.000001)
logdir = os.path.join("result/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpointer = ModelCheckpoint(logdir+'/best_model.h5'
                                        ,monitor='acc'
                                        ,verbose=1
                                        ,save_best_only=True
                                        ,save_weights_only=False)
#fit_history=model.fit(np.array(trainImages),np.array(trainY),validation_data=(np.array(testImages),np.array(testY)), epochs=100 ,batch_size=20) #,steps_per_epoch=13000/20,
fit_history=model.fit_generator(generator=traingen,validation_data=testgen,validation_steps=test.shape[0]/batch,epochs=100,callbacks=[earlystopper,reduce_lr,tensorboard_callback,checkpointer])
