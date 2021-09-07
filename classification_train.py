import tensorflow as tf
from tensorflow import keras
#from keras import backend as K
import os
import multiprocessing
import tqdm
import pickle
from keras.utils import to_categorical
from keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from keras.applications import vgg16
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Bidirectional, LSTM, TimeDistributed, Dropout, Reshape, Average, Conv1D, MaxPooling1D, concatenate,Conv2D,Activation,Dropout,BatchNormalization,MaxPooling2D,GlobalAveragePooling1D


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
        include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224, 3),
        pooling=None, classes=3)
    for layer in model.layers:
        layer.trainable = False
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
img_height, img_witdh  = (224,224)
batch_size = 32
batch=16
BATCH_SIZE=batch
NUM_CLASSES=3
#tensorboard --logdir C:\Users\Student240914\Desktop\PycharmProjects\keylogger\Praca\log\20210403-140900
os.environ["CUDA_VISIBLE_DEVICES"]="0"
train_data_dir = r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Dataset_1_out\train"
valid_data_dir = r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Dataset_1_out\val"
test_data_dir = r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Dataset_1_out\test"

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                  shear_range=0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,
                                  validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                   target_size= (img_height,img_witdh),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical',
                                                   subset='training')

valid_generator = train_datagen.flow_from_directory(valid_data_dir,
                                                   target_size= (img_height,img_witdh),
                                                   batch_size = batch_size,
                                                   class_mode = 'categorical',
                                                   subset='validation')
test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                   target_size= (img_height,img_witdh),
                                                   batch_size = 1,
                                                   subset='validation')

METRICS = [
  tf.keras.metrics.CategoricalAccuracy(name='acc'),
  CategoricalTruePositives(NUM_CLASSES, BATCH_SIZE),
]
####conda update -n tf --all
vgg=create_model()
x = layers.Dense(3, activation='softmax')(vgg.output)#(x)
#checkpoint_path=r'C:\Dane\Pandora2\result\logs\20210210-062028\best_model.h5'
#model=tf.keras.models.load_model(checkpoint_path,compile=False)
model = Model(inputs=vgg.input, outputs=x)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=METRICS)#
print(model.summary())

earlystopper = EarlyStopping(monitor='acc', patience=7, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.5, patience=5,verbose=1, mode='max', min_lr=0.000001)
logdir = os.path.join(r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\result\logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
checkpointer = ModelCheckpoint(logdir+'/best_model.h5'
                                        ,monitor='acc'
                                        ,verbose=1
                                        ,save_best_only=True
                                        ,save_weights_only=False)
#fit_history=model.fit(np.array(trainImages),np.array(trainY),validation_data=(np.array(testImages),np.array(testY)), epochs=100 ,batch_size=20) #,steps_per_epoch=13000/20,
fit_history=model.fit_generator(generator=train_generator,validation_data=valid_generator,epochs=10,callbacks=[earlystopper,reduce_lr,tensorboard_callback,checkpointer])
model.save(r"E:\Uczelnia\Magister\Semestr1\ZMAP\Praca_semestralna\Datasets\Log\best_model_VHGG19_map_type.h5")