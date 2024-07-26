
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Model,Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.optimizers import Adam
# list paths of train and test datasets

TRAIN_PATH = "/content/drive/MyDrive/haticemodeller/haticeveriseti/verisetibuzdolabı_train"
TEST_PATH = "/content/drive/MyDrive/haticemodeller/haticeveriseti/verisetibuzdolabı_test"
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
IMG_SHAPE= (224,224)
## train generator
train_datagen = ImageDataGenerator(rescale=1/255.0,
                                 zoom_range=0.2,
                                 shear_range=0.3,
                                 horizontal_flip=True,
                                 brightness_range=[0.5,1.5])
#test generator
test_datagen = ImageDataGenerator(rescale=1/255.0)

#generate data from train and test directories
train_gen = train_datagen.flow_from_directory(TRAIN_PATH,
                                            target_size=IMG_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            class_mode="binary")

test_gen = test_datagen.flow_from_directory(TEST_PATH,
                                            target_size=IMG_SHAPE,
                                            batch_size=BATCH_SIZE,
                                            class_mode="binary")
#get classes dict
classes_dict = dict(test_gen.class_indices)
#reverse
classes_dict = {v: k for k,v in classes_dict.items()}
#let's plot sone images
images,labels=train_gen.next()
plt.figure(figsize=(20,10))
for i in range(12):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(classes_dict[labels[i]])
#create Inception Model
inception = InceptionV3(weights='imagenet',input_shape=(224, 224, 3),include_top=False)
#show the base model summary
inception.summary()
#show how manay layers in the Resnet Network
layers = inception.layers
print(f'Number of Layers: {len(layers)}')
# number of samples for each set
TRAIN_SIZE = train_gen.samples
TEST_SIZE = test_gen.samples
# early stopping
callbacks = EarlyStopping(patience = 3, monitor='val_acc')

# let's train our Model
inputs = inception.input
# get the output of inception NN and add an average pooling layer
x = inception.output
x = GlobalAveragePooling2D()(x)
# add the a dense layer
x = Dense(256, activation='relu')(x)
# add a dropout
x = Dropout(0.2)(x)
# finally, add an output layer
outputs = Dense(20, activation ='softmax')(x)
# build the model to train
model = Model(inputs=inputs, outputs=outputs)

# freeze all convolutional inception layers
for layer in layers:
    layer.trainable = False

# compile the model
model.compile(optimizer=Adam(lr=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train the model for 10 epochs


# Plot training loss vs validation loss
import matplotlib.pyplot as plt
loss = []
validation_loss = []
accuracy = []
validation_accuracy = []

history = model.fit_generator(
                train_gen,
                epochs= EPOCHS,
                validation_data = test_gen,
                validation_steps = TEST_SIZE//BATCH_SIZE,
                steps_per_epoch = TRAIN_SIZE//BATCH_SIZE,
                callbacks = [callbacks]) # Plot training loss vs validation loss

loss.extend(history.history['loss'])
validation_loss.extend(history.history['val_loss'])
accuracy.extend(history.history['accuracy'])
validation_accuracy.extend(history.history['val_accuracy'])

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

epochs1 = range(1, len(accuracy)+1)
fig2 = plt.figure(figsize=(10,6))
plt.plot(epochs1,accuracy,'r*-',label="Training")
plt.plot(epochs1,validation_accuracy,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0,max(plt.ylim())])
plt.xticks(epochs1)
plt.legend()

def save_txt(file_name='',object=[]):
    file = open(file_name, 'w')
    file.write(str(object))
    file.close()

save_txt('/content/drive/MyDrive/haticemodeller/haticeinception/loss.txt', loss)
save_txt('/content/drive/MyDrive/haticemodeller/haticeinception/val_loss.txt', validation_loss)
save_txt('/content/drive/MyDrive/haticemodeller/haticeinception/accuracy.txt', accuracy)
save_txt('/content/drive/MyDrive/haticemodeller/haticeinception/val_accuracy.txt', validation_accuracy)

from keras.models import load_model
saved_model = model.save("/content/drive/MyDrive/haticemodeller/haticeinception/inceptionmodel.h5")



loss, test_acc = model.evaluate(test_gen)
print("Validation Accuracy = %f \nValidation Loss = %f " % (test_acc, loss))
class_names = list(classes_dict.values())
labels = test_gen.classes
preds =  model.predict(test_gen)
predictions = np.argmax(preds, axis=1)
#show the confusion matrix
conf_matrix = confusion_matrix(labels, predictions)
# plot the confusion matrix
fig,ax = plt.subplots(figsize=(12, 10))
sb.heatmap(conf_matrix, annot=True, linewidths=0.01,cmap="magma",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
ax.set_xticklabels(labels = class_names,fontdict=None)
ax.set_yticklabels(labels = class_names,fontdict=None)
plt.show()
test_images,test_labels=test_gen.next()
plt.figure(figsize=(20,15))



