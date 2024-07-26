import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import os
# Create ImageDataGenerator for testing data (no data augmentation)

train_dir = '/content/drive/MyDrive/haticemodeller/haticeveriseti/verisetibuzdolabı_train'
test_dir = '/content/drive/MyDrive/haticemodeller/haticeveriseti/verisetibuzdolabı_test'

# Define image size and batch size
img_size = (224, 224)
batch_size = 32

# Create ImageDataGenerator for training data with data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.densenet.preprocess_input
)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load testing data (no augmentation)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_generator.classes),
                                     y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# Load pre-trained DenseNet201 model without top layer
base_model = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of DenseNet201
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
predictions_layer = Dense(20, activation='softmax')(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions_layer)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('/content/drive/MyDrive/haticemodeller/haticedensenet/densenet201_checkpoint.h5',
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             save_freq='epoch')

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

model.save('/content/drive/MyDrive/haticemodeller/haticedensenet/densenet201.h5')

def save_txt(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
save_txt('/content/drive/MyDrive/haticemodeller/haticedensenet/train_loss.txt', history.history['loss'])
save_txt('/content/drive/MyDrive/haticemodeller/haticedensenet/train_accuracy.txt', history.history['accuracy'])
save_txt('/content/drive/MyDrive/haticemodeller/haticedensenet/val_loss.txt', history.history['val_loss'])
save_txt('/content/drive/MyDrive/haticemodeller/haticedensenet/val_accuracy.txt', history.history['val_accuracy'])

