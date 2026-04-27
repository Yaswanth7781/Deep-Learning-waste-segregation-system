
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# PATHS
train_path = "/root/dataset_split/train"
val_path = "/root/dataset_split/val"
test_path = "/root/dataset_split/test"

# DATA
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(128,128), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(128,128), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_path, target_size=(128,128), batch_size=32, class_mode='categorical')

# MODEL
cnn_model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4,activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TRAIN
cnn_model.fit(train_data, epochs=15, validation_data=val_data)

# EVALUATE
cnn_model.evaluate(test_data)

# SAVE
cnn_model.save("cnn_model.h5")