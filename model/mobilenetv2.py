import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# PATHS
train_path = "/root/dataset_split/train"
val_path = "/root/dataset_split/val"
test_path = "/root/dataset_split/test"

# DATA
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(train_path, target_size=(224,224), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(224,224), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_path, target_size=(224,224), batch_size=32, class_mode='categorical')

# BASE MODEL
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

# CUSTOM HEAD
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(4, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

# COMPILE
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN (PHASE 1)
model.fit(train_data, epochs=10, validation_data=val_data)

# FINE-TUNING
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN (PHASE 2)
model.fit(train_data, epochs=10, validation_data=val_data)

# EVALUATE
model.evaluate(test_data)

# SAVE (USE THIS FOR DEPLOYMENT)
model.save("waste_mobilenetv2.h5")