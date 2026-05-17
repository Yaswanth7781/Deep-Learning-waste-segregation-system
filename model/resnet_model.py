import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# PATHS (using the same split paths as CNN.py and mobilenetv2.py)
train_path = "./dataset_split/train"
val_path = "./dataset_split/val"
test_path = "./dataset_split/test"

# DATA
# ResNet models typically expect input in the range [-1, 1] or [0, 1] depending on preprocess_input.
# resnet50.preprocess_input handles this.
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Note: ResNet50 typically expects input size of (224, 224)
train_data = train_gen.flow_from_directory(train_path, target_size=(224,224), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(224,224), batch_size=32, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_path, target_size=(224,224), batch_size=32, class_mode='categorical')

# BASE MODEL - ResNet50
# Load ResNet50 with pre-trained ImageNet weights, excluding the top (classification) layer
base_model_resnet = ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model_resnet.trainable = False # Freeze the base model layers initially

# CUSTOM HEAD - Add a custom classification head for our specific task
x = base_model_resnet.output
x = layers.GlobalAveragePooling2D()(x) # Global Average Pooling to reduce dimensionality
x = layers.BatchNormalization()(x) # Batch normalization for stability
x = layers.Dense(128, activation='relu')(x) # Dense layer with ReLU activation
x = layers.Dropout(0.5)(x) # Dropout for regularization
output = layers.Dense(4, activation='softmax')(x) # Final output layer with 4 classes and softmax activation

model_resnet = models.Model(inputs=base_model_resnet.input, outputs=output)

# COMPILE MODEL (PHASE 1 - Feature Extraction)
# Compile the model with a relatively low learning rate for initial training
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL (PHASE 1 - Feature Extraction)
# Train only the custom head on our dataset
print("\n--- Training ResNet Model (Phase 1: Feature Extraction) ---")
model_resnet.fit(train_data, epochs=10, validation_data=val_data)

# FINE-TUNING (PHASE 2)
# Unfreeze some layers of the base model for fine-tuning
base_model_resnet.trainable = True
# Freeze a portion of the base model to prevent destroying early learned features
for layer in base_model_resnet.layers[:-30]: # Unfreeze the last 30 layers for fine-tuning
    layer.trainable = False

# Recompile the model with a very low learning rate for fine-tuning
model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# TRAIN MODEL (PHASE 2 - Fine-tuning)
# Continue training with fine-tuning
print("\n--- Training ResNet Model (Phase 2: Fine-tuning) ---")
model_resnet.fit(train_data, epochs=10, validation_data=val_data)

# EVALUATE MODEL
print("\n--- Evaluating ResNet Model ---")
loss, accuracy = model_resnet.evaluate(test_data)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# SAVE MODEL
model_resnet.save("waste_resnet50.h5")
print("ResNet50 model saved as waste_resnet50.h5")