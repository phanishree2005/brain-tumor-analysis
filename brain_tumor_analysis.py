from google.colab import drive
drive.mount('/content/drive')
import zipfile

zip_path = "/content/drive/MyDrive/Brain_Tumor_Project/archive.zip"
extract_path = "/content/brain_tumor_dataset"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("✅ Dataset extracted successfully!")
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
dataset_path = "/content/brain_tumor_dataset"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
model.save("/content/brain_tumor_model.h5")
print("✅ Model saved successfully!")
from google.colab import files
files.download("/content/brain_tumor_model.h5")
from tensorflow.keras.preprocessing import image
import numpy as np # Import numpy if not already imported
import os # Import os to list directory contents

# List files in the 'yes' directory to find a valid image name
# You might need to adjust the path based on your extracted dataset structure
image_files = os.listdir("/content/brain_tumor_dataset/yes/")
print("Available image files in /content/brain_tumor_dataset/yes/:")
for file in image_files:
    print(file)

# Update img_path with an actual image file name from the list above
# Replace 'Y###.jpg' with a real filename, for example, 'Y1.jpg' or similar
img_path = "/content/brain_tumor_dataset/yes/Y1.jpg"  # Example: Replace 'Y1.jpg' with an actual file name

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

prediction = model.predict(img_tensor)
# Assuming your model is trained to predict 1 for 'yes' (tumor) and 0 for 'no' (no tumor)
# Based on your flow_from_directory class_mode='binary' and subset='training' which typically assigns 0/1 based on directory name alphabetical order.
# Check the train_generator.class_indices attribute to confirm the mapping if needed.
if prediction[0][0] > 0.5:
    print("✅ Tumor Detected") # If prediction is closer to 1
else:
    print("❌ No Tumor Detected") # If prediction is closer to 0
  model.save("/content/brain_tumor_model.h5")
print("✅ Model saved successfully!")
rom google.colab import files
files.download("/content/brain_tumor_model.h5")
