import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models


image_fp = './images 2'  # updated with your real folder name
image_paths = glob.glob(os.path.join(image_fp, '*.jpg'))
print(f"Total images found: {len(image_paths)}")

image_names = [os.path.basename(path) for path in image_paths]
labels = [' '.join(name.split('_')[:-1]) for name in image_names]

unique_labels = sorted(list(set(labels)))
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
encoded_labels = [label_to_index[label] for label in labels]

data = pd.DataFrame({
    'filepath': image_paths,
    'label': labels,
    'encoded_label': encoded_labels
})
print(data.head())

train_paths, test_paths, train_labels, test_labels = train_test_split(
    data['filepath'], data['encoded_label'], test_size=0.2, random_state=42, stratify=data['encoded_label'])

print(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")

IMG_SIZE = 128  

def process_image(filepath):
    img = mpimg.imread(filepath)
    
    if img.ndim == 2:  
        img = np.stack((img,)*3, axis=-1)
    elif img.shape[-1] == 4:  
        img = img[:, :, :3]
        
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # 
    return img

X_train = np.array([process_image(fp) for fp in train_paths])
X_test = np.array([process_image(fp) for fp in test_paths])
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(unique_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='x')
plt.legend()
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

model.save('pet_faces_classifier.h5')
print("Model saved successfully as pet_faces_classifier.h5")

def show_predictions(model, X_test, y_test, index_to_label):
    idxs = np.random.choice(len(X_test), 5)
    for idx in idxs:
        img = X_test[idx]
        true_label = index_to_label[y_test[idx]]
        pred_label = index_to_label[np.argmax(model.predict(img[np.newaxis, ...]))]
        plt.imshow(img)
        plt.title(f"True: {true_label}, Predicted: {pred_label}")
        plt.axis('off')
        plt.show()

show_predictions(model, X_test, y_test, index_to_label)
m = model.evaluate(X_test, y_test)
print(m)