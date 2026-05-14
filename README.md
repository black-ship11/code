
## 1️⃣ DL: Linear Regression
```python
# ── STEP 1: Import Libraries ──────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

print('TensorFlow version:', tf.__version__)

# ── STEP 2: Load Dataset from CSV ────────────────────────────────
df = pd.read_csv('HousingData.csv')
df.columns = df.columns.str.strip().str.upper()

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('\nMissing values per column:')
print(df.isnull().sum())
df.head()

# ── STEP 3: Fix Missing Values (NaN) ── KEY FIX ──────────────────
# NaN values were causing the model to get stuck at loss=72
# Solution: replace each NaN with the mean of that column
df.fillna(df.mean(numeric_only=True), inplace=True)

print('Missing values AFTER fix:')
print(df.isnull().sum())   # all must be 0
print('\nMEDV stats:')
print(df['MEDV'].describe())

# ── STEP 4: Separate Features and Target ─────────────────────────
X = df.drop(columns=['MEDV']).values
y = df['MEDV'].values

print('X shape:', X.shape)
print('y shape:', y.shape)
print('y sample:', y[:5])
# ── STEP 5: Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('Train:', X_train.shape, '| Test:', X_test.shape)
# ── STEP 6: Scale Features ────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print('Scaling done.')
# ── STEP 7: Build Deep Neural Network ────────────────────────────
tf.random.set_seed(42)

model = Sequential([
    Input(shape=(13,)),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1)                        # linear output — no activation
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
model.summary()
# ── STEP 8: Train ─────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

print(f'First 5 losses : {[round(x,2) for x in history.history["loss"][:5]]}')
print(f'Last  5 losses : {[round(x,2) for x in history.history["loss"][-5:]]}')
print(f'\nFinal Train MSE : {history.history["loss"][-1]:.4f}')
print(f'Final Val   MSE : {history.history["val_loss"][-1]:.4f}')
# ── STEP 9: Evaluate on Test Set ─────────────────────────────────
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f'Test MSE  : {loss:.4f}')
print(f'Test MAE  : {mae:.4f}')
print(f'Test RMSE : {np.sqrt(loss):.4f}')
# ── STEP 10: Predictions ──────────────────────────────────────────
predictions = model.predict(X_test, verbose=0).flatten()

print('First 10 Predictions vs Actual:')
print(f'{"Predicted ($k)":>16}  {"Actual ($k)":>12}  {"Error":>8}')
print('-' * 42)
for i in range(10):
    err = predictions[i] - y_test[i]
    print(f'  {predictions[i]:>12.1f}      {y_test[i]:>8.1f}   {err:>+7.1f}')
  # ── STEP 11: Plots ────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss (MSE) over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, predictions, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', label='Perfect Prediction')
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.title('Actual vs Predicted House Prices')
plt.legend()

plt.tight_layout()
plt.show()
````

---

## 2️⃣ DL: Multiclass Classification using Deep Neural Network

```python
# Dataset: UCI Letter Recognition Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop


# ------------------------------------------------------
# Step 1: Load Dataset
# ------------------------------------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"

columns = [
    "letter", "x-box", "y-box", "width", "high", "onpix",
    "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
    "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"
]

data = pd.read_csv(url, names=columns)

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nClass Labels:")
print(data["letter"].unique())


# ------------------------------------------------------
# Step 2: Separate Input and Output
# ------------------------------------------------------

X = data.drop("letter", axis=1)
y = data["letter"]


# ------------------------------------------------------
# Step 3: Encode Target Labels
# ------------------------------------------------------
# A-Z letters are converted into numbers 0-25

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert labels into one-hot encoding
# Example: A -> [1,0,0,...], B -> [0,1,0,...]
y_categorical = to_categorical(y_encoded)


# ------------------------------------------------------
# Step 4: Feature Scaling
# ------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ------------------------------------------------------
# Step 5: Split Dataset
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------
# Step 6: Build Deep Neural Network Model
# ------------------------------------------------------

model = Sequential()

# Input layer + first hidden layer
model.add(Dense(128, activation="relu", input_shape=(16,)))

# Dropout layer to reduce overfitting
model.add(Dropout(0.3))

# Second hidden layer
model.add(Dense(64, activation="relu"))

# Dropout layer
model.add(Dropout(0.2))

# Third hidden layer
model.add(Dense(32, activation="relu"))

# Output layer
# 26 neurons because there are 26 classes A-Z
model.add(Dense(26, activation="softmax"))


# ------------------------------------------------------
# Step 7: Compile Model
# ------------------------------------------------------

model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()


# ------------------------------------------------------
# Step 8: Train Model
# ------------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


# ------------------------------------------------------
# Step 9: Evaluate Model
# ------------------------------------------------------

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ------------------------------------------------------
# Step 10: Predictions
# ------------------------------------------------------

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nAccuracy Score:", accuracy_score(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=label_encoder.classes_
))


# ------------------------------------------------------
# Step 11: Confusion Matrix
# ------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("Confusion Matrix - Letter Recognition")
plt.xlabel("Predicted Letter")
plt.ylabel("Actual Letter")
plt.show()


# ------------------------------------------------------
# Step 12: Accuracy Graph
# ------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ------------------------------------------------------
# Step 13: Loss Graph
# ------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# ------------------------------------------------------
# Step 14: Test Single Sample Prediction
# ------------------------------------------------------

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)
predicted_class_index = np.argmax(prediction)
predicted_letter = label_encoder.inverse_transform([predicted_class_index])

actual_class_index = np.argmax(y_test[0])
actual_letter = label_encoder.inverse_transform([actual_class_index])

print("\nSingle Sample Prediction")
print("Actual Letter:", actual_letter[0])
print("Predicted Letter:", predicted_letter[0])

```

---

## 3️⃣ DL: Binary Classification using Deep Neural Network

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Load IMDB Dataset
# num_words = top 10,000 frequently used words
vocab_size = 10000

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=vocab_size
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# Pad sequences to same length
max_length = 200

X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding='post'
)

X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding='post'
)

print("Shape after Padding:")
print(X_train.shape)


# Build Deep Neural Network Model
model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=32,
              input_length=max_length),

    Flatten(),

    Dense(64, activation='relu'),
    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

# Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Model Summary
print("\nModel Summary:")
model.summary()


# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

# Evaluate Model
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:", accuracy * 100)


# Predict Sentiment
predictions = model.predict(X_test)

# Convert probability into Positive/Negative
predicted_labels = (
    predictions > 0.5
).astype("int32")

# Show Sample Predictions
print("\nSample Predictions:")

for i in range(10):
    actual = (
        "Positive"
        if y_test[i] == 1
        else "Negative"
    )

    predicted = (
        "Positive"
        if predicted_labels[i] == 1
        else "Negative"
    )

    print(
        f"Review {i+1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )


 # Plot Accuracy Graph
plt.figure(figsize=(8,5))
plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)
plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()


# Plot Loss Graph
plt.figure(figsize=(8,5))
plt.plot(
    history.history['loss'],
    label='Training Loss'
)
plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()


# Predict Custom Review
word_index = tf.keras.datasets.imdb.get_word_index()

reverse_word_index = {
    value: key
    for (key, value)
    in word_index.items()
}

def decode_review(text):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?')
         for i in text]
    )

print("\nExample Review:")
print(decode_review(X_test[0]))

print("\nPredicted Sentiment:")
if predicted_labels[0] == 1:
    print("Positive Review")
else:
    print("Negative Review")
```

---

## 4️⃣ DL: CNN using Fashion MNIST Dataset

```python
#dataset : https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases


# =========================================================
# RICE LEAF DISEASE DETECTION USING CNN
# Works for BOTH Google Colab and Jupyter Notebook
# =========================================================

# =========================================================
# STEP 1 — MOUNT GOOGLE DRIVE (COLAB ONLY)
# =========================================================

# Uncomment these 2 lines ONLY if using Google Colab

# from google.colab import drive
# drive.mount('/content/drive')


# =========================================================
# STEP 2 — IMPORT LIBRARIES
# =========================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image


# =========================================================
# STEP 3 — DATASET PATH
# =========================================================

# ---------------- GOOGLE COLAB PATH ----------------
# Uncomment if using Google Colab

# dataset_path = "/content/drive/MyDrive/rice_leaf_diseases"


# ---------------- JUPYTER NOTEBOOK PATH ----------------
# Uncomment if using Jupyter Notebook

dataset_path = r"C:\Users\YourName\Desktop\rice_leaf_diseases"


# =========================================================
# STEP 4 — IMAGE PREPROCESSING
# =========================================================

img_size = (224, 224)
batch_size = 16

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# =========================================================
# STEP 5 — BUILD CNN MODEL
# =========================================================

model = models.Sequential()

# First Convolution Layer
model.add(layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(224,224,3)
))

model.add(layers.MaxPooling2D((2,2)))

# Second Convolution Layer
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(layers.MaxPooling2D((2,2)))

# Third Convolution Layer
model.add(layers.Conv2D(
    128,
    (3,3),
    activation='relu'
))

model.add(layers.MaxPooling2D((2,2)))

# Flatten Layer
model.add(layers.Flatten())

# Dense Layer
model.add(layers.Dense(
    128,
    activation='relu'
))

# Dropout Layer
model.add(layers.Dropout(0.5))

# Output Layer (3 classes)
model.add(layers.Dense(
    3,
    activation='softmax'
))


# =========================================================
# STEP 6 — COMPILE MODEL
# =========================================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# =========================================================
# STEP 7 — MODEL SUMMARY
# =========================================================

model.summary()


# =========================================================
# STEP 8 — TRAIN MODEL
# =========================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)


# =========================================================
# STEP 9 — EVALUATE MODEL
# =========================================================

loss, accuracy = model.evaluate(val_data)

print("\nValidation Accuracy:", accuracy)


# =========================================================
# STEP 10 — PLOT ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(8,5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()


# =========================================================
# STEP 11 — SAVE MODEL
# =========================================================

# -------- GOOGLE COLAB SAVE PATH --------
# Uncomment for Colab

# model.save("/content/drive/MyDrive/rice_disease_model.h5")


# -------- JUPYTER NOTEBOOK SAVE PATH --------
# Uncomment for Jupyter Notebook

model.save("rice_disease_model.h5")

print("\nModel Saved Successfully!")


# =========================================================
# STEP 12 — PREDICT SINGLE IMAGE
# =========================================================

# Give test image path here

test_image_path = r"C:\Users\YourName\Desktop\test.jpg"

# Example for Colab:
# test_image_path = "/content/test.jpg"

img = image.load_img(
    test_image_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = img_array / 255.0

prediction = model.predict(img_array)

class_names = list(train_data.class_indices.keys())

predicted_class = class_names[np.argmax(prediction)]

print("\nPredicted Disease:", predicted_class)


# =========================================================
# DATASET FOLDER STRUCTURE
# =========================================================

"""
rice_leaf_diseases/
│
├── Bacterial leaf blight/
├── Brown spot/
└── Leaf smut/
"""

# =========================================================
# INSTALLATION COMMANDS (IF NEEDED)
# =========================================================

"""
pip install tensorflow
pip install matplotlib
```

---

## 5️⃣ DL: CNN using Fashion MNIST Dataset

```python
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# STEP 1: Load Fashion MNIST Dataset
# ============================================================

(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()

print("Dataset Loaded Successfully")
print("Training Images Shape:", X_train.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Images Shape:", X_test.shape)
print("Testing Labels Shape:", y_test.shape)


# ============================================================
# STEP 2: Class Names
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# STEP 3: Normalize Pixel Values
# ============================================================
# Pixel values are from 0 to 255.
# Convert them into range 0 to 1.

X_train = X_train / 255.0
X_test = X_test / 255.0


# ============================================================
# STEP 4: Reshape Data for CNN
# ============================================================
# CNN requires input shape: height, width, channels.
# Fashion MNIST images are grayscale, so channel = 1.

X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

print("\nAfter Reshaping:")
print("Training Images Shape:", X_train.shape)
print("Testing Images Shape:", X_test.shape)


# ============================================================
# STEP 5: Display Sample Images
# ============================================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.suptitle("Sample Images from Fashion MNIST Dataset")
plt.show()


# ============================================================
# STEP 6: Build CNN Model
# ============================================================

model = models.Sequential()

# First Convolution + Pooling Layer
model.add(layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    input_shape=(28, 28, 1)
))
model.add(layers.MaxPooling2D((2, 2)))

# Second Convolution + Pooling Layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolution Layer
model.add(layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu"
))

# Flatten Layer
model.add(layers.Flatten())

# Fully Connected Dense Layer
model.add(layers.Dense(64, activation="relu"))

# Dropout Layer to reduce overfitting
model.add(layers.Dropout(0.3))

# Output Layer
# 10 neurons because Fashion MNIST has 10 classes
model.add(layers.Dense(10, activation="softmax"))


# ============================================================
# STEP 7: Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n========== MODEL SUMMARY ==========")
model.summary()


# ============================================================
# STEP 8: Train Model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)


# ============================================================
# STEP 9: Evaluate Model
# ============================================================

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n========== MODEL EVALUATION ==========")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# STEP 10: Make Predictions
# ============================================================

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

print("\nFirst 10 Predictions:")
for i in range(10):
    print(
        "Image", i + 1,
        "| Actual:", class_names[y_test[i]],
        "| Predicted:", class_names[predicted_labels[i]]
    )


# ============================================================
# STEP 11: Plot Accuracy Graph
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training Accuracy vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# STEP 12: Plot Loss Graph
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# STEP 13: Show Prediction Images
# ============================================================

plt.figure(figsize=(12, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")

    actual = class_names[y_test[i]]
    predicted = class_names[predicted_labels[i]]

    plt.title(f"A: {actual}\nP: {predicted}")
    plt.axis("off")

plt.suptitle("Actual vs Predicted Fashion Categories")
plt.show()


# ============================================================
# STEP 14: Save Model
# ============================================================

model.save("fashion_mnist_cnn_model.h5")

print("\nModel saved successfully as fashion_mnist_cnn_model.h5")
```

---

## 6️⃣ DL: N-Queens (Backtracking)

```python
import pandas as pd 
import numpy as np 

df1 = pd.read_csv("Google_Stock_Price_Train.csv")
df2 = pd.read_csv("Google_Stock_Price_Test.csv")

df1.info()

df1['Close'] = df1['Close'].astype(str).str.replace(",","").astype(float)
df2['Close'] = df2['Close'].astype(str).str.replace(",","").astype(float)

from sklearn.preprocessing import MinMaxScaler
train_scaler = MinMaxScaler()
df1['Normalized close'] = train_scaler.fit_transform(df1['Close'].values.reshape(-1,1))


test_scaler = MinMaxScaler()
df2['Normalized close'] = test_scaler.fit_transform(df2['Close'].values.reshape(-1,1))


x_train = df1['Normalized close'].values[:-1].reshape(-1,1,1)
y_train = df1['Normalized close'].values[1:].reshape(-1,1,1)

x_test = df2['Normalized close'].values[:-1].reshape(-1,1,1)
y_test = df2['Normalized close'].values[1:].reshape(-1,1,1)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(4, input_shape = (1,1)))
model.add(Dense(1))

model.compile(optimizer = 'adam', loss = 'mse')

model.summary()



model.fit(x_train,y_train, validation_data= (x_test,y_test), epochs = 100, batch_size= 1)


test_loss = model.evaluate(x_test,y_test)
print('Testing loss: ', test_loss)


pred = model.predict(x_test)


y_test_actual = test_scaler.inverse_transform(y_test.reshape(-1,1))
y_test_pred = test_scaler.inverse_transform(pred.reshape(-1,1))

index = 1
print("Actual: ", y_test_actual[index])
print("Predicted: ", y_test_pred[index])
```

---
\
