# %%
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

# %%-----------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv('data.csv.gz', compression='gzip')
feature_cols = ['Ax','Ay','Az','Gx','Gy','Gz']
X_raw = df[feature_cols].values
y_raw = df['Label'].values.astype(int)

print(df.shape)
df.head()

# %%-----------------------------
# 2️⃣ Encode labels
# -------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
y_onehot = to_categorical(y_encoded)

# %%-----------------------------
# 3️⃣ Sliding windows
# -------------------------------
def create_windows(X, y, window_size=128, step_size=64):
    X_windows = []
    y_windows = []
    for start in range(0, len(X) - window_size + 1, step_size):
        end = start + window_size
        X_windows.append(X[start:end])
        # Majority vote label in window
        y_windows.append(np.bincount(y_encoded[start:end]).argmax())
    return np.array(X_windows), np.array(y_windows)

window_size = 128
step_size = 64
X_win, y_win = create_windows(X_raw, y_onehot, window_size, step_size)
y_win_onehot = to_categorical(y_win)

# %%-----------------------------
# 4️⃣ Train/test/val split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_win, y_win_onehot, test_size=0.2, random_state=42, stratify=y_win
)

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=np.argmax(y_train, axis=1)
)

# %%-----------------------------
# 5️⃣ Prepare tf.data for GPU
# -------------------------------
batch_size = 512

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# %%-----------------------------
# 6️⃣ Build CNN model
# -------------------------------
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, len(feature_cols))),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.6),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(y_win_onehot.shape[1], activation='softmax')
])

optimizer = RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%-----------------------------
# 7️⃣ Train model
# -------------------------------
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset
)

model.save("har_model.keras")

# %%-----------------------------
# 8️⃣ Evaluate
# -------------------------------
labels = ['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs', 'downstairs']
y_pred_probs = model.predict(test_dataset)          # probabilities
y_pred = np.argmax(y_pred_probs, axis=1)          # predicted labels
y_true = np.argmax(y_test, axis=1)                # true labels
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
cm = cm / cm.sum(axis=1)[:, None]
labels = ['walking', 'standing', 'jogging', 'sitting', 'biking', 'upstairs', 'downstairs']
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# %%-----------------------------
# 9️⃣ Convert to TFLite
# -------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]   # optional: reduces size (quantization)
tflite_model = converter.convert()

with open("har_model.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="har_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(interpreter, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

y_tflite_pred = []
progress = tqdm(total=len(X_test))
for i in range(len(X_test)):
    input_data = X_test[i:i+1].astype(np.float32)
    output_data = predict_tflite(interpreter, input_data)
    y_tflite_pred.append(np.argmax(output_data, axis=1)[0])
    progress.update(1)
progress.close()

y_tflite_pred = np.array(y_tflite_pred)
print(classification_report(y_true, y_tflite_pred, target_names=labels))

cm = confusion_matrix(y_true, y_tflite_pred)
cm = cm / cm.sum(axis=1)[:, None]
sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt=".2f")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('TFLite Model Confusion Matrix')
plt.show()

# %%
