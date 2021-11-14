import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model
import numpy as np
from image_preprocessor import ImagePreprocessor
from early_stopping import EarlyStopping

np.set_printoptions(suppress=True)

vehicle_types = {
    0 : 'car',
    1 : 'plane',
    2 : 'boat',
    3 : 'motorcycle'
}
pix = 64

es = EarlyStopping(depth=5, ignore=20, method='consistency')

ip = ImagePreprocessor(normalization=255, training_threshold=0.7, color_mode='L')
package = ip.preprocess_dirs(
    ['images/boat', 'images/car', 'images/motorcycle', 'images/plane'],
    [2, 0, 3, 1],
    True
)

train_features = package['TRAIN_IMAGES']
train_labels = package['TRAIN_LABELS']
test_features = package['TEST_IMAGES']
test_labels = package['TEST_LABELS']

train_ds = tf.data.Dataset.from_tensors((train_features, train_labels)).shuffle(10000)
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels)).shuffle(10000)

class VehiclePredictor(Model):
    def __init__(self):
        super(VehiclePredictor, self).__init__()
        self.conv1 = Conv2D(16, (3,3), activation='relu', input_shape=(pix,pix,1))
        self.mp1 = MaxPool2D()
        self.conv2 = Conv2D(32, (3,3), activation='relu')
        self.mp2 = MaxPool2D()
        self.conv3 = Conv2D(64, (3,3), activation='relu')
        self.mp3 = MaxPool2D()
        self.dropout1 = Dropout(0.5)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(4)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = VehiclePredictor()

loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=0.001)

train_loss = keras.metrics.Mean()
train_accuracy = keras.metrics.SparseCategoricalAccuracy()

test_loss = keras.metrics.Mean()
test_accuracy = keras.metrics.SparseCategoricalAccuracy()

@tf.function
def train_step(features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_function(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(features, labels):
    predictions = model(features, training=False)
    t_loss = loss_function(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(200):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for features, label in train_ds:
        train_step(features, label)

    for features, label in test_ds:
        test_step(features, label)
    
    if es.check(test_loss.result()):
        print('BREAKING LOOP')
        break
    
    print(
        f'Epoch {epoch+1} || '
        f'Training Loss: {train_loss.result()}, '
        f'Training Accuracy: {train_accuracy.result()}, '
        f'Testing Loss: {test_loss.result()}, '
        f'Testing Accuracy: {test_accuracy.result()}'
    )

def format_prediction(image):
    pred = model.predict(np.array(ip.file_to_array(image)))
    return pred, vehicle_types[np.argmax(pred[0])]
print('---------------EXTERNAL TESTING PREDICTIONS---------------')
print(f'Car: {format_prediction("images/external test/car1.jpg")[1]}')
print(f'Boat: {format_prediction("images/external test/boat1.jpg")[1]}')
print(f'Plane: {format_prediction("images/external test/plane1.jpg")[1]}')
print(f'Motorcycle: {format_prediction("images/external test/motorcycle1.jpg")[1]}')