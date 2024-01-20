
# OCR
# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from timeit import default_timer as timer

start = timer()
characters = {'T', 'N', 'O', '6', 'E', 'W', '8', '5', 'D', 'P', 'L', 'G', 'S', 'M', 'K', 'I', 'J', 'Q', 'U', '2', 'R', 'Y', 'A', 'X', '3', '4', 'C', 'H', 'B', '9', '0', 'Z', '7', 'V', 'a', 'F', '1'}
# Desired image dimensions
img_height = 50
img_width = 200
# Maximum length of any image label in the dataset
max_length = 8

# Mapping characters to integers
charlist    = list(characters)
charlist.sort()
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=charlist, num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

# Mapping characters to integers
charlist    = list(characters)
charlist.sort()
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=charlist, num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomRotation(factor=0.05),
        layers.experimental.preprocessing.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
        #layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        #layers.experimental.preprocessing.Resizing(),
        #layers.experimental.preprocessing.Rescaling(1./255)

    ],
    name="data_augmentation",
)

def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")
    augmented = data_augmentation(input_img)
    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(augmented)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    #opt = keras.optimizers.Adam(learning_rate = 0.01)
    opt = keras.optimizers.Adam(learning_rate = 0.001)
    #opt = keras.optimizers.Adam(learning_rate = 0.0001)
    #opt = keras.optimizers.Adamax(learning_rate = 0.001)
    #opt = keras.optimizers.SGD(learning_rate = 0.01)
    #opt = keras.optimizers.SGD(learning_rate = 0.1)
    # Compile the model and return
    model.compile(optimizer=opt)
    return model

# Get the model
model = build_model()
model.summary()
model.load_weights('my_model_weights.h5')


# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
#prediction_model.summary()

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf_8")
        output_text.append(res)
    return output_text

# %%
start = timer()
# from tensorflow.keras.models import load_model
image_path = 'outputs_1/2962_0.jpg'
img = cv2.imread(image_path, 0) / 255.
img = cv2.resize(img, (200, 50)).T
final = np.array([img])
final = np.expand_dims(final, -1)
final.shape

#  Let's check results on my data
preds = prediction_model.predict(final)
pred_texts = decode_batch_predictions(preds)
    
_, ax = plt.subplots(1, 1, figsize=(15, 5))
for i in range(len(pred_texts)):
    img = (final[i, :, :, 0] * 255).astype(np.uint8)
    img = img.T
    title = f"{pred_texts[i]}"
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.show()
end = timer()
print(end - start)
# %%
