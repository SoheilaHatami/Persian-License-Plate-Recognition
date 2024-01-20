import os
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from collections import Counter
from tensorflow.python.saved_model import tag_constants
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/2.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_string('output', 'outputvideo.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', True, 'disable cv2 window during the process') # this is good for the .ipynb

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_id = 0

    characters = {'G', 'W', 'R', 'M', 'K', '7', '6', '1', 'N', 'P', 'T', 'U', 'F', 'V', 'A', 'D', 'Q', 'B', 'a', 'L', '5', '2', '0', 'O', '3', 'J', '9', '4', 'S', 'Y', 'X', 'Z', 'I', 'E', 'C', '8', 'H'}
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
    counter = 0
    try:
        os.unlink('Detection_result.txt')
    except:
        pass
    while True:
        return_value, frame = vid.read()
        counter += 1
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        # print(bboxes, len(bboxes), type(bboxes))
        with open('Detection_result.txt', 'a') as f:
        #### Added Section
            detection_flg = False
            f.write('###################frame {}#######################\n\n'.format(counter))
            bboxes, detection_flg, image = utils.find_bbox(frame, pred_bbox, prediction_model, decode_batch_predictions, f)
            plt.imshow(image)
            plt.savefig('test.png')
            # plt.show()
            # for box in bboxes:
            #     detection_flg = True
            #     #for i in range(len(pred_bbox[0][0][0])):
            #     print(box)
            #     xmin, ymin = box[0]
            #     xmax, ymax = box[1]
            #     # Convert array to image
            #     #img_crop = Image.fromarray(frame_save)  
            #     f.write('**************************\n')
                
            #     f.write('box: xmin = {}, ymin = {}, xmax = {}, ymax = {} \n'.format(xmin, ymin, xmax, ymax))          
            #     img_crop = np.copy(frame[ymin:ymax, xmin:xmax, :])
            #     # print(img_crop.shape)
            #     # print(frame.shape)
            #     # OCR
            #     # Give cropped frame to OCR

                
            #     # name = str(1) + '.jpg'
            #     # img_save = cv2.imwrite('outputs/1.jpg', img_crop)
            #     # img = cv2.imread(img_save, 0) / 255.
            #     img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY) 
            #     img = cv2.resize(img, (200, 50)).T
            #     img = img / 255.0
            #     final = np.array(img)
            #     final = final[np.newaxis, :, :, np.newaxis]
            #     #final.shape

            #     #  Let's check results on my data
            #     preds = prediction_model.predict(final)
            #     pred_text = decode_batch_predictions(preds)
            #     f.write('OCR detection: \n')
            #     print(pred_text)
            #     for text in pred_text:
            #         f.write(text)
            #         f.write('\n')
                #i += 1
            if not detection_flg:
                print('No Prediction!')
        
        
        # image = utils.draw_bbox(frame, pred_bbox)

       #### Added Section




        #image = utils.draw_bbox(frame, pred_bbox)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not FLAGS.dis_cv2_window:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        if FLAGS.output:
            out.write(result)
        #os.remove('Frame0.jpg')
        # os.remove('outputs/1.jpg')
        # frame_id += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
