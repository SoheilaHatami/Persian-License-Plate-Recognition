#!python save_model.py --weights ./data/config_last.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny



# %%
import tensorflow as tf
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from PIL import Image


def main(image_path, weights_path):
    input_size = 416

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if True:
        saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
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
        iou_threshold=0.45,
        score_threshold=0.25
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    #image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    #cv2.imwrite('277.jpg', image)
    out = pred_bbox[0][0][0]
    return out

#main('out_0/277.jpg', 'checkpoints/yolov4-tiny-416')


# %%
import os
from PIL import Image
import cv2
import numpy as np
from PIL import Image

os.makedirs('outputs_3/', exist_ok = True)
src_path = 'out_0/'

for img in os.listdir(src_path):
  #print(img)
  img_path = src_path + img
  #print(img_path)
  bbox = main(img_path, 'checkpoints/yolov4-tiny-416')
  if bbox.any() == True:
      im = Image.open(r'out_0/' + str(img))
      left = bbox[1]
      top = bbox[0]
      right = bbox[3]
      bottom = bbox[2]
      img_crop = im.crop((left, top, right, bottom))
      new_name = img.split('.')
      new_name = new_name[0] + '_' + str(0) + '.' + new_name[1]
      img_crop.save(os.path.join('outputs_3/', os.path.split(new_name)[-1]))
  else:
       print('No prediction for ' + img + ' image.')

# %%


# For Video

# %%
#!python detectvideo.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --video ./data/2.mp4 --tiny
