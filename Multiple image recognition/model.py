# Convert XML to TFRecord (For Pascal VOC annotations)

import tensorflow as tf
import xml.etree.ElementTree as ET
import os
import numpy as np
from object_detection.utils import dataset_util

def create_tf_example(xml_file, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract image info
    img_path = os.path.join(img_dir, root.find('filename').text)
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_image_data = fid.read()

    # Extract bounding boxes and class labels
    object_list = []
    for obj in root.iter('object'):
        xmin = float(obj.find('bndbox/xmin').text)
        ymin = float(obj.find('bndbox/ymin').text)
        xmax = float(obj.find('bndbox/xmax').text)
        ymax = float(obj.find('bndbox/ymax').text)
        label = obj.find('name').text
        object_list.append([xmin, ymin, xmax, ymax, label])

    # Prepare tf.Example
    tf_example = dataset_util.create_example(
        image_path=img_path,
        image_data=encoded_image_data,
        boxes=object_list,
    )
    
    return tf_example

def convert_to_tfrecord(annotation_dir, img_dir, output_file):
    writer = tf.io.TFRecordWriter(output_file)
    for xml_file in os.listdir(annotation_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotation_dir, xml_file)
            tf_example = create_tf_example(xml_path, img_dir)
            writer.write(tf_example.SerializeToString())
    writer.close()



# Faster R-CNN Model Training Setup
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.models import faster_rcnn_resnet50_fpn

# Path to the config file for Faster R-CNN
pipeline_config_path = 'models/faster_rcnn_resnet50_fpn_640x640_coco17_tpu-8.config'

# Load pipeline config file
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

# Modify the pipeline config for custom dataset
pipeline_config.model.faster_rcnn.num_classes = 80  # Change to your number of classes
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = 'models/faster_rcnn_resnet50_fpn_640x640_coco17_tpu-8/checkpoint'  # Path to pretrained checkpoint

# Save the modified config to a new file
config_util.create_configs_from_configs(pipeline_config, 'models/')


# training model
import tensorflow as tf
from object_detection.utils import model_util
from object_detection import model_lib_v2

def train_model():
    pipeline_config = 'models/faster_rcnn_resnet50_fpn_640x640_coco17_tpu-8.config'
    model_dir = 'output_model_dir'
    
    # Train the model
    model_lib_v2.train_loop(
        pipeline_config_path=pipeline_config,
        model_dir=model_dir,
        num_train_steps=20000,
        sample_1_of_n_eval_examples=1,
        use_tpu=False
    )

train_model()



# Evaluate the Model
def evaluate_model():
    pipeline_config = 'models/faster_rcnn_resnet50_fpn_640x640_coco17_tpu-8.config'
    model_dir = 'output_model_dir'
    
    # Evaluate the model
    model_lib_v2.eval_loop(
        pipeline_config_path=pipeline_config,
        model_dir=model_dir,
        checkpoint_dir='output_model_dir/checkpoint',
        eval_dir='output_model_dir/eval'
    )

evaluate_model()


# Inference on New Images
import cv2
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

# Load the trained model
model = tf.saved_model.load('output_model_dir/saved_model')

# Load label map (create this file based on your dataset)
category_index = label_map_util.create_category_index_from_labelmap('path/to/label_map.pbtxt', use_display_name=True)

def run_inference_for_single_image(model, image_np):
    # Actual detection.
    image = np.asarray(image_np)
    model_fn = model.signatures['serving_default']
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]
    output_dict = model_fn(input_tensor)

    return output_dict

def show_inference(model, image_path):
    image_np = cv2.imread(image_path)
    output_dict = run_inference_for_single_image(model, image_np)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'][0].numpy(),
        output_dict['detection_classes'][0].numpy().astype(np.int32),
        output_dict['detection_scores'][0].numpy(),
        category_index,
        instance_masks=output_dict.get('detection_masks', None),
        use_normalized_coordinates=True,
        line_thickness=8
    )
    
    # Display the result
    cv2.imshow('Inference', image_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run inference
show_inference(model, 'path_to_image.jpg')
