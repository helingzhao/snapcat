import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util

UPLOAD_FOLDER = './lux-photo'
ALLOWED_EXTENSIONS = set(['jpg', 'png'])
PATH_TO_CHECKPOINT = './ssd_inception_v2_coco_11_06_2017/frozen_inference_graph.pb'
PATH_TO_LABELS = './labels/mscoco_label_map.pbtxt'
MAX_NUM_CLASSES = 90
ANNOTATED_IMAGE_SIZE = (12, 8)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST'])
def snap_cat():
    filename = upload_file()
    if filename is None:
        print("File upload failed! :(")
        return "Upload failed"
    contains_cat = find_cat(filename)
    print("***********************")
    print("Contains cat: " + str(contains_cat))
    print("***********************")
    if contains_cat:
        tweet_with_caption(filename)
        return "Success!"
    return "Cat"


def upload_file():
    print("Executing upload_file")
    if request.method == 'POST':
        print("Method is POST...")
        if 'file' not in request.files:
            print("Could not find file in request...")
        file = request.files['file']
        if file.filename == '':
            print("The file name is empty!")
        if file and allowed_file(file.filename):
            print("The file exists and has a valid filename")
            filename = secure_filename(file.filename)
            print("The filename is " + filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return app.config['UPLOAD_FOLDER'] + '/' + filename


def find_cat(filename):
    image = load_image(filename)
    image = np.expand_dims(image, axis=0)
    (boxes, scores, classes) = session.run(
        [boxes_tensor, scores_tensor, classes_tensor],
        feed_dict={image_tensor: image})
    categories = [category_index[int(classes[0, i])]['name']
                  for i in range(len(classes[0]))]

    record_result(filename.replace('.jpg', '_out.txt'), categories, np.squeeze(scores))
    annotate_image(filename.replace('.jpg', '_out.jpg'), np.squeeze(image), np.squeeze(boxes), np.squeeze(scores),
                   np.squeeze(classes).astype(np.int32))

    top5_categories = categories[:5]
    twopercent_categories, _ = zip(*filter(lambda x: x[1] > 0.02, zip(categories, np.squeeze(scores))))
    return 'cat' in top5_categories and 'person' not in twopercent_categories


def tweet_with_caption(filename):
    pass


def load_image(filename):
    image = Image.open(filename)
    (image_width, image_height) = image.size
    return np.array(image.getdata()).reshape((image_height, image_width, 3)).astype(np.uint8)


def load_graph(path_to_checkpoint):
    # load graph into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_category_index(path_to_labels):
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, MAX_NUM_CLASSES)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def record_result(filename, categories, scores):
    with open(filename, 'w') as handle:
        for category, score in zip(categories, scores):
            handle.write('%s (%0.2f)\n' % (category, score))


def annotate_image(filename, image, boxes, scores, classes):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8
    )
    plt.figure(figsize=ANNOTATED_IMAGE_SIZE)
    plt.imshow(image)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()
    plt.close()


@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'


@app.route('/test', methods=['POST'])
def test(request):
    return 'Online!'


if __name__ == '__main__':
    detection_graph = load_graph(PATH_TO_CHECKPOINT)
    category_index = load_category_index(PATH_TO_LABELS)
    session = tf.Session(graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    app.run(host='0.0.0.0', port=80)

