import keras2onnx
import numpy as np
import onnx
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

ONNX_FILE_NAME = "onnx_yolov2_tiny_voc.onnx"


def conv_model():
    target_opset = 8
    keras_model = load_model("yolov2-tiny-voc.h5")
    onnx_model = keras2onnx.convert_keras(
        model=keras_model, target_opset=target_opset, channel_first_inputs="input_1"
    )
    onnx.save_model(onnx_model, ONNX_FILE_NAME)


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def softmax(x):
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)


def detect_img():
    """
    - https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/onnx/onnx-convert-aml-deploy-tinyyolo.ipynb
    - https://tech-blog.optim.co.jp/entry/2018/12/05/160831
    - https://qiita.com/miyamotok0105/items/1aa653512dd4657401db
    - https://machinethink.net/blog/object-detection-with-yolo/
    - https://github.com/shi3z/onnx-example
    """
    import onnxruntime

    sess = onnxruntime.InferenceSession(ONNX_FILE_NAME)
    input_name = sess.get_inputs()[0].name
    img = Image.open("person.jpg")
    img = img.resize((416, 416))  # for tiny_yolov2
    image_data = np.array(img, dtype="float32")
    image_data /= 255.0
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    image_data = image_data.transpose(0, 3, 1, 2)
    print(image_data.shape)

    out = sess.run(None, {input_name: image_data.astype(np.float32)})
    out = out[0][0]
    print("shape", np.shape(out))
    numClasses = 20
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    # Color
    clut = [
        (0, 0, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 0, 255),
        (0, 255, 0),
        (0, 255, 128),
        (128, 255, 0),
        (128, 128, 0),
        (0, 128, 255),
        (128, 0, 128),
        (255, 0, 128),
        (128, 0, 255),
        (255, 128, 128),
        (128, 255, 128),
        (255, 255, 0),
        (255, 0, 128),
        (128, 0, 255),
        (255, 128, 128),
        (128, 255, 128),
        (255, 255, 0),
    ]
    print(len(clut))

    # Label
    label = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    bb_cnt = 5  # number of bounding box prediction in each ancohrs
    draw = ImageDraw.Draw(img)
    for cy in range(0, 13):
        for cx in range(0, 13):
            for b in range(0, bb_cnt):
                channel = b * (numClasses + bb_cnt)
                # tx = out[channel][cy][cx]
                # ty = out[channel + 1][cy][cx]
                # tw = out[channel + 2][cy][cx]
                # th = out[channel + 3][cy][cx]
                # tc = out[channel + 4][cy][cx]
                tx = out[cy][cx][channel]
                ty = out[cy][cx][channel + 1]
                tw = out[cy][cx][channel + 2]
                th = out[cy][cx][channel + 3]
                tc = out[cy][cx][channel + 4]

                x = (float(cx) + sigmoid(tx)) * 32
                y = (float(cy) + sigmoid(ty)) * 32

                w = np.exp(tw) * 32 * anchors[2 * b]
                h = np.exp(th) * 32 * anchors[2 * b + 1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0, numClasses):
                    # classes[c] = out[channel + bb_cnt + c][cy][cx]
                    classes[c] = out[cy][cx][channel + bb_cnt + c]
                classes = softmax(classes)
                detectedClass = classes.argmax()

                if 0.5 < classes[detectedClass] * confidence:
                    print("probability of classes", classes)
                    color = clut[detectedClass]
                    print(
                        detectedClass,
                        label[detectedClass],
                        classes[detectedClass] * confidence,
                    )
                    x = x - w / 2
                    y = y - h / 2
                    draw.line((x, y, x + w, y), fill=color)
                    draw.line((x, y, x, y + h), fill=color)
                    draw.line((x + w, y, x + w, y + h), fill=color)
                    draw.line((x, y + h, x + w, y + h), fill=color)
    img.save("result.png")


if __name__ == "__main__":
    # conv_model()
    detect_img()
