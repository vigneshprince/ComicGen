from pathlib import Path
import cv2
import dlib
import numpy as np
import argparse
from contextlib import contextmanager
from wide_resnet import WideResNet



def yield_images_from_dir(img_name):
    img = cv2.imread(img_name)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r)))


def genage(img_name):
    
    detector = dlib.get_frontal_face_detector()
    weight_file="./pretrained_models/weights.28-3.73.hdf5"
    img_size = 64
    model = WideResNet(img_size, depth=16, k=8)()
    model.load_weights(weight_file)

    image_generator = yield_images_from_dir(img_name) 

    for img in image_generator:
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))
        margin=0.4
        if len(detected) > 0:
            for i, d in enumerate(detected):
                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                xw1 = max(int(x1 - margin * w), 0)
                yw1 = max(int(y1 - margin * h), 0)
                xw2 = min(int(x2 + margin * w), img_w - 1)
                yw2 = min(int(y2 + margin * h), img_h - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()
            finalage = int(predicted_ages[0])
            finalgender="M" if predicted_genders[i][0] < 0.5 else "F"
            print(finalage,finalgender)
    return finalage,finalgender

