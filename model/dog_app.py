from keras.preprocessing import image
import numpy as np
from glob import glob
import pickle
import random
import cv2
from model.extract_bottleneck_features import extract_InceptionV3
from keras.applications.resnet50 import preprocess_input
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential
from keras.applications.resnet50 import ResNet50
import urllib.request
import os


random.seed(8675309)
path = __file__.replace('model/dog_app.py', '')


def load_dog_names():
    with open(os.path.join(path, 'model/names.pkl'), 'rb') as f:
        dog_names = pickle.load(f)
        dog_names = [x.split('/')[-1][4:] for x in dog_names]
    return dog_names


def load_dataset(path):
    files = glob(path + '*')
    targets = [x.split('/')[-1][:-10] for x in files]
    return files, targets


def path_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in img_paths]
    return np.vstack(list_of_tensors)


def get_model(p=os.path.join(path, 'model/DogInceptionV3Data.npz')):
    url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz'
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    bottleneck_features = np.load(p)
    train_Inception = bottleneck_features['train']
    Inception_model = Sequential()
    Inception_model.add(GlobalAveragePooling2D(input_shape=train_Inception.shape[1:]))
    Inception_model.add(Dense(256, activation='relu'))
    Inception_model.add(Dense(133, activation='softmax'))
    Inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    Inception_model.load_weights(os.path.join(path, 'saved_models/weights.best.Inception.hdf5'))
    return Inception_model


class DogDetection:
    def __init__(self):
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz'
        p = os.path.join(path, 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
        if not os.path.exists(p):
            urllib.request.urlretrieve(url, p)
        self.resnet = ResNet50(weights='imagenet')
        # pickle.dump(self.resnet, open('resnet.pkl', 'wb'))
        # self.resnet = pickle.load(open('resnet.pkl', 'rb'))
        self.model = get_model()
        # pickle.dump(self.model, open('model.pkl', 'wb'))
        # self.model = pickle.load(open('model.pkl', 'rb'))
        self.dog_names = load_dog_names()
        self.train_files, self.train_targets = load_dataset(os.path.join(path, 'static/'))
        self.face_cascade = cv2.CascadeClassifier(os.path.join(path, 'model/haarcascade_frontalface_alt.xml'))

    def face_detector(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray)
        return len(faces) > 0

    def Inception_predict_breed(self, img_path):
        bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
        predicted_vector = self.model.predict(bottleneck_feature)
        return self.dog_names[np.argmax(predicted_vector)]

    def ResNet50_predict_labels(self, img_path):
        img = preprocess_input(path_to_tensor(img_path))
        return np.argmax(self.resnet.predict(img))

    def dog_detector(self, img_path):
        prediction = self.ResNet50_predict_labels(img_path)
        return (prediction <= 268) & (prediction >= 151)

    def which_dog(self, img_path):
        """
        This function determines the breed of the dog using an input image.
        If the image is not an image of dog, but rather of a human,
        it will find a dog that resembles the picture of the human. If the image is neither human face nor dog face, it will return
        a string indicating "no face detected".

        Args:
            img_path (str): The path to an image
        """
        is_dog = self.dog_detector(img_path)
        is_face = self.face_detector(img_path)
        dog_breed = self.Inception_predict_breed(img_path)
        if is_dog or is_face:

            if is_face:
                message = f"The dog look alike is {dog_breed}"
            else:
                message = dog_breed
        else:
            message = "no face or dog detected"
        return message, dog_breed

    def get_image(self, breed):
        ids = np.where([breed == x for x in self.train_targets])[0]
        idx = np.random.choice(ids)
        image = self.train_files[idx]
        return image


def main():
    import matplotlib.pyplot as plt
    from glob import glob
    import cv2
    files = glob('../images/*')
    breed_detector = DogDetection()
    for f in files:
        message, breed = breed_detector.which_dog(f)
        print(message)
        img = cv2.imread(f)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file = breed_detector.get_image(breed)
        img_similar = cv2.imread(file)
        cv_rgb_similar = cv2.cvtColor(img_similar, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(cv_rgb)
        ax[1].imshow(cv_rgb_similar)
        ax[1].set_title(breed)
        plt.pause(0.0001)
    plt.show()


if __name__ == '__main__':
    main()