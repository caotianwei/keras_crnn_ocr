import PIL.Image as Image
import numpy as np
from keras.utils import to_categorical
import cv2
from config.keys import alphabet
from crnn.model import get_model


def get_demo_train_set():
    x = np.array(Image.open('test_1.jpg').resize((208,32)).convert('L')).reshape((32, 208, 1))
    y = [1,2,1,3,4,5,6,7,8,9,10]  # '号牌号码晥E35087'
    return np.array([x, x]), to_categorical([y, y])


def train(model, X, Y):
    for i in range(1000):
        length = 208 / 4 - 2
        Y_ = Y.argmax(axis=2)
        xs, ys = [X, Y_, np.ones(2) * length, np.ones(2) * 11], np.ones(2)
        model.train_on_batch(xs, ys)
        loss = model.evaluate(xs, ys)
        print("step-{}, loss-{}".format(i, loss))


def decode(pred, characters=alphabet):
    charactersS = characters + u' '
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(charactersS[t[i]])
    return u''.join(char_list)


model, crnn_model = get_model(height=32, nclass=len(alphabet) + 1)
xs, ys = get_demo_train_set()
print(xs.shape)

'''train or load'''
train(model, xs, ys)
# crnn_model.load_weights('keras.hdf5')
print("loaded...")


'''predict'''
yp = crnn_model.predict(xs)

print(xs.shape)
print(yp.shape)
# print(ys.shape)
print(yp.argmax(axis=2))
print(decode(yp[:,2:,:]))