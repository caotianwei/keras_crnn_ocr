from crnn.model import get_model
import PIL.Image as Image
import numpy as np
from keras.utils import to_categorical
from crnn.keys import alphabet


def get_demo_train_set():
    x = np.array(Image.open('test_1.jpg').convert('L')).reshape((49, 227, 1))
    y = [1,2,1,3,4,5,6,7,8,9,10]  # '号牌号码晥E35087'
    return np.array([x, x]), to_categorical([y, y])


def train(model, X, Y):
    for i in range(1000):
        length = 227 / 4 - 2
        Y_ = Y.argmax(axis=2)
        xs, ys = [X, Y_, np.ones(2) * length, np.ones(2) * 11], np.ones(2)
        model.train_on_batch(xs, ys)
        loss = model.evaluate(xs, ys)
        print("step-{}, loss-{}".format(i, loss))



model, crnn_model = get_model(height=32, nclass=len(alphabet) + 1)
xs, ys = get_demo_train_set()

'''train or load'''
# train(model, xs, ys)
crnn_model.load_weights('keras.hdf5')
print('ok')


'''predict'''
# yp = crnn_model.predict(xs)
#
# print(xs.shape)
# print(yp.shape)
# print(ys.shape)
# print(yp.argmax(axis=2))