import pandas as pd
import os
import PIL.Image as Image
from config.keys import alphabet
import numpy as np


def load_data(path, file_name):
    test_data = pd.read_table(os.path.join(path, file_name), header=None, sep='\t')
    imgs = test_data[0].map(lambda img_path: read_image(os.path.join(path, img_path)))
    labels = test_data[1].map(lambda w: w.strip())
    label_ids = test_data[1].map(lambda words: words_to_category(words))
    return imgs, label_ids, labels


def read_image(path):
    img = Image.open(path)
    width, height = img.size
    return np.array(img.resize((width, 32)).convert('L')).reshape((32, width, 1))


def words_to_category(words):
    ids = []
    max_id = len(alphabet)
    for word in words:
        if word in alphabet:
            ids.append(alphabet.index(word))
        else:
            ids.append(max_id)
    return ids


def decode(pred, characters=alphabet):
    characterss = characters + u' '
    t = pred.argmax(axis=2)[0]
    length = len(t)
    char_list = []
    n = len(characters)
    for i in range(length):
        if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
            char_list.append(characterss[t[i]])
    return u''.join(char_list)
