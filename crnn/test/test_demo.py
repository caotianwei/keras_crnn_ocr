from utils.data_utils import load_data
from crnn.model import get_model, predict, train
from config.keys import alphabet
from metrics.metrics import avg_edit_distance, avg_accuracy
import config.train_settings as train_cfg
import config.test_settings as test_cfg
import pandas as pd
# path = 'C:\\Users\\David\\driving_card_crnn\\labeling\\test_demo\\imgs'
# fname = 'fields.txt'
# model_path = 'C:\\Users\\David\\driving_card_crnn\\keras_crnn_ocr\\crnn\\pretrained\\keras.hdf5'



'''load'''
# xs, ys, labels = load_data(train_cfg.TRAIN_SET_PATH, train_cfg.TRAIN_FILE_NAME)
xs, ys, labels = load_data(test_cfg.TEST_SET_PATH, test_cfg.TEST_FILE_NAME)
test_x, test_y, test_labels = load_data(test_cfg.TEST_SET_PATH, test_cfg.TEST_FILE_NAME)

def test_model(model_path):
    model, crnn_model = get_model(height=32, nclass=len(alphabet) + 1)
    crnn_model.load_weights(model_path)
    train(model, xs, ys, labels)
    # crnn_model.save_weights('crnn_1_weights.h5')
    '''test'''
    predict_labels = predict(crnn_model, test_x)
    predict_labels = [label.replace('_', '-').replace('(', '（').replace(')', '）') for label in predict_labels]
    score = avg_edit_distance(test_labels, predict_labels)
    accuracy = avg_accuracy(test_labels, predict_labels)

    example = pd.DataFrame({'test': test_labels[:20], 'predict': predict_labels[:20]})
    # print(example)
    # print('score-{}'.format(score))
    return example, score, accuracy


# e1, s1, a1 = test_model('keras.hdf5')
e2, s2, a2 = test_model('crnn_1_weights.h5')

# print('-------训练前--------')
# print(e1)
# print('score-{}'.format(s1))
# print('acc-{}'.format(a1))
# print('---------------------')

print('-------训练后--------')
print(e2)
print('score-{}'.format(s2))
print('acc-{}'.format(a2))
print('---------------------')
