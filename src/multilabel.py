#！ -*- coding:utf-8 -*-
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score


set_gelu('tanh')  # 切换gelu版本


num_classes = 31
data_path = './data/financial_borrowing_focus/'
model_path = './model/'
maxlen = 128
batch_size = 16
config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    data = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            items = line.strip().split('\t')
            length = len(items)
            if num_classes != length-1:
                print('error data')
                continue
            # 默认第一列是text
            text = items[0]
            # label 是one-hot
            label = [0] * (length - 1)
            for i in range(1, length):
                label[i-1] = int(items[i])
            data.append((text, label))
    return data


# 加载数据集
train_data = load_data(data_path + 'train.data')
valid_data = load_data(data_path + 'valid.data')
test_data = load_data(data_path + 'test.data')


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=1,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
    #     1000: 1,
    #     2000: 0.1
    # }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate2(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


def evaluate(data):
    y_t, y_p = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true) > 0.5
        y_t.extend(y_true)
        y_p.extend(y_pred)

    total = len(y_t)
    right = 0.
    for i in range(total):
        if all(y_t[i] == y_p[i]):
            right += 1
    acc = right / total
    return acc


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(model_path + 'best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='min'),
        ModelCheckpoint(model_path + 'best_model.weights', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

evaluator = Evaluator()
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=valid_generator.forfit(),
    validation_steps=len(valid_generator),
    callbacks=callbacks
)

model.load_weights(model_path + 'best_model.weights')
print(u'final test acc: %05f\n' % (evaluate(test_generator)))
