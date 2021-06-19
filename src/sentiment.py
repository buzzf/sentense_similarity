#! -*- coding:utf-8 -*-
import json
import numpy as np
import pandas as pd
import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Dense
from keras.optimizers import Adam


maxlen = 128
batch_size = 16
model_path = './model/'
config_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


neg = pd.read_excel('data/neg.xls', header=None)
pos = pd.read_excel('data/pos.xls', header=None)
data = []
for d in neg[0]:
    data.append((d, 0))
for d in pos[0]:
    data.append((d, 1))

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


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


train_generator = data_generator(train_data, batch_size=32)  # 默认32
valid_generator = data_generator(valid_data, batch_size=32)


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
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5), # 用足够小的学习率
    metrics=['accuracy']
)
model.summary()


callbacks = [
        EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, mode='min'),
        ModelCheckpoint(model_path + 'best_model.weights', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ]

model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=3,
    validation_data=valid_generator.forfit(),
    validation_steps=len(valid_generator),
    callbacks=callbacks
)

# model.load_weights(model_path + 'best_model.weights')
# print(u'final test acc: %05f\n' % (evaluate(test_generator)))