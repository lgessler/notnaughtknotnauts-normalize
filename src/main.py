import argparse
import os, random
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics, optimizers

import numpy as np
from keras import backend as K

# default args

args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=1, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.1, type=float, help='Learning rate')
args.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
args.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
args.add_argument('-em','--embedding-size', default=100, type=int, help='Embedding dimension size')
args.add_argument('-hs','--hidden-size', default=10, type=int, help='Hidden layer size')
args.add_argument('-b','--batch-size', default=50, type=int, help='Batch Size')
args = args.parse_args()
print("Arguments: " + str(args))




train_file = r'pos-data/a3/en-ud-train.upos.tsv'
dev_file = r'pos-data/a3/en-ud-dev.upos.tsv'
test_file = r'pos-data/a3/en-ud-test.upos.tsv'

UNK = '[UNK]'
PAD = '[PAD]'
START = '<s>'
END = '</s>'

def get_vocabulary_and_data(data_file, max_vocab_size=10000):
    vocab = Counter()
    data = []
    with open(data_file, 'r', encoding='utf8') as f:
        sent = [START]
        for line in f:
            if line.strip():
                tok, pos = line.strip().split('\t')[0], line.strip().split('\t')[1]
                sent.append(tok)
                vocab[tok]+=1
                vocab[START]+=1
                vocab[END]+=1
            elif sent:
                sent.append(END)
                sent = transform_text_sequence(sent)
                data.append(sent)
                sent = [START]
    vocab = sorted(list(vocab))
    if max_vocab_size:
        vocab = vocab[:len(vocab)-max_vocab_size]
    vocab = [UNK, PAD] + vocab

    return {k:v for v,k in enumerate(vocab)}, data


def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label, vocab):
    vec = [1.0 if l==label else 0.0 for l in vocab]
    return vec


def batch_generator_lm(data, vocab, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for sent in data:
            batch_x.append(vectorize_sequence(sent, vocab))
            batch_y.append([one_hot_encode_label(token, vocab) for token in shift_by_one(sent)])
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                batch_y = pad_sequences(batch_y, one_hot_encode_label(PAD, vocab))
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def describe_data(data, gold_labels, label_set, generator):
    batch_x, batch_y = [], []
    for bx, by in generator:
        batch_x = bx
        batch_y = by
        break
    print('Data example:',data[0])
    print('Label:',None)
    print('Label count:', None)
    print('Batch input shape:', batch_x.shape)
    print('Batch output shape:', batch_y.shape)


def pad_sequences(batch_x, pad_value):
    ''' This function should take a batch of sequences of different lengths
        and pad them with the pad_value token so that they are all the same length.

        Assume that batch_x is a list of lists.
    '''
    pad_length = len(max(batch_x, key=lambda x: len(x)))
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))

    return batch_x


def generate_text(language_model, vocab):
    prediction = [START]
    while not (prediction[-1] == END or len(prediction)>=50):
        next_token_one_hot = language_model.predict(np.array([[vocab[p] for p in prediction]]), batch_size=1)[0][-1]
        next_tokens = sorted([i for i,v in enumerate(next_token_one_hot)], key=lambda i:next_token_one_hot[i], reverse=True)[:10]
        next_token = next_tokens[random.randint(0, 9)]
        for w, i in vocab.items():
            if i==next_token:
                prediction.append(w)
                break
    return prediction


# TODO
def transform_text_sequence(seq):
    '''
    Implement this function if you want to transform the input text,
    for example normalizing case.
    '''
    return seq

# TODO
def shift_by_one(seq):
    '''
    input: ['<s>', 'The', 'dog', 'chased', 'the', 'cat', 'around', 'the', 'house', '</s>']
    output: ['The', 'dog', 'chased', 'the', 'cat', 'around', 'the', 'house', '</s>', '[PAD]']
    '''
    return seq[1:] + [PAD]


vocab, train_data = get_vocabulary_and_data(train_file)
_, dev_data = get_vocabulary_and_data(dev_file)
_, test_data = get_vocabulary_and_data(test_file)

describe_data(train_data, None, None,
              batch_generator_lm(train_data, vocab, args.batch_size))

# Implement your model here! ----------------------------------------------------------------------
# Use the variables args.batch_size, args.hidden_size, args.embedding_size, args.dropout, args.epochs
# You can input these as command line parameters.
num_words = len(vocab.keys())

language_model = Sequential()
language_model.add(Embedding(num_words, args.embedding_size))
language_model.add(Dropout(args.dropout))
language_model.add(LSTM(args.hidden_size, return_sequences=True))
language_model.add(Dropout(args.dropout))
language_model.add(LSTM(args.hidden_size, return_sequences=True))
language_model.add(Dropout(args.dropout))
language_model.add(TimeDistributed(Dense(num_words, activation='softmax')))


# ------------------------------------------------------------------------------------------------

adadelta = optimizers.Adadelta(clipnorm=1.0)
language_model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])


# Training
language_model.fit_generator(batch_generator_lm(train_data, vocab, args.batch_size),
                             epochs=args.epochs,
                             steps_per_epoch=len(train_data)/args.batch_size,
                             callbacks=[EarlyStopping(monitor="acc", patience=2)])

# Evaluation
loss, acc = language_model.evaluate_generator(batch_generator_lm(dev_data, vocab),
                                              steps=len(dev_data))
print('Dev Loss:', loss, 'Dev Acc:', acc)
loss, acc = language_model.evaluate_generator(batch_generator_lm(test_data, vocab),
                                          steps=len(test_data))
print('Test Loss:', loss, 'Test Acc:', acc)


for i in range(10):
    text = generate_text(language_model, vocab)
    print(text)
