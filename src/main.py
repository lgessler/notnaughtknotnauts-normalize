import argparse
import os, random
import pickle
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics, optimizers

import numpy as np
from keras import backend as K

from parse_spanish import retrieve_all_data

# caching
def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """
    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):   # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                    with open(cachefile, 'rb') as cachehandle:
                        print("using cached result from '%s'" % cachefile)
                        return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                print("saving result to cache '%s'" % cachefile)
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator
retrieve_all_data = cached("cache.pickle")(retrieve_all_data)


# default args
argparser = argparse.ArgumentParser(description='Program description.')
argparser.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
argparser.add_argument('-e','--epochs', default=1, type=int, help='Number of epochs')
argparser.add_argument('-lr','--learning-rate', default=0.1, type=float, help='Learning rate')
argparser.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
argparser.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
argparser.add_argument('-em','--embedding-size', default=100, type=int, help='Embedding dimension size')
argparser.add_argument('-hs','--hidden-size', default=10, type=int, help='Hidden layer size')
argparser.add_argument('-b','--batch-size', default=50, type=int, help='Batch Size')

UNK = 'अ'
PAD = 'आ'
START = 'श'
END = 'स'


def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(), key=lambda k: vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label, labels):
    vec = [1.0 if l == label else 0.0 for l in labels]
    return np.array(vec)


def batch_generator(data, labels, vocab, labels_vocab, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for word, norm_word in zip(data, labels):
            word = START + word + END
            norm_word = START + norm_word + END
            batch_x.append(vectorize_sequence(word, vocab))
            batch_y.append([one_hot_encode_label(label, labels_vocab) for label in norm_word])
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                pad_length = len(max(batch_x + batch_y, key=lambda x: len(x)))
                batch_x = pad_sequences(batch_x, pad_length, vocab[PAD])
                batch_y = pad_sequences(batch_y, pad_length, one_hot_encode_label(PAD, labels_vocab))
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def describe_data(data, gold_labels, label_set, generator):
    batch_x, batch_y = [], []
    for bx, by in generator:
        batch_x = bx
        batch_y = by
        break
    print('Data example:', data[0])
    print('Label:', gold_labels[0])
    print('Label count:', len(label_set))
    print('Batch input shape:', batch_x.shape)
    print('Batch output shape:', batch_y.shape)


def pad_sequences(batch_x, pad_length, pad_value):
    ''' This function should take a batch of sequences of different lengths
        and pad them with the pad_value token so that they are all the same length.

        Assume that batch_x is a list of lists.
    '''
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))

    return batch_x


# TODO
def transform_text_sequence(seq):
    '''
    Implement this function if you want to transform the input text,
    for example normalizing case.
    '''
    return seq


def make_model(vocab, labels_vocab, args):
    num_chars = len(vocab.keys())
    num_labels = len(labels_vocab.keys())

    model = Sequential()
    model.add(Embedding(num_chars, args.embedding_size))
    model.add(Dropout(args.dropout))
    model.add(Bidirectional(LSTM(args.hidden_size, return_sequences=True)))
    model.add(Dropout(args.dropout))
    model.add(TimeDistributed(Dense(num_labels, activation='softmax')))

    for layer in model.layers:
        print(layer.output_shape)

    adadelta = optimizers.Adadelta(clipnorm=1.0)
    model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def eval_model(model, data, labels, vocab, labels_vocab, args):
    loss, acc = model.evaluate_generator(batch_generator(data, labels, vocab, labels_vocab, args.batch_size), steps=len(data))
    print('Loss:', loss, 'Acc:', acc)


def main(args):
    train_data, train_labels, vocab, \
    dev_data, dev_labels, \
    test_data, test_labels = retrieve_all_data()

    vocab = {c: i for i, c in enumerate(list(set(c for w in vocab for c in w)) + [UNK, PAD, START, END])}
    labels_vocab = {c: i for i, c in
                    enumerate(list(set(c for w in train_labels + dev_labels + test_labels for c in w))
                              + [UNK, PAD, START, END])}

    describe_data(train_data, train_labels, vocab, batch_generator(train_data, train_labels, vocab, labels_vocab, args.batch_size))

    model = make_model(vocab, labels_vocab, args)

    # training
    model.fit_generator(batch_generator(train_data, train_labels, vocab, labels_vocab, args.batch_size),
                        epochs=args.epochs,
                        steps_per_epoch=len(train_data) / args.batch_size,
                        callbacks=[EarlyStopping(monitor="acc", patience=2)])

    # Evaluation
    eval_model(model, train_data, train_labels, vocab, labels_vocab, args)


if __name__ == '__main__':
    args = argparser.parse_args()
    print("Arguments: " + str(args))
    main(args)

