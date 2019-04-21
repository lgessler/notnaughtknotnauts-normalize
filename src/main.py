import argparse
import os, random
import pickle
import numpy as np
from collections import Counter

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras.callbacks import EarlyStopping
from keras import metrics, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from parse_spanish import retrieve_tokens, retrieve_century

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

retrieve_tokens = cached("tokens.pickle")(retrieve_tokens)
retrieve_century = cached("century.pickle")(retrieve_century)


# default args
argparser = argparse.ArgumentParser(description='Program description.')
argparser.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
argparser.add_argument('-e','--epochs', default=3, type=int, help='Number of epochs')
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


def vectorize_sequence(word, vocab):
    word = [char if char in vocab else UNK for char in word]
    return [vocab[char] for char in word]


def one_hot_encode(char, vocab):
    vec = [1.0 if c == char else 0.0 for c in vocab]
    return np.array(vec)


def batch_generator(orig, norm, orig_vocab, norm_vocab, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for orig_word, norm_word in zip(orig, norm):
            orig_word = START + orig_word + END
            norm_word = START + norm_word + END
            batch_x.append(vectorize_sequence(orig_word, orig_vocab))
            batch_y.append([one_hot_encode(char, norm_vocab) for char in norm_word])
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                pad_length = len(max(batch_x + batch_y, key=lambda x: len(x)))
                batch_x = pad_sequences(batch_x, pad_length, orig_vocab[PAD])
                batch_y = pad_sequences(batch_y, pad_length, one_hot_encode(PAD, norm_vocab))
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []
        yield np.array(batch_x), np.array(batch_y)


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


def make_model(orig_vocab, norm_vocab, args):
    num_chars_orig = len(orig_vocab.keys())
    num_chars_norm = len(norm_vocab.keys())

    model = Sequential()
    model.add(Embedding(num_chars_orig, args.embedding_size))
    model.add(Dropout(args.dropout))
    model.add(Bidirectional(LSTM(args.hidden_size, return_sequences=True)))
    model.add(Dropout(args.dropout))
    model.add(TimeDistributed(Dense(num_chars_norm, activation='softmax')))

    for layer in model.layers:
        print(layer.output_shape)

    adadelta = optimizers.Adadelta(clipnorm=1.0)
    model.compile(optimizer=adadelta, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def eval_model(model, orig, norm, orig_vocab, norm_vocab, args):
    loss, acc = model.evaluate_generator(
        batch_generator(orig, norm, orig_vocab, norm_vocab, args.batch_size), steps=len(orig))

    # Vectorized inputs and labels
    orig_vecs = []
    norm_vecs = []

    for orig_word, norm_word in zip(orig, norm):
        orig_word = START + orig_word + END
        norm_word = START + norm_word + END
        orig_vecs.append(vectorize_sequence(orig_word, orig_vocab))
        norm_vecs.append([one_hot_encode(c, norm_vocab) for c in norm_word])
    pad_length = len(max(orig_vecs + norm_vecs, key=lambda x: len(x)))
    orig_vecs = pad_sequences(orig_vecs, pad_length, orig_vocab[PAD])
    norm_vecs = pad_sequences(norm_vecs, pad_length, one_hot_encode(PAD, norm_vocab))
    orig_vecs = np.array(orig_vecs)
    norm_vecs = np.array(norm_vecs)

    loss, acc = model.evaluate(orig_vecs, norm_vecs, batch_size=1000, verbose=1)

    print('Loss:', loss, 'Acc:', acc)

def main(args):
    orig_toks, norm_toks = retrieve_tokens()
    vocab_orig = {c: i for i, c in enumerate(list(set(c for w in orig_toks for c in w)) + [UNK, PAD, START, END])}
    vocab_norm = {c: i for i, c in enumerate(list(set(c for w in norm_toks for c in w)) + [UNK, PAD, START, END])}
    vocab_both = {c: i for i, c in
                  enumerate(list(set(c for w in (orig_toks + norm_toks) for c in w)) + [UNK, PAD, START, END])}
    train_orig, test_orig, train_norm, test_norm = train_test_split(np.array(orig_toks), np.array(norm_toks),
                                                                    test_size=0.2, random_state=42)

    describe_data(train_orig, train_norm, vocab_orig,
                  batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args.batch_size))

    model = make_model(vocab_orig, vocab_norm, args)

    # training
    model.fit_generator(batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args.batch_size),
                        epochs=args.epochs,
                        steps_per_epoch=len(train_orig) / args.batch_size,
                        max_queue_size=1000,
                        callbacks=[EarlyStopping(monitor="acc", patience=2)])

    # Evaluation
    eval_model(model, test_orig, test_norm, vocab_orig, vocab_norm, args)


if __name__ == '__main__':
    args = argparser.parse_args()
    print("Arguments: " + str(args))
    main(args)

