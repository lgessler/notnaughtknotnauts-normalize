import argparse
import os, random
import pickle
import numpy as np
from collections import Counter

from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Input
from keras.callbacks import EarlyStopping
from keras import metrics, optimizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

from parse_spanish import retrieve_tokens, retrieve_century

# default args
argparser = argparse.ArgumentParser(description='Program description.')
argparser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
argparser.add_argument('-b', '--batch-size', default=50, type=int, help='Batch Size')
argparser.add_argument('-do', '--dropout', default=0.3, type=float, help='Dropout rate')
argparser.add_argument('-em', '--embedding-size', default=100, type=int, help='Embedding dimension size')
argparser.add_argument('-hs', '--hidden-size', default=10, type=int, help='Hidden layer size')
argparser.add_argument('-c', '--century', default="all", type=str, help='Century')
argparser.add_argument('-cn', '--context-n', default=0, type=int, help='Number of toks to look to the left and right')

UNK = 'अ'
PAD = 'आ'
START = 'श'
END = 'स'
TOKSEP = 'क'


def vectorize_sequence_with_context(tok_index, vocab, tokens, args):
    result = []
    for i in range(-args.context_n, args.context_n + 1):
        chars = [vocab[char] if char in vocab else UNK
                   for char in tokens[(tok_index + i) % len(tokens)]]
        if i == 0:
            chars = [vocab[START]] + chars + [vocab[END]]
        if i < args.context_n:
            chars += [vocab[TOKSEP]]
        result += chars

    return result


def vectorize_sequence(word, vocab):
    word = [char if char in vocab else UNK for char in word]
    return [vocab[char] for char in word]


def one_hot_encode(char, vocab):
    vec = [1.0 if c == char else 0.0 for c in vocab]
    return np.array(vec)


def batch_generator(orig, norm, orig_vocab, norm_vocab, args):
    batch_size = args.batch_size
    while True:
        batch_x = []
        batch_y = []
        for i, (orig_word, norm_word) in enumerate(zip(orig, norm)):
            orig_word = START + orig_word + END
            norm_word = START + norm_word + END
            batch_x.append(vectorize_sequence_with_context(i, orig_vocab, orig, args))
            batch_y.append([one_hot_encode(char, norm_vocab) for char in norm_word])
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                pad_length = len(max(batch_x + batch_y, key=lambda x: len(x)))
                batch_x = pad_sequences(batch_x, pad_length, orig_vocab[PAD])
                batch_y = pad_sequences(batch_y, pad_length, one_hot_encode(PAD, norm_vocab))
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


def char_accuracy(results):
    accs = []
    for pred, norm in results:
        acc = np.mean([1 if i < len(norm) and norm[i] == c else 0
                       for i, c in enumerate(pred)])
        accs.append(acc)
    return np.mean(accs)

def word_accuracy(results):
    return np.mean([1 if result[0] == result[1] else 0 for result in results])

def eval_model(model, orig, norm, orig_vocab, norm_vocab, args):
    loss, acc = model.evaluate_generator(
        batch_generator(orig, norm, orig_vocab, norm_vocab, args), steps=len(orig))

    # Vectorized inputs and labels
    orig_vecs = []
    norm_vecs = []

    for i, (orig_word, norm_word) in enumerate(zip(orig, norm)):
        orig_word = START + orig_word + END
        norm_word = START + norm_word + END
        orig_vecs.append(vectorize_sequence_with_context(i, orig_vocab, orig, args))
        norm_vecs.append([one_hot_encode(c, norm_vocab) for c in norm_word])
    pad_length = len(max(orig_vecs + norm_vecs, key=lambda x: len(x)))
    orig_vecs = pad_sequences(orig_vecs, pad_length, orig_vocab[PAD])
    norm_vecs = pad_sequences(norm_vecs, pad_length, one_hot_encode(PAD, norm_vocab))
    orig_vecs = np.array(orig_vecs)
    norm_vecs = np.array(norm_vecs)

    loss, acc = model.evaluate(orig_vecs, norm_vecs, batch_size=1000, verbose=1)

    print('Loss:', loss, 'Acc:', acc)

    # A list of possible output characters
    norm_chars = list(norm_vocab.keys())

    predicted_words = []
    # Each prediction is something like a 26 x 57 array, where 26 is the number of
    # characters in the word and 57 is the size of the vocabulary
    test_predictions = model.predict(orig_vecs, batch_size=1000)
    for word_vector in test_predictions:
        word = ""
        for character_num in range(len(word_vector)):
            # Find the index of the character in the vocabulary with highest probability according to the model
            vocab_index = np.argmax(word_vector[character_num])
            # Convert that index back into a character and append it to the current word
            character = norm_chars[vocab_index]
            word += character

        # Remove all PAD characters in each word and START and END tokens
        word = word.replace(PAD, "")
        word = word.replace(START, "")
        word = word.replace(END, "")
        predicted_words.append(word)

    results = np.column_stack((predicted_words, norm))
    for i, (predicted_word, gold_word) in enumerate(results):
        orig_word = orig[i]
        print(f"Input: {orig_word}\tPredicted: {predicted_word}\tGold: {gold_word}")

    # Calculate accuracy as the percentage of exact matches between the model output (without PAD) and the labels
    word_acc = word_accuracy(results)
    char_acc = char_accuracy(results)
    print(f"Word accuracy after removing all padding characters: {word_acc}")
    print(f"Char accuracy after removing all padding characters: {char_acc}")


def eval_model_with_baseline(model, train_orig, train_norm, orig, norm, orig_vocab, norm_vocab, args):
    loss, acc = model.evaluate_generator(
        batch_generator(orig, norm, orig_vocab, norm_vocab, args), steps=len(orig))

    # Vectorized inputs and labels
    orig_vecs = []
    norm_vecs = []

    for i, (orig_word, norm_word) in enumerate(zip(orig, norm)):
        orig_word = START + orig_word + END
        norm_word = START + norm_word + END
        orig_vecs.append(vectorize_sequence_with_context(i, orig_vocab, orig, args))
        norm_vecs.append([one_hot_encode(c, norm_vocab) for c in norm_word])
    pad_length = len(max(orig_vecs + norm_vecs, key=lambda x: len(x)))
    orig_vecs = pad_sequences(orig_vecs, pad_length, orig_vocab[PAD])
    norm_vecs = pad_sequences(norm_vecs, pad_length, one_hot_encode(PAD, norm_vocab))
    orig_vecs = np.array(orig_vecs)
    norm_vecs = np.array(norm_vecs)

    loss, acc = model.evaluate(orig_vecs, norm_vecs, batch_size=1000, verbose=1)

    print('Loss:', loss, 'Acc:', acc)

    # A list of possible output characters
    norm_chars = list(norm_vocab.keys())

    predicted_words = []
    # Each prediction is something like a 26 x 57 array, where 26 is the number of
    # characters in the word and 57 is the size of the vocabulary
    test_predictions = model.predict(orig_vecs, batch_size=1000)
    for word_vector in test_predictions:
        word = ""
        for character_num in range(len(word_vector)):
            # Find the index of the character in the vocabulary with highest probability according to the model
            vocab_index = np.argmax(word_vector[character_num])
            # Convert that index back into a character and append it to the current word
            character = norm_chars[vocab_index]
            word += character

        # Remove all PAD characters in each word and START and END tokens
        word = word.replace(PAD, "")
        word = word.replace(START, "")
        word = word.replace(END, "")
        predicted_words.append(word)

    results = np.column_stack((predicted_words, norm))
    for i, (predicted_word, gold_word) in enumerate(results):
        orig_word = orig[i]
        print(f"Input: '{orig_word}'\tPredicted: '{predicted_word}'\tGold: '{gold_word}'")

    # get baseline results and choose baseline prediction if the word was seen before
    baseline_dict, baseline_results = predict_baseline(train_orig, train_norm, orig, norm)
    combined_results = np.array([baseline_results[i] if baseline_results[i][0] in baseline_dict
                                 else results[i] for i in range(len(results))])
    print("Model results")
    print(results)
    print("Baseline results")
    print(baseline_results)
    print("Combined results")
    print(combined_results)

    # Calculate accuracy as the percentage of exact matches between the model output (without PAD) and the labels
    word_acc = word_accuracy(baseline_results)
    char_acc = char_accuracy(baseline_results)
    print(f"Baseline word accuracy after removing all padding characters: {word_acc}")
    print(f"Baseline char accuracy after removing all padding characters: {char_acc}")
    word_acc = word_accuracy(results)
    char_acc = char_accuracy(results)
    print(f"Model word accuracy after removing all padding characters: {word_acc}")
    print(f"Model char accuracy after removing all padding characters: {char_acc}")
    word_acc = word_accuracy(combined_results)
    char_acc = char_accuracy(combined_results)
    print(f"Hybrid word accuracy after removing all padding characters: {word_acc}")
    print(f"Hybrid char accuracy after removing all padding characters: {char_acc}")



def predict_baseline(train_orig, train_norm, test_orig, test_norm):
    # Initialize an empty dictionary, each entry for a word will be a counter
    word_map = {}
    for index, word in enumerate(train_orig):
        if word not in word_map:
            word_map[word] = Counter()
            word_map[word][train_norm[index]] += 1
        else:
            word_map[word][train_norm[index]] += 1

    baseline_predicted_words = []
    for word in test_orig:
        # For each original word, predict its most common mapping
        if word in word_map:
            predicted_word = word_map[word].most_common(1)[0][0]
        else:
            predicted_word = UNK
        baseline_predicted_words.append([word, predicted_word])

    return word_map, np.array(baseline_predicted_words)


def eval_baseline(train_orig, train_norm, test_orig, test_norm):
    # Initialize an empty dictionary, each entry for a word will be a counter
    _, results = predict_baseline(train_orig, train_norm, test_orig, test_norm)
    print(results)
    word_acc = word_accuracy(results)
    char_acc = char_accuracy(results)
    print(f"Word accuracy for baseline: {word_acc}")
    print(f"Char accuracy for baseline: {char_acc}")


def vocab_dict(toks):
    return {c: i for i, c in enumerate(list(set(c for w in toks for c in w)))}


def lstm_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both):
    num_chars_orig = len(vocab_orig.keys())
    num_chars_norm = len(vocab_norm.keys())

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

    # training
    model.fit_generator(batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args),
                        epochs=args.epochs,
                        steps_per_epoch=len(train_orig) / args.batch_size,
                        max_queue_size=1000,
                        callbacks=[EarlyStopping(monitor="acc", patience=2)])

    # Evaluation
    eval_model(model, test_orig, test_norm, vocab_orig, vocab_norm, args)
    eval_baseline(train_orig, train_norm, test_orig, test_norm)


def hybrid_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both):
    num_chars_orig = len(vocab_orig.keys())
    num_chars_norm = len(vocab_norm.keys())

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

    # training
    model.fit_generator(batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args),
                        epochs=args.epochs,
                        steps_per_epoch=len(train_orig) / args.batch_size,
                        max_queue_size=1000,
                        callbacks=[EarlyStopping(monitor="acc", patience=2)])

    # Evaluation
    eval_model_with_baseline(model, train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, args)



def encoder_decoder_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both):
    num_chars_orig = len(vocab_orig.keys())
    num_chars_norm = len(vocab_orig.keys())
    n_units = 256

	# define training encoder
    encoder_inputs = Input(shape=(None, num_chars_orig))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, num_chars_norm))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_chars_norm, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


    # return all models
    model.fit_generator(batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args),
                        epochs=args.epochs,
                        steps_per_epoch=len(train_orig) / args.batch_size,
                        max_queue_size=1000,
                        callbacks=[EarlyStopping(monitor="acc", patience=2)])




def main(args):
    orig_toks, norm_toks = retrieve_tokens(century=args.century)
    vocab_orig = vocab_dict(orig_toks + [UNK, PAD, START, END, TOKSEP])
    vocab_norm = vocab_dict(norm_toks + [START, END, PAD])
    vocab_both = vocab_dict(orig_toks + norm_toks + [UNK, PAD, START, END, TOKSEP])
    train_orig, test_orig, train_norm, test_norm = train_test_split(np.array(orig_toks), np.array(norm_toks),
                                                                    test_size=0.2, random_state=42)

    describe_data(train_orig, train_norm, vocab_orig,
                  batch_generator(train_orig, train_norm, vocab_orig, vocab_norm, args))

    #lstm_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both)
    hybrid_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both)
    #encoder_decoder_system(train_orig, train_norm, test_orig, test_norm, vocab_orig, vocab_norm, vocab_both)



if __name__ == '__main__':
    args = argparser.parse_args()
    print("Arguments: " + str(args))
    main(args)

