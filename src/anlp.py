import itertools
import argparse

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer

from parse_spanish import retrieve_tokens, retrieve_century

ORIG_EMBEDDING_DIM = 256
NORM_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0

def main(args):
    reader = Seq2SeqDatasetReader(
        source_tokenizer=CharacterTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('../data/all_centuries_toks.train.tsv')
    validation_dataset = reader.read('../data/all_centuries_toks.dev.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=ORIG_EMBEDDING_DIM)
    # encoder = PytorchSeq2SeqWrapper(
    #     torch.nn.LSTM(ORIG_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    encoder = StackedSelfAttentionEncoder(
        input_dim=ORIG_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        projection_dim=128,
        feedforward_hidden_dim=128,
        num_layers=1,
        num_attention_heads=8)

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 20   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=NORM_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8)
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1)
                      #cuda_device=CUDA_DEVICE)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)

        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', "".join(instance.fields['source_tokens'].tokens))
            print('GOLD:', "".join(instance.fields['target_tokens'].tokens))
            print('PRED:', "".join(predictor.predict_instance(instance)['predicted_tokens']))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Program description.')
    argparser.add_argument('-e', '--epochs', default=5, type=int, help='Number of epochs')
    argparser.add_argument('-b', '--batch-size', default=50, type=int, help='Batch Size')
    argparser.add_argument('-lr', '--learning-rate', default=0.1, type=float, help='Learning rate')
    argparser.add_argument('-do', '--dropout', default=0.3, type=float, help='Dropout rate')
    argparser.add_argument('-em', '--embedding-size', default=100, type=int, help='Embedding dimension size')
    argparser.add_argument('-hs', '--hidden-size', default=10, type=int, help='Hidden layer size')
    argparser.add_argument('-c', '--century', default="all", type=str, help='Century')
    argparser.add_argument('-cn', '--context-n', default=1, type=int, help='Number of toks to look to the left and right')

    args = argparser.parse_args()
    main(args)
