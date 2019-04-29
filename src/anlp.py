import itertools
import argparse
from overrides import overrides

import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.nn import util
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from allennlp.training.metrics import Metric
from allennlp.training.metrics import CategoricalAccuracy

from parse_spanish import retrieve_tokens, retrieve_century

ORIG_EMBEDDING_DIM = 128
NORM_EMBEDDING_DIM = 128
HIDDEN_DIM = 256
if torch.cuda.is_available():
    CUDA_DEVICE = 0
else:
    CUDA_DEVICE = -1

class CharAccuracy(Metric):
    def __init__(self):
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self, predictions, gold_labels, mask=None):
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        batch_size = predictions.size()[0]
        training = len(predictions.size()) == 2

        #if not training:
        #    for j in range(batch_size):
        #        s = ""
        #        for i in range(predictions.size()[2]):
        #            s += vocab.get_token_from_index(predictions[j, 0, i].item(), namespace="target_tokens")
        #        print(s)

        #    for j in range(batch_size):
        #        s = ""
        #        for i in range(gold_labels.size()[1]):
        #            s += vocab.get_token_from_index(gold_labels[j, i].item(), namespace="target_tokens")
        #        print(s)

        # a quirk of SimpleSeq2Seq is that if the longest item in the batch has length
        # n, it produces predictions that are n-1 long. (The reasoning: it's either
        # going to be '@end@' or '@@PADDING@@', so it's not worth producing.) This
        # means, however, that we need to reshape the gold labels by trimming off
        # the last bit of dimension 1. (We're only interested in calculating accuracy
        # for "meaningful" tokens, anyway.)
        if training:
            gold = gold_labels.narrow(1, 0, gold_labels.size()[1] - 1)
            mask = mask.narrow(1, 0, mask.size()[1] - 1)
            pred = predictions.narrow(1, 0, mask.size()[1])
        else:
            # take the best prediction
            pred = predictions.narrow(1, 0, 1).squeeze(1)

            # same as above
            gold = gold_labels.narrow(1, 0, gold_labels.size()[1])
            mask = mask.narrow(1, 0, mask.size()[1])

            # prediction length might not be as long as or as short as the longest
            # gold sequence--make sure they're the same
            if pred.size()[1] >= mask.size()[1]:
                pred = pred.narrow(1, 0, mask.size()[1])
            else:
                len_diff = mask.size()[1] - pred.size()[1]
                pred = torch.cat((pred,
                                  torch.zeros(batch_size,
                                              len_diff,
                                              dtype=torch.long)),
                                 dim=1)

        eqs = mask * gold.eq(pred).long()
        if CUDA_DEVICE > -1:
            eqs = eqs.cuda()
        correct_chars = eqs.sum(dim=1)
        total_chars = mask.sum(dim=1)

        self.correct_count += correct_chars.sum().item()
        self.total_count += total_chars.sum().item()

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.total_count > 0:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0

        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0


class AttentionSeq2Seq(SimpleSeq2Seq):
    def __init__(self, vocab, source_embedder, encoder, max_decoding_steps,
                       target_embedding_dim=None,
                       target_namespace=None,
                       attention=None,
                       beam_size=8):
        super().__init__(vocab, source_embedder, encoder, max_decoding_steps,
                         target_embedding_dim=target_embedding_dim,
                         target_namespace=target_namespace,
                         attention=attention,
                         beam_size=beam_size)
        self.accuracy = CharAccuracy()

    def forward(self, source_tokens, target_tokens):
        output_dict = super().forward(source_tokens, target_tokens)
        top_k_predictions = output_dict["predictions"]
        if target_tokens is not None:
            mask = util.get_text_field_mask(target_tokens)
            self.accuracy(top_k_predictions, target_tokens["tokens"], mask=mask)
        return output_dict

    def get_metrics(self, reset=False):
        return {"accuracy": self.accuracy.get_metric(reset)}


def main(args):
    # uncomment if you want to print batches above
    #global vocab
    reader = Seq2SeqDatasetReader(
        source_tokenizer=CharacterTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('../data/all_centuries_toks.train.tsv')
    #train_dataset = reader.read('../data/all_centuries_toks.train.tiny.tsv')
    validation_dataset = reader.read('../data/all_centuries_toks.dev.tsv')
    test_dataset = reader.read('../data/all_centuries_toks.test.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    orig_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
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

    source_embedder = BasicTextFieldEmbedder({"tokens": orig_embedding})

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 20   # TODO: make this variable
    model = AttentionSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
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
                      #patience=5,
                      #num_epochs=100,
                      cuda_device=CUDA_DEVICE)

    for instance in itertools.islice(train_dataset, 5):
        print('SOURCE:', instance.fields['source_tokens'].tokens)
        print('GOLD:', instance.fields['target_tokens'].tokens)

    for i in range(50):
        print('Epoch: {}'.format(i))
        trainer.train()

        #predictor = SimpleSeq2SeqPredictor(model, reader)

        #for instance in itertools.islice(validation_dataset, 10):
        #    print('SOURCE:', (instance.fields['source_tokens'].tokens))
        #    print('GOLD:', (instance.fields['target_tokens'].tokens))
        #    print('PRED:', (predictor.predict_instance(instance)['predicted_tokens']))


    predictor = SimpleSeq2SeqPredictor(model, reader)


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
