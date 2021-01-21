import time
from typing import *
from random import randint
import numpy as np
import torchtext
from torchtext import data

from utils import tensor2text


class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(
            iter(self.pos_iter), iter(self.neg_iter)
        ):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text


# Domain Dataset Iterator
class DomainDatasetIterator(object):
    def __init__(self, domain_iter: DatasetIterator, domain: str):
        self._domain_iter = domain_iter
        self._domain = domain

    def __iter__(self):
        for batch in iter(self._domain_iter):
            yield batch, self._domain


class MultiDomainDatasetIterator(object):
    def __init__(self, domain_iters: List[DomainDatasetIterator]):
        self._domain_iters = domain_iters
        self.iter_cnt = len(domain_iters)

    def __iter__(self):
        domain_iters = [iter(x) for x in self._domain_iters]
        while self.iter_cnt > 0:
            iter_idx = randint(0, self.iter_cnt-1)
            batch, domain = next(domain_iters[iter_idx])
            yield batch, domain
        #     yield_flg = False
        #     try:
        #         batch, domain =next(self._domain_iters[iter_idx])
        #         yield_flg = True
        #     except StopIteration:
        #         self._domain_iters.pop(iter_idx)
        #         self.iter_cnt -= 1
        #     finally:
        #         if yield_flg:
        #             yield batch, domain
        # raise StopIteration


# TODO: Implement MamlDataIterator


def load_dataset(
    config,
    train_pos="train.pos",
    train_neg="train.neg",
    dev_pos="dev.pos",
    dev_neg="dev.neg",
    test_pos="test.pos",
    test_neg="test.neg",
):

    root = config.data_path
    TEXT = data.Field(batch_first=True, eos_token="<eos>")

    dataset_fn = lambda name: data.TabularDataset(
        path=root + name, format="tsv", fields=[("text", TEXT)]
    )

    train_pos_set, train_neg_set = map(dataset_fn, [train_pos, train_neg])
    dev_pos_set, dev_neg_set = map(dataset_fn, [dev_pos, dev_neg])
    test_pos_set, test_neg_set = map(dataset_fn, [test_pos, test_neg])

    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=config.min_freq)

    if config.load_pretrained_embed:
        start = time.time()

        vectors = torchtext.vocab.GloVe(
            "6B", dim=config.embed_size, cache=config.pretrained_embed_path
        )
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print("vectors", TEXT.vocab.vectors.size())

        print("load embedding took {:.2f} s.".format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device,
    )

    train_pos_iter, train_neg_iter = map(
        lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set]
    )
    dev_pos_iter, dev_neg_iter = map(
        lambda x: dataiter_fn(x, False), [dev_pos_set, dev_neg_set]
    )
    test_pos_iter, test_neg_iter = map(
        lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set]
    )

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    dev_iters = DatasetIterator(dev_pos_iter, dev_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)

    return train_iters, dev_iters, test_iters, vocab


# TODO: Add meta (sub task abstract and ratio control)
# TODO: add load domain dataset
def load_multi_domain_dataset(
    config,
    train_pos="train.pos",
    train_neg="train.neg",
    dev_pos="dev.pos",
    dev_neg="dev.neg",
    test_pos="test.pos",
    test_neg="test.neg",
):
    yelp_root = config.yelp_data_path
    imdb_root = config.imdb_data_path
    TEXT = data.Field(batch_first=True, eos_token="<eos>")

    dataset_fn = lambda root, name: data.TabularDataset(
        path=root + name, format="tsv", fields=[("text", TEXT)]
    )
    yelp_dataset_fn = lambda name: dataset_fn(yelp_root, name)
    imdb_dataset_fn = lambda name: dataset_fn(imdb_root, name)

    yelp_train_pos_set, yelp_train_neg_set = map(
        yelp_dataset_fn, [train_pos, train_neg]
    )
    yelp_dev_pos_set, yelp_dev_neg_set = map(
        yelp_dataset_fn, [dev_pos, dev_neg]
    )
    yelp_test_pos_set, yelp_test_neg_set = map(
        yelp_dataset_fn, [test_pos, test_neg]
    )

    imdb_train_pos_set, imdb_train_neg_set = map(
        imdb_dataset_fn, [train_pos, train_neg]
    )
    imdb_dev_pos_set, imdb_dev_neg_set = map(
        imdb_dataset_fn, [dev_pos, dev_neg]
    )
    imdb_test_pos_set, imdb_test_neg_set = map(
        imdb_dataset_fn, [test_pos, test_neg]
    )

    TEXT.build_vocab(
        yelp_train_pos_set,
        yelp_train_neg_set,
        imdb_train_pos_set,
        imdb_train_neg_set,
        min_freq=config.min_freq,
    )

    if config.load_pretrained_embed:
        start = time.time()

        vectors = torchtext.vocab.GloVe(
            "6B", dim=config.embed_size, cache=config.pretrained_embed_path
        )
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print("vectors", TEXT.vocab.vectors.size())

        print("load embedding took {:.2f} s.".format(time.time() - start))

    vocab = TEXT.vocab

    dataiter_fn = lambda dataset, train: data.BucketIterator(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=train,
        repeat=train,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        device=config.device,
    )

    yelp_train_pos_iter, yelp_train_neg_iter = map(
        lambda x: dataiter_fn(x, True), [yelp_train_pos_set, yelp_train_neg_set]
    )
    yelp_dev_pos_iter, yelp_dev_neg_iter = map(
        lambda x: dataiter_fn(x, False), [yelp_dev_pos_set, yelp_dev_neg_set]
    )
    yelp_test_pos_iter, yelp_test_neg_iter = map(
        lambda x: dataiter_fn(x, False), [yelp_test_pos_set, yelp_test_neg_set]
    )

    yelp_train_iters = DomainDatasetIterator(
        DatasetIterator(yelp_train_pos_iter, yelp_train_neg_iter), "yelp"
    )
    yelp_dev_iters = DomainDatasetIterator(
        DatasetIterator(yelp_dev_pos_iter, yelp_dev_neg_iter), "yelp"
    )
    yelp_test_iters = DomainDatasetIterator(
        DatasetIterator(yelp_test_pos_iter, yelp_test_neg_iter), "yelp"
    )

    imdb_train_pos_iter, imdb_train_neg_iter = map(
        lambda x: dataiter_fn(x, True), [imdb_train_pos_set, imdb_train_neg_set]
    )
    imdb_dev_pos_iter, imdb_dev_neg_iter = map(
        lambda x: dataiter_fn(x, False), [imdb_dev_pos_set, imdb_dev_neg_set]
    )
    imdb_test_pos_iter, imdb_test_neg_iter = map(
        lambda x: dataiter_fn(x, False), [imdb_test_pos_set, imdb_test_neg_set]
    )

    imdb_train_iters = DomainDatasetIterator(
        DatasetIterator(imdb_train_pos_iter, imdb_train_neg_iter), "imdb"
    )
    imdb_dev_iters = DomainDatasetIterator(
        DatasetIterator(imdb_dev_pos_iter, imdb_dev_neg_iter), "imdb"
    )
    imdb_test_iters = DomainDatasetIterator(
        DatasetIterator(imdb_test_pos_iter, imdb_test_neg_iter), "imdb"
    )

    train_iters = MultiDomainDatasetIterator(
        (imdb_train_iters, yelp_train_iters)
    )
    dev_iters = MultiDomainDatasetIterator((imdb_dev_iters, yelp_dev_iters))
    test_iters = MultiDomainDatasetIterator((imdb_test_iters, yelp_test_iters))

    return train_iters, dev_iters, yelp_test_iters, vocab


if __name__ == "__main__":
    train_iter, _, _, vocab = load_dataset("../data/yelp/")
    print(len(vocab))
    for batch in train_iter:
        text = tensor2text(vocab, batch.text)
        print("\n".join(text))
        print(batch.label)
        break
