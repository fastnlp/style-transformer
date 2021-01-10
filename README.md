# Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation

This folder contains the code for the paper [《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》](https://arxiv.org/abs/1905.05621)



## Requirements

pytorch >= 0.4.0

torchtext >= 0.4.0

nltk

fasttext == 0.8.3

kenlm



## Usage

The hyperparameters for the Style Transformer can be found in ''main.py''.

The most of them are listed below:

```
    data_path : the path of the datasets
    log_dir : where to save the logging info
    save_path = where to save the checkpoing
    
    discriminator_method : the type of discriminator ('Multi' or 'Cond')
    min_freq : the minimun frequency for building vocabulary
    max_length : the maximun sentence length 
    embed_size : the dimention of the token embedding
    d_model : the dimention of Transformer d_model parameter
    h : the number of Transformer attention head
    num_layers : the number of Transformer layer
    batch_size : the training batch size
    lr_F : the learning rate for the Style Transformer
    lr_D : the learning rate for the discriminator
    L2 : the L2 norm regularization factor
    iter_D : the number of the discriminator update step pre training interation
    iter_F : the number of the Style Transformer update step pre training interation
    dropout : the dropout factor for the whole model

    log_steps : the number of steps to log model info
    eval_steps : the number of steps to evaluate model info

    slf_factor : the weight factor for the self reconstruction loss
    cyc_factor : the weight factor for the cycle reconstruction loss
    adv_factor : the weight factor for the style controlling loss
```

You can adjust them in the Config class from the ''main.py''.



If you want to run the model, use the command:

```shell
python main.py
```





To evaluation the model, we used Fasttext,  NLTK and KenLM toolkit to evaluate the style control, content preservation and fluency respectively. The evaluation related files for the Yelp dataset are placed in the ''evaluator'' folder. 

Because the file "ppl_yelp.binary" is too big to upload, we exclude it from the "evaluator" folder. As a result, you can not evaluate the ppl score via evaluator. To solve this problem, you can use the KenLM toolkit to train a language model by yourself or use other script to evaluate it.



## Outputs

Update: You can find the outputs of our model in the "outputs" folder.

# build ppl_yelp.binary

[Reference: KenLM Training](https://github.com/kmario23/KenLM-training)

## Install KenLM dependencies

For Ubuntu:
```bash
$ sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
```

For Mac:
```bash
brew install gcc boost
```

## Install KenLM toolkit

```bash
$ pip install nltk
$ git clone --recursive https://github.com/kpu/kenlm.git
$ cd kenlm
$ mkdir -p build && cd build
$ cmake ..
$ make -j 4
$ python setup.py install
```

## Training a language model

Download punkt from http://www.nltk.org/nltk_data/, the link is https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
```bash
$ wget https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip
$ unzip punkt.zip -d /usr/local/share/nltk_data/tokenizers
$ export NLTK_DATA=/usr/local/share/nltk_data/tokenizers
$ cat data/yelp/dev.* data/yelp/train.* data/yelp/test.* | python kenlm_vocab_preprocess.py | ./kenlm/build/bin/lmplz -o 3 > ppl_yelp.arpa
$ ./kenlm/build/bin/build_binary ppl_yelp.arpa ppl_yelp.binary
```
-o select n-gram, -o 3 is 3-gram lm, it will be better to use 5-gram.
