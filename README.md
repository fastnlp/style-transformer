# Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation

This folder contains the code for the paper [《Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation》](https://arxiv.org/abs/1905.05621)



## Requirements

pytorch >= 0.4.0

torchtext >= 0.4.0

nltk

fasttext == 0.8.3 [Using Future Versions Can Cause Errors such as ```ValueError: /PATH/evaluator/acc_yelp.bin has wrong file format!```](https://github.com/fastnlp/style-transformer/issues/2)

kenlm This step might cause errors so might want to check this [fork](https://github.com/MarvinChung/HW5-TextStyleTransfer) which uses ```pypi-kenlm```

Since TorchText 0.9.0 some functions have become legacy and their imports need to be modified slightly else they would raise attribute not found error. Incase of such errors check out [this answer](https://stackoverflow.com/a/68278366/13858953) 

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

If you wish to get the "ppl_yelp.binary" file you might want to check out this [issue](https://github.com/fastnlp/style-transformer/issues/11) which points to this [fork](https://github.com/MarvinChung/HW5-TextStyleTransfer) which has instructions for getting it


## Outputs

Update: You can find the outputs of our model in the "outputs" folder.
