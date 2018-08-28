linear-attention

1. Build Transformer with only decoder (language modeler)
2. Train model on dataset and obtain negative log probability on test set
3. Amend Transformer with linear attention
4. Train and compare negative log likelihood

Notes:

Go through this [paper](https://arxiv.org/abs/1808.04444) by Al-Rfou et al. (2018) on character-level language modeling with very deep (64-layer) Transformers. Dataset Links: [text8](http://mattmahoney.net/dc/text8.zip) [enwiki8](http://mattmahoney.net/dc/enwik8.zip)

Go through this [paper](https://arxiv.org/abs/1609.05866) (de Brébisson & Vincent, 2016) for a previous attempt at linear attenion.