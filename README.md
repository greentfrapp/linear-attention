linear-attention

1. Build Transformer with only decoder (language modeler)
2. Train model on dataset and obtain negative log probability on test set
3. Amend Transformer with linear attention
4. Train and compare negative log likelihood

Notes:

Go through this [paper](https://arxiv.org/abs/1808.04444) by Al-Rfou et al. (2018) on character-level language modeling with very deep (64-layer) Transformers. Dataset Links: [text8](http://mattmahoney.net/dc/text8.zip) [enwiki8](http://mattmahoney.net/dc/enwik8.zip)

Go through this [paper](https://arxiv.org/abs/1609.05866) (de Brébisson & Vincent, 2016) for a previous attempt at linear attenion. [This](https://arxiv.org/abs/1506.03340) paper gives the dataset for the experiments by Brébisson & Vincent.

Go through this [paper](https://arxiv.org/abs/1610.06258) for an understanding of the slow-fast weights paradigm.

**Notes on fast weights paper:**

> Assuming that the fast weights decay exponentially, we now show that the effect of the fast weights on the hidden vector during an iterative settling phase is to provide an additional input that is proportional to the sum over all recent hidden activity vectors of the scalar product of that recent hidden vector with the current hidden activity vector, with each term in this sum being weighted by the decay rate raised to the power of how long ago that hidden vector occurred.

- Fast weights provide an additional input to the hidden vector
- Fast weights are proportional to a sum over all recent hidden vectors
	- The sum is of the recent hidden vector with the current hidden vector
	- Each term in this sum is weighted by the decay rate proportional to the lifetime of the hidden vector

> So fast weights act like a kind of attention to
the recent past but with the strength of the attention being determined by the scalar product between the current hidden vector and the earlier hidden vector rather than being determined by a separate parameterized computation of the type used in neural machine translation models [Bahdanau et al., 2015].

It is much clearer when we see Equation 1 in the context of Equation 2. The last term in Equation 2 then obviously gives the scalar product between the current hidden vector and decayed representations of the previous hidden vectors.

**Notes on Requests for Research:**

The context given by the Request seems to be to use the Transformer architecture as a way of embedding the history of observations. My guess is that the intention is to set the history of observations as keys/values with the current observation being the query. Given a long history, the Transformer architecture would not be tenable but the Transformer coupled with Linear Attention would be far more tractable.

Also see Section 4.4 of fast weights paper.
