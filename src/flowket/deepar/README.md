DeepAR
=
This package provide simple & high level API for Neural Autoregressive algorithms.

Currently we provide easy to use keras models of

    WaveNet (one dimension input)
    PixelCNN (two dimensions input)

As well as dynamically creating tensorflow graph for (fast) sampling from any valid keras autoregressive model.

Future Plans
=
We plan to implement

    Image Transformer 
    Sparse Transformer (see https://openai.com/blog/sparse-transformer/)

We also want to support natural gradient & Kfac for this models 