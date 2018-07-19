# RNN-RBM

Implementation of an RNN-RBM for music compostion. The model was inspired by [this](http://www-etud.iro.umontreal.ca/~boulanni/ICML2012.pdf) paper.

The RNN-RBM mixes the capability of the RNN for learning temporal dependecies, with the capability of the RBM for learning a probability distribution over the training dataset (being a generative model). It is comprised of an RNN where each recurrent hidden layer is connected to a different RBM.

![RNN-RBM Architecture](https://cdn-ak.f.st-hatena.com/images/fotolife/n/nkdkccmbr/20161006/20161006222106.png)
In this figure representing an RNN-RBM, `u(t)` represents an RNN layer at time-step `t`, while `h(t)` and `v(t)` represnet the RBM's hidden and visible layers respectively, at time `t`
