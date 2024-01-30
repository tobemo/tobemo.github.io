# Rocket
> Dempster et al.
> https://arxiv.org/abs/1910.13051
> https://arxiv.org/abs/2012.08791
> https://arxiv.org/abs/2102.00457

Rocket is a model architecture where instead of using a moderate amount of learnable convolution kernels followed by a learnable linear layer 
(i.e. a standard convolution network) a huge amount of non-learnable convolution kernels are used. 
The advantage being that this simplifies computation, only the last linear layer still needs weight updates. 

A second concept rocket introduces is that of pooling in relation to 0.
The kernels previously mentioned are copies of the same set of weights either more diluted and/or shifted via a bias term. 
Said sets of weights are designed to sum to 0 so a large part of Rocket is how the bias term shifts the logits around zero.
They therefore came up with some pooling alternatives like PPV or 'proportion of positive values'. 
These pooling operations are not differentiable (and thus can't backpropagate weight changes) so I had to make some adaptations,
these are in model/pooling/positive_pooling.py.