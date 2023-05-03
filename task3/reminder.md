First try efficientnet_b3 with 100 epoches. Appreantly overfit.
Then try 10 epoches. Still no good.
Try embedding with lager model : REGNET_Y_128GF. It's too big for my machine
Change embedding model to REGNET_Y_16GF. It's better then before but still not pass the easy line.
Because I think the model is overfit, so I try to reduce the hidden layer to only one layer. But it doesn't change much.
Then I found out I shouldn't use the pretrained model's weight.transforms. Instead, I write according the model's pytorch website.

So the final result is using REGNET_Y_16GF to embedding the images. Then it uses a 4-layer neural network. Inside the network, it has dropout and relu to avoid overfit. For the network training, I use learning rate reduce and early stop. For final prediction, I get the test set reverse, and predict the both reverse and non-reverse version and compare the result.