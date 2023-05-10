First, try efficientnet_b3, REGNET_Y_128GF, and REGNET_Y_16GF as embedding models.
REGNET_Y_128GF is too big for my machine.
REGNET_Y_16GF is better than before but still does not pass the easy line.
Because I think the model is overfitting, so I try to reduce the hidden layer to only one layer. But it doesn't change much.
Then I found out I shouldn't use the pre-trained model's weight.transforms. Instead, I write according to the model's pytorch website.

So the decision is to use REGNET_Y_16GF to embed the images. Then it uses a 4-layer neural network. Inside the network, it has dropout and relu to avoid overfitting. For the network training, I use learning rate reduction and early stop. For the final prediction, I get the test set reverse and predict the both reverse and non-reverse version and compare the result.

And I think maybe can use the location information. So I use vstack instead of hstack to concatenate embedding features. And then I can use conv2d inside the neural network. The result is a little better, but I still choose the linear version.
