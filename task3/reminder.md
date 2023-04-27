First try efficientnet_b3 with 100 epoches. Appreantly overfit.
Then try 10 epoches. Still no good.
Try embedding with lager model : REGNET_Y_128GF. It's too big for my machine
Change embedding model to REGNET_Y_16GF. It's better then before but still not pass the easy line.
Because I think the model is overfit, so I try to reduce the hidden layer to only one layer. But it doesn't change much.
