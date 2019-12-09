Prototype for a model, that trains on the Internet Movie Script Database (IMSDB) and thereby learns to have a chitchat with the user. The idea is to do regression on word embeddings instead of classification on word tokens, so that the output is continous and can thereby be trained with a GAN. The adversarial training is meant to give the model more creativity, so that it won't collapse to answers like "I don't know".

Run "python gan_chitchat_train.py" in order to train and test the model!

Additional information can be found with "python gan_chitchat_train.py --help"

TODO:
Make adversarial training run.
