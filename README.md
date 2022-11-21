# Yoga_position

&nbsp;
## Yoga position: a deep learning model app that identify yoga position from a photo

&nbsp;


![yoga](https://github.com/SalvatoreRa/Yoga_position/blob/main/DALL%C2%B7E%202022-11-15%2015.55.47%20-%20digital%20art%20of%20a%20humanoide%20android%20doing%20yoga%20in%20a%20park,%20high%20quality,%204k.png?raw=true')

photo by the author using Dall-E
&nbsp;
## Information about

* dataset: [Kaggle](https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset)
* Medium corresponding article: [Medium link](https://medium.com/mlearning-ai/make-an-app-with-streamlit-in-minutes-bec48ee19d67)

This is the code for the tutorial on how to build and to deploy a web app with [streamlit](https://docs.streamlit.io/). Contained here is both the code used for pre-processing the data, training the [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) model (trained with [PyTorch](https://pytorch.org/) ),  building the app, and to deploy on the cloud. Also, here you can find the dataset used.

The app uses an artificial intelligence model to predict the position of yoga in an image. The user can upload an image, press a button, and get the top 5 most likely positions.

The model used is [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html), an artificial intelligence model (originally trained on ImageNet) and repurposed for this tutorial.

The app can be found at this link, you can test yourself:

[app](https://salvatorera-yoga-position-yoga-model-8um8ih.streamlit.app/)

The code is an example of how to develop a [computer vision](https://en.wikipedia.org/wiki/Computer_vision) app for classification. Having used [transfer learning](https://cs231n.github.io/transfer-learning/) in this case, the model is easily reusable for other possible tasks. 


&nbsp;

# License

This project is licensed under the **MIT License** 

