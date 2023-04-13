# Project 5 - Traffic

> CS50AI exercise link: <https://cs50.harvard.edu/ai/2020/projects/5/traffic/>

My solution for the Traffic project of the CS50AI course. The data and the signatures of the functions in the traffic.py file were provided by the CS50AI staff, so my goal was to design a neural network able to classify 43 different traffic signs using TensorFlow.

| :placard: Vitrine.Dev |     |
| -------------  | --- |
| :sparkles: Nome        | **Traffic**
| :label: Tecnologias | Python, TensorFlow, OpenCV

![The result of the project](https://github.com/BrenoMorim/traffic/blob/main/project-image.png?raw=true#vitrinedev)

## Classification of traffic signs

The goal of this project is to correctly classify 43 categories of road signs, using the TensorFlow module to implement a convolutional neural network and the OpenCV module to read the images. I built the neural network using two Convolutional-MaxPooling layers, followed by a hidden layer of size 256, which improved the accuracy in comparison to the other sizes. I noticed that the model was overfitting a lot, during the train stage the model reached almost 90% of accuracy, which dropped to about 30% in the test evaluation. I tried to increase the size of the input images, but the difference in accuracy wasn't satisfying enough to make up for the increase in the time needed to train the model.

To avoid overfitting, I put one Batch Normalization layer between each of the two Conv2D - MaxPooling2D layers, which helped the training process to be more optimized and reduce the overfit. Moreover, I added a Dropout layer after the Hidden layer to decrease even more the overfitting, with a dropout rate of 50%. After adding these layers, the accuracy during the training process was taking longer to increase and didn't get really high, about 60%, however the test evaluation showed about 50% of accuracy as well, which is more consistent with the training. Also, in order to compensate the decrease in the rate with which the accuracy was rising, I increased the number of epochs to 25.

## Structure of the neural network

- Conv2D, relu with 32 3x3 filters, input of size (35, 35, 3)
- BatchNormalization
- MaxPooling2D 2x2
- Conv2D, relu with 64 3x3 filters
- BatchNormalization
- MaxPooling2D 2x2
- Flatten
- Hidden layer of size 256 and relu activation
- Dropout with 50% of dropout rate
- Softmax output with size 43
- Adam optimizer
- Cross entropy loss

### Data used

The data used for this project comes from the German Traffic Sign Recognition Benchmark: A multi-class classification competition. However, through the .gitignore file I reduced the amount of data sent to GitHub, using only 20% of all the images, to avoid excessive consumption of memory.

## Try it yourself

```sh
git clone https://github.com/BrenoMorim/traffic.git traffic
cd traffic
virtualenv .venv
source ./.venv/bin/activate
pip install -r requirements.txt
python traffic.py gtsrb model.h5
```

The usage is **python traffic.py (folder with the data) (name of the file to save the model)**. After training, you can use the following command to test the model:

```sh
python test.py model.h5 test_image.ppm
```

If everything went right, the output should be 2, which is the category of this sign.

---
