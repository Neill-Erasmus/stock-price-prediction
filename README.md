# Stock Price Prediction (Recurrent Neural Network)

This project aims to develop a predictive model for stock prices using Recurrent Neural Networks (RNNs). RNNs are a class of neural networks particularly suited for sequential data, making them well-suited for time series prediction tasks like stock price forecasting. The project involves collecting historical stock price data from financial markets, preprocessing the data to handle missing values and normalize the features, and then training an RNN model on this data.

## Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are a type of artificial neural network designed to process sequential data by capturing patterns over time. Unlike traditional neural networks, RNNs have connections that form loops, allowing information to persist.

In an RNN, information flows through a series of nodes, or neurons, organized into layers. At each step, the network receives an input and produces an output, while also maintaining an internal memory of past information.

During each step:

- The network combines the current input with the previous memory to update its current memory.
- The updated memory is then used to produce the output for that step.

RNNs are trained by comparing their output to the expected output and adjusting the connections between neurons to minimize the difference. This process, known as backpropagation through time, allows the network to learn to make accurate predictions based on sequential data.

To address the issue of vanishing gradients, which can make it difficult for the network to capture long-term dependencies, variants of RNNs such as Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) have been developed. These variants introduce mechanisms to selectively retain or forget information over long time scales, improving the network's ability to capture long-range dependencies.

## Dataset Overview

The dataset used in this project aims to predict stock market indices using a recurrent neural network (RNN) and Long Short-Term Memory (LSTM) approach. It contains the following features:

1. Date: The date of the stock market data.
2. Open: The opening price of the stock on a given day.
3. High: The highest price of the stock reached during the trading day.
4. Low: The lowest price of the stock reached during the trading day.
5. Close: The closing price of the stock on a given day.
6. Volume: The total number of shares traded during the trading day.

These features provide essential information about the behavior of the stock market, such as price movement and trading activity, which are crucial for predicting future stock market indices. By utilizing RNN and LSTM models, researchers and analysts aim to create effective prediction systems that can assist traders, investors, and analysts in making informed decisions about their investments based on the future direction of the stock market.

## Architecture of the Neural Network

In the context of predicting stock market indices, the architecture of the neural network plays a crucial role in learning patterns from historical stock data and making predictions for future stock prices. Let's break down the architecture described:

- LSTM Layers

LSTM stands for Long Short-Term Memory, which is a type of neural network layer specialized in processing sequences of data, like stock prices over time. Here, we have four LSTM layers stacked on top of each other. Each layer looks at the historical stock prices and tries to understand the patterns within them. Think of each layer as learning different aspects of how the stock prices change over time.

- Dropout Layers

Dropout layers help prevent the neural network from memorizing the training data too well, which can lead to overfitting. Overfitting is when the model learns to perform well on the training data but fails to generalize to new, unseen data. The dropout layers randomly "drop out" some connections between neurons during training, forcing the network to learn more robust and generalizable patterns.

- Dense Output Layer

This is the final layer of the neural network. It takes the information learned by the LSTM layers and combines it to produce the predicted stock price for the next time step. It's like the conclusion drawn from analyzing all the historical data – it provides the forecasted stock price based on the patterns learned by the network.

Overall, this architecture is designed to learn from past stock price data, including factors like opening price, highest price, lowest price, closing price, and trading volume. By understanding these patterns, the neural network aims to make accurate predictions about future stock prices, which can be invaluable for traders, investors, and analysts in making informed decisions in the stock market.

## Model Training

```
Epoch 1/150
14/14 [==============================] - 7s 71ms/step - loss: 0.0817
Epoch 2/150
14/14 [==============================] - 1s 78ms/step - loss: 0.0163
Epoch 3/150
14/14 [==============================] - 1s 68ms/step - loss: 0.0101
Epoch 4/150
14/14 [==============================] - 1s 69ms/step - loss: 0.0099
Epoch 5/150
14/14 [==============================] - 1s 74ms/step - loss: 0.0076
Epoch 6/150
14/14 [==============================] - 1s 69ms/step - loss: 0.0078
Epoch 7/150
14/14 [==============================] - 1s 65ms/step - loss: 0.0074
Epoch 8/150
14/14 [==============================] - 1s 69ms/step - loss: 0.0076
Epoch 9/150
14/14 [==============================] - 1s 68ms/step - loss: 0.0070
Epoch 10/150
14/14 [==============================] - 1s 71ms/step - loss: 0.0095
      ................................
Epoch 148/150
14/14 [==============================] - 2s 160ms/step - loss: 0.0022
Epoch 149/150
14/14 [==============================] - 2s 156ms/step - loss: 0.0022
Epoch 150/150
14/14 [==============================] - 2s 153ms/step - loss: 0.0024
```


The training process of the neural network involves several epochs, where each epoch represents one complete pass through the entire training dataset. During each epoch, the neural network adjusts its internal parameters (weights and biases) based on the training data to minimize the loss, which is a measure of how far off the model's predictions are from the actual stock prices.

In this specific training log, we see that the training process lasts for 150 epochs. At the end of each epoch, the loss, which represents the difference between the predicted stock prices and the actual stock prices, is calculated. The goal of training is to minimize this loss.

As the training progresses, we observe a gradual decrease in the loss, indicating that the neural network is learning and improving its ability to predict stock prices over time. This decreasing trend in the loss suggests that the model is becoming more accurate in its predictions.

The training process involves iterative adjustments to the neural network's parameters, guided by optimization algorithms such as stochastic gradient descent (SGD) or Adam. These adjustments aim to find the optimal values for the parameters that result in the lowest possible loss, thereby improving the model's predictive performance.

Overall, the training process of the neural network involves repeatedly feeding the training data through the network, updating its parameters based on the observed errors, and gradually improving its ability to make accurate predictions of stock prices.

<p align="center">
    <img src="">
</p>

### Overfitting

Overfitting occurs when a machine learning model learns the training data too well, capturing noise and specific details that are unique to the training set but may not generalize well to new, unseen data. In essence, the model becomes too tailored to the idiosyncrasies of the training data, resulting in poor performance on validation or test datasets.

A lower model accuracy on the training set is often preferred to avoid overfitting. This might seem counterintuitive, but it signifies that the model is not memorizing the training data but rather learning the underlying patterns and features that are more likely to generalize to new data. A model with slightly lower training accuracy but better generalization capability is preferred because it is more likely to perform well on unseen data, demonstrating its ability to make accurate predictions in real-world scenarios beyond the training set. Regularization techniques, such as dropout and weight regularization, are commonly employed to help prevent overfitting and promote better generalization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.