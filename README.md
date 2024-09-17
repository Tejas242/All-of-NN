# All-of-NN (Ongoing) 🧠🔧
A comprehensive collection of neural network implementations from scratch using **Python** and **PyTorch**. This project explores various deep learning architectures across tasks like classification, regression, and generative modeling.

## Features
- Implementations of key deep learning models such as **RNN**, **CNN**, **LeNet**, **ResNet**, and **Transformers**.
- Designed from scratch to enhance understanding of neural network architectures and their performance.
- Modular and extensible for further research and development.

---

## Progress Tracking 📈

| Task                         | Model                          | Implementation   | Link            
|------------------------------|--------------------------------|------------------|-----------------
| **Classification**           | CNN                            | ✅ Done          | [Sign-Language-CNN](https://github.com/Tejas242/Sign-Language-CNN)
| **Classification**           | RNN                            | ❌ Not Started   |
| **Classification**           | ResNet (Transfer Learning)     | ❌ Not Started   | 
| **Regression**               | NN (House Price Prediction)    | ❌ Not Started   |
| **Regression**               | LSTM (Time Series Forecasting) | ❌ Not Started   |
| **Clustering**               | VAE Clustering (MNIST)         | ❌ Not Started   |
| **Generative Models**        | GAN (MNIST Generation)         | ❌ Not Started   |
| **Generative Models**        | Style Transfer CNN             | ❌ Not Started   |
| **NLP**                      | LSTM (Text Generation)         | ❌ Not Started   |
| **NLP**                      | Seq2Seq Translation            | ❌ Not Started   |
| **Reinforcement Learning**   | DQN (CartPole)                 | ❌ Not Started   | 
| **Advanced Topics**          | Transformer (Attention)        | ❌ Not Started   |
| **Advanced Topics**          | Capsule Networks               | ❌ Not Started   | 

## Folder Structure 📂 (Expected)

```bash
All-of-NN/
│
├── README.md
├── LICENSE
├── .gitignore
├── datasets/                          # Datasets 
├── models/                            # Neural network models (from scratch)
│   ├── cnn.py                         # Convolutional Neural Network (CNN)
│   ├── rnn.py                         # Recurrent Neural Network (RNN)
│   ├── resnet.py                      # Transfer Learning with ResNet
│   ├── gan.py                         # Generative Adversarial Network (GAN)
│   ├── lstm.py                        # Long Short-Term Memory (LSTM)
│   └── transformer.py                 # Transformer Architecture
│
├── notebooks/                         # Jupyter Notebooks
│   ├── classification/
│   │   ├── image_classification_cnn.ipynb
│   │   ├── text_classification_rnn.ipynb
│   │   ├── transfer_learning_resnet.ipynb
│   │   └── multiclass_classification_mlp.ipynb
│   │
│   ├── regression/
│   │   ├── house_price_prediction_nn.ipynb
│   │   ├── time_series_forecasting_lstm.ipynb
│   │   └── energy_consumption_prediction_nn.ipynb
│   │
│   ├── clustering/
│   │   ├── autoencoder_clustering.ipynb
│   │   └── vae_clustering_mnist.ipynb
│   │
│   ├── generative_models/
│   │   ├── gan_mnist_generation.ipynb
│   │   ├── style_transfer_cnn.ipynb
│   │   └── vae_image_generation.ipynb
│   │
│   ├── nlp/
│   │   ├── text_generation_lstm.ipynb
│   │   ├── seq2seq_translation.ipynb
│   │   └── word_embeddings_nn.ipynb
│   │
│   ├── reinforcement_learning/
│   │   ├── dqn_cartpole.ipynb
│   │   └── ppo_lunarlander.ipynb
│   │
│   └── advanced_topics/
│       ├── attention_mechanism_transformer.ipynb
│       ├── capsule_networks_classification.ipynb
│       └── neural_style_transfer_gan.ipynb
│
├── utils/                             # Helper functions for data preprocessing, model training
│   ├── data_loader.py
│   ├── metrics.py
│   └── visualization.py
│
└── Extra Notebooks or Files while learning
```
