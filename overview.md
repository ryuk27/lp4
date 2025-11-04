# Deep Learning Labs - Overview and Documentation

## 🔬 Lab 1: DL 2.ipynb (ANN for MNIST)

### What It Is
This lab is a "Hello, World!" for Deep Learning. It introduces the Artificial Neural Network (ANN), also known as a Multi-Layer Perceptron (MLP). It's a simple type of neural network that uses fully connected (Dense) layers to learn patterns. We use it for a basic image classification task.

### What We Are Doing
We are building, training, and evaluating a simple ANN to recognize handwritten digits from the MNIST dataset. The MNIST dataset contains 70,000 images (60k for training, 10k for testing) of the digits 0 through 9. Our goal is to create a model that looks at one of these 28x28 pixel images and correctly predicts which digit it is.

### Code Explanation

**Load Data:** `(x_train, y_train), (x_test, y_test) = mnist.load_data()`
- This command loads the MNIST dataset, which is built into TensorFlow/Keras.

**Preprocess Data:** `x_train = x_train / 255`
- The original pixel values range from 0 (black) to 255 (white). We normalize them to a 0-1 range. This helps the model train faster and more stably.

**Define Model:** `model = keras.Sequential([...])`
- `keras.layers.Flatten(input_shape=(28, 28))`: This layer is the input. It "unrolls" or "flattens" the 28x28 pixel grid into a single 1D vector of 784 neurons (28 * 28 = 784).
- `keras.layers.Dense(128, activation="relu")`: This is our one hidden layer. It has 128 neurons. The relu (Rectified Linear Unit) activation function introduces non-linearity, allowing the model to learn complex patterns.
- `keras.layers.Dense(10, activation="softmax")`: This is the output layer. It has 10 neurons, one for each digit (0-9). The softmax function converts the raw output scores into a probability distribution, showing the model's confidence for each class.

**Compile Model:** `model.compile(optimizer="sgd", ...)`
- `optimizer="sgd"`: We use Stochastic Gradient Descent to update the model's weights.
- `loss="sparse_categorical_crossentropy"`: This is the loss function. We use "sparse" categorical cross-entropy because our labels (y_train) are single integers (like 5 or 7).
- `metrics=['accuracy']`: We ask the model to report its accuracy during training.

**Train & Evaluate:** `model.fit(...)` and `model.evaluate(...)`
- `model.fit()` trains the model for 10 epochs (passes through the data).
- `model.evaluate()` checks the final model's performance on the unseen x_test data, which resulted in ~95.2% accuracy.

### ❓ Viva Questions & Answers

1. **Why do we normalize the images by dividing by 255?**
   - **Answer:** Normalization scales pixel values from [0, 255] to [0, 1], which helps the model train faster and more stably. It prevents large input values from causing exploding gradients and ensures all features are on a similar scale, making gradient descent more efficient.

2. **What is the purpose of the Flatten layer? Why can't we feed the 28x28 image directly into a Dense layer?**
   - **Answer:** Dense layers expect 1D input (a vector), but images are 2D (28x28 matrix). The Flatten layer converts the 2D array into a 1D vector of 784 values (28×28=784), making it compatible with Dense layers while preserving all pixel information.

3. **What is the difference between the relu and softmax activation functions? Where are they used?**
   - **Answer:** ReLU (Rectified Linear Unit) outputs max(0, x), introducing non-linearity in hidden layers to help learn complex patterns. Softmax converts raw scores into a probability distribution (summing to 1) and is used in the output layer for multi-class classification to show confidence for each class.

4. **Why did we use sparse_categorical_crossentropy for the loss? What would we use if our y_train labels were one-hot encoded?**
   - **Answer:** We use `sparse_categorical_crossentropy` because our labels are integers (0-9). If labels were one-hot encoded (e.g., [0,0,0,0,1,0,0,0,0,0] for digit 4), we would use `categorical_crossentropy` instead.

5. **Explain what an "epoch" is.**
   - **Answer:** An epoch is one complete pass through the entire training dataset. If we train for 10 epochs, the model sees and learns from every training example 10 times, updating its weights after each batch.

6. **In your output, model.summary() shows the dense layer has 100,480 parameters. How is this number calculated?**
   - **Answer:** (Input neurons × Layer neurons) + Biases = (784 × 128) + 128 = 100,480. Each of the 784 input neurons connects to each of the 128 hidden neurons (784×128 weights), plus 128 bias terms (one per neuron).

---

## 🔬 Lab 2: DL 3.ipynb (CNN for MNIST)

### What It Is
This lab introduces the Convolutional Neural Network (CNN). A CNN is a specialized deep learning architecture that is highly effective for grid-like data, such as images. It uses "convolution" and "pooling" layers to automatically learn spatial hierarchies of features (like edges, textures, and then shapes).

### What We Are Doing
We are solving the same problem as Lab 1 (MNIST classification) but using a CNN instead of a simple ANN. By comparing the results, we can see how much more powerful and efficient CNNs are for image tasks (this model gets ~98.7% accuracy, a significant jump from the ANN's ~95.2%).

### Code Explanation

**Reshape Data:** `X_train = X_train.reshape((X_train.shape + (1,)))`
- Conv2D layers expect a 4D input: (batch, height, width, channels). The original MNIST data is (60000, 28, 28). This line adds the 4th dimension, "channels," which is 1 because the images are grayscale.

**Define Model:** `model = Sequential([...])`
- `Conv2D(32, (3, 3), activation="relu", ...)`: This is the convolutional layer. It applies 32 different filters (also called kernels), each 3x3 in size, across the image. Each filter learns to detect a specific low-level feature (like a vertical edge, a small curve, etc.).
- `MaxPooling2D((2, 2))`: This is the pooling layer. It downsamples the feature map by taking the maximum value from every 2x2 block. This reduces the size of the data, speeds up computation, and helps the model become "translation invariant" (it doesn't care exactly where the feature is, just that it exists).
- `Flatten()`: This flattens the 2D feature map from the pooling layer into a 1D vector so it can be fed into the final Dense layers.
- `Dense(100, "relu") & Dense(10, "softmax")`: These are the same fully connected classifier layers as in the ANN. They take the high-level features learned by the CNN and make the final prediction.

### ❓ Viva Questions & Answers

1. **What is the main difference between this model and the ANN from the previous lab?**
   - **Answer:** The main difference is the use of convolutional and pooling layers instead of just fully connected (Dense) layers. CNNs use Conv2D layers to detect spatial features like edges and patterns, and MaxPooling to reduce dimensions. This preserves spatial relationships in images, making CNNs much more effective for image tasks (98.7% vs 95.2% accuracy).

2. **Why did you have to reshape the input data to (28, 28, 1) for this model?**
   - **Answer:** Conv2D layers require 4D input with shape (batch_size, height, width, channels). MNIST images are grayscale (single channel), so we add a dimension for channels=1, resulting in shape (60000, 28, 28, 1).

3. **What is a "convolution"? What does Conv2D(32, (3, 3)) mean?**
   - **Answer:** Convolution is a mathematical operation where a small filter (kernel) slides across the image, performing element-wise multiplication and summing the results. Conv2D(32, (3, 3)) means we use 32 different 3×3 filters, each learning to detect different features like edges, curves, or textures.

4. **What is "parameter sharing," and why does it make CNNs efficient?**
   - **Answer:** Parameter sharing means the same 3×3 filter (with its 9 weights) is reused across the entire image. This is why the Conv2D layer has only 320 parameters, whereas the ANN's first Dense layer needed over 100,000. It reduces the number of parameters dramatically while capturing spatial patterns effectively.

5. **What is the purpose of the MaxPooling2D layer? What is "translation invariance"?**
   - **Answer:** MaxPooling2D reduces spatial dimensions by taking the maximum value from each 2×2 block, which decreases computation and memory requirements. Translation invariance means the network can recognize a feature regardless of its exact position in the image—if an edge moves slightly, the max pooling operation still captures it.

6. **Look at the model.summary(). Why does the output of the Conv2D layer have a shape of (None, 26, 26, 32)?**
   - **Answer:** A 3×3 filter can't be centered on the edge pixels (without padding), so the 28×28 image shrinks to 26×26 after convolution. The 32 comes from using 32 different filters, creating 32 feature maps.

---

## 🔬 Lab 3: DL 4.ipynb (Autoencoder for Anomaly Detection)

### What It Is
This lab uses an Autoencoder, an unsupervised neural network. An autoencoder is trained to compress its input data into a low-dimensional "bottleneck" (the encoding) and then reconstruct the original data from that encoding (the decoding). If the network is trained only on "normal" data, it becomes very good at reconstructing it, but very bad at reconstructing "abnormal" (anomalous) data.

### What We Are Doing
We are building an autoencoder to detect fraudulent credit card transactions. This is an anomaly detection task. The key idea is to train the autoencoder only on the normal transactions (Class == 0). When the model later sees a fraudulent transaction (Class == 1), its reconstruction will be poor, resulting in a high reconstruction error (Mean Squared Error). We can then set a threshold on this error to flag potential fraud.

### Code Explanation

**Load & Prep Data:** We load creditcard.csv, which is highly imbalanced (many normal transactions, very few frauds).

**Separate Data:** `normal_train_data = train_data[~train_labels]`
- This is the most important step. We create a training set that only contains normal data.

**Define Model (Autoencoder):**
- **Encoder:** The first half of the model compresses the 30 input features down to a bottleneck of 4: Dense(14) → Dense(7) → Dense(4).
- **Decoder:** The second half of the model is a mirror image. It tries to rebuild the original 30 features from the 4-feature encoding: Dense(7) → Dense(14) → Dense(30).

**Train Model:** `autoencoder.fit(normal_train_data, normal_train_data, ...)`
- Note that the input (x) and the target (y) are the same data. The model is learning to reproduce its own input.

**Evaluate:** `test_x_predictions = autoencoder.predict(test_data)`
- We get predictions for the entire test set (which contains both normal and fraud data).
- `mse = np.mean(np.power(test_data - test_x_predictions, 2), axis=1)`: We calculate the reconstruction error (MSE) for every single transaction.

**Detect Anomalies:** `pred_y = [1 if e > threshold_fixed else 0 ...]`
- We set a threshold_fixed. Any transaction with an error above this threshold is classified as fraud (1).
- The confusion matrix shows your model (with threshold_fixed = 52) classified everything as normal. It caught zero fraud.

### ❓ Viva Questions & Answers

1. **What is an Autoencoder? What is it used for?**
   - **Answer:** An autoencoder is an unsupervised neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding). It's used for dimensionality reduction, denoising, anomaly detection, and feature learning.

2. **Why is this an "unsupervised" learning method?**
   - **Answer:** It's unsupervised because we don't need labeled data. The autoencoder learns by trying to reconstruct its own input, using the input data as both the feature (x) and the target (y). It discovers patterns without being told what's "normal" or "fraudulent."

3. **Why did we train the model only on normal transactions?**
   - **Answer:** By training only on normal transactions, the autoencoder learns to reconstruct normal patterns very well. When it encounters fraud (which it has never seen), it can't reconstruct it accurately, resulting in high reconstruction error. This error becomes our anomaly signal.

4. **What is the "bottleneck" of the network?**
   - **Answer:** The bottleneck is the Dense(4) layer in the middle of the autoencoder. It's the smallest layer that compresses 30 input features down to just 4 dimensions, forcing the network to learn the most important patterns. This compressed representation is the "encoding."

5. **What loss function did we use? Why?**
   - **Answer:** We used `mean_squared_error` (MSE) because we're measuring the difference between the input and reconstructed output vectors. MSE calculates the average squared difference between corresponding features, penalizing larger errors more heavily.

6. **How does a high reconstruction error tell us that a transaction might be fraudulent?**
   - **Answer:** The autoencoder was trained only on normal transactions, so it learned to reconstruct normal patterns accurately (low error). When it sees a fraudulent transaction with unusual patterns it hasn't learned, it can't reconstruct it well, resulting in high error. High reconstruction error indicates the transaction is unlike the normal data the model was trained on.

7. **In your results, the Recall and Precision are 0.0. What does this mean, and how would you fix it?**
   - **Answer:** Recall and Precision of 0.0 means the model predicted "Normal" for every single transaction and caught zero frauds. The threshold_fixed (52) was too high. To fix it, lower the threshold to make the model more sensitive to errors, which will flag more transactions as potential fraud and increase recall.

8. **What is the trade-off between Precision and Recall when setting this threshold?**
   - **Answer:** Lower threshold → Higher Recall (catches more fraud) but Lower Precision (more false positives). Higher threshold → Higher Precision (fewer false alarms) but Lower Recall (misses more fraud). The threshold must balance catching fraud while minimizing disruption to legitimate customers.

---

## 🔬 Lab 4: DL 5.ipynb (Word Embeddings - CBOW)

### What It Is
This lab introduces Word Embeddings, a core concept in Natural Language Processing (NLP). Instead of representing words as sparse vectors (like one-hot encoding), embeddings represent them as dense, low-dimensional vectors. Words with similar meanings (e.g., "king" and "queen") will have similar vectors (i.e., they will be "close" to each other in the vector space). This lab implements the Continuous Bag of Words (CBOW) model from scratch.

### What We Are Doing
We are building a simple CBOW model using only NumPy. We take a tiny text corpus, clean it, and create context-target pairs. For example, in the sentence "We are about to study", the context is [we, are, to, study] and the target word is about. The model is trained to predict the target word based on its surrounding context. The "learned" part of this model is the embeddings matrix.

### Code Explanation

**Data Prep:** The text is cleaned, and a vocab (set of all unique words) is created.

**Indexing:** `word_to_ix` and `ix_to_word` are dictionaries that map words to integer indices and back.

**Create Data Bags:** `data.append((context, target))`
- This loop (cell 8) builds our training set. The context window is 2, so it takes two words before (i-2, i-1) and two words after (i+1, i+2) as the input, and the word at i as the target.

**Initialize Weights:**
- **embeddings:** This is the embedding matrix (W1). It's a lookup table where embeddings[i] is the vector for the i-th word. Its shape is (vocab_size, embed_dim).
- **theta:** This is the weight matrix for the output layer (W2). Its shape is (context_window_size * embed_dim, vocab_size).

**forward(context_idxs, theta):**
- `m = embeddings[context_idxs].reshape(1, -1)`: This is the key step. It gets the embedding vectors for all words in the context and concatenates them into one flat vector.
- `n = linear(m, theta)`: A standard linear layer (matrix multiplication).
- `o = log_softmax(n)`: Converts the output into probabilities for each word in the vocab.

**backward(...) & optimize(...):**
- These functions calculate the gradient (error) and update the theta (W2) matrix using gradient descent.
- **Note:** This is a simplified model. A full Word2Vec implementation would also update the embeddings (W1) matrix. This code only trains the output layer theta.

### ❓ Viva Questions & Answers

1. **What is a word embedding? Why is it useful in NLP?**
   - **Answer:** A word embedding is a dense, low-dimensional vector representation of a word (e.g., 50-300 dimensions) that captures semantic meaning. Unlike one-hot encoding which treats all words as equally different, embeddings place similar words close together in vector space. This allows models to understand that "king" and "queen" are related, improving performance on NLP tasks.

2. **What is the difference between CBOW and Skip-gram?**
   - **Answer:** CBOW (Continuous Bag of Words) predicts a target word from its surrounding context words. Skip-gram does the opposite—it predicts context words from a single target word. CBOW is faster and better for frequent words, while Skip-gram works better for rare words and smaller datasets.

3. **Explain the data list you created in cell 8.**
   - **Answer:** The data list contains (context, target) tuples for training CBOW. For each word in the sentence, we create a training example where the context is a list of surrounding words (2 before and 2 after) and the target is the center word. For example: context=[we, are, to, study], target=about.

4. **What is the "context window" in this model?**
   - **Answer:** The context window is size 2, meaning we look at 2 words before and 2 words after the target word. So the total context size is 4 words (2+2). This window size controls how much surrounding text the model considers when learning word meanings.

5. **What are the two weight matrices in this model?**
   - **Answer:** 
     - **embeddings (W1):** The embedding matrix with shape (vocab_size, embed_dim). Each row is the embedding vector for one word.
     - **theta (W2):** The output weight matrix with shape (context_window_size × embed_dim, vocab_size). It transforms the combined context embeddings into predictions for all vocabulary words.

6. **In your code, which of these matrices is actually being trained (updated)?**
   - **Answer:** Only theta (W2) is being trained. The embeddings matrix (W1) is initialized randomly but never updated by the optimize function. A full Word2Vec implementation would update both matrices, allowing the embeddings to learn better representations.

7. **What does the final accuracy() of 1.0 tell you?**
   - **Answer:** An accuracy of 1.0 means the model perfectly memorized the tiny training dataset and can predict every target word correctly from its context. However, this doesn't mean it learned general language—it just overfit to this specific small corpus. It hasn't learned generalizable word meanings.

---

## 🔬 Lab 5: DL 6'.ipynb (Transfer Learning with VGG16)

### What It Is
This lab demonstrates Transfer Learning, one of the most powerful techniques in deep learning. Instead of training a massive model from scratch (which can take days or weeks), we use a pre-trained model that has already been trained on a huge dataset (like ImageNet) by experts.

### What We Are Doing
We are not training a model. We are loading VGG16, a famous 16-layer CNN that won the ImageNet competition, and using it as a "plug-and-play" classifier. We give it three different images (a castle, a valley, and a dog), and it tells us what it thinks they are, based on the 1000 classes it learned from ImageNet.

### Code Explanation

**Load Image:** `image = load_img('download.jpg', target_size=(224, 224))`
- Loads the image and, importantly, resizes it to 224×224 pixels. This is the exact input size VGG16 was trained on.

**Preprocess Image:** `image = preprocess_input(image)`
- This is a critical, model-specific step. The preprocess_input function for VGG16 scales the pixel values and converts the image from RGB to BGR channel order, which is what VGG16 expects.

**Load Model:** `model = VGG16()`
- This downloads and creates the VGG16 model, automatically loading its pre-trained weights from the ImageNet dataset.

**Predict:** `yhat = model.predict(image)`
- This runs the image through the VGG16 network. The output yhat is a vector of 1000 scores, representing the probability for each of the 1000 ImageNet classes.

**Decode Predictions:** `label = decode_predictions(yhat)`
- This is a helper function that takes the 1000-score vector and converts it to a human-readable list of the top predictions.
- `label = label[0][0]` just grabs the single best guess (e.g., ('n09618757', 'castle', 0.3403)).

### ❓ Viva Questions & Answers

1. **What is Transfer Learning?**
   - **Answer:** Transfer Learning is the technique of using a model pre-trained on one task/dataset and applying it to a different but related task. Instead of training from scratch, we leverage knowledge (learned features) from a large dataset like ImageNet. This saves time, computational resources, and often produces better results, especially with limited data.

2. **What is VGG16? What dataset was it trained on?**
   - **Answer:** VGG16 is a 16-layer Convolutional Neural Network developed by the Visual Geometry Group at Oxford. It was trained on ImageNet, a massive dataset containing over 14 million images across 1000 different classes (animals, objects, scenes, etc.). It won the ImageNet competition in 2014.

3. **Why did we have to resize our input images to (224, 224)?**
   - **Answer:** VGG16 was trained with images of size 224×224 pixels. Neural networks expect consistent input dimensions, so we must resize our images to match what the model was trained on. Using different dimensions would cause shape mismatches and errors.

4. **What is the purpose of the preprocess_input function?**
   - **Answer:** The preprocess_input function performs model-specific preprocessing that matches what was done during VGG16's training. For VGG16, this includes subtracting the mean RGB values from ImageNet and converting from RGB to BGR color channel order. Without this preprocessing, predictions would be inaccurate.

5. **In this lab, are we training a model?**
   - **Answer:** No, we are only doing inference (prediction) using a model that is already fully trained. We load VGG16 with pre-trained weights and directly use it to classify new images without any training or weight updates.

6. **The model predicted "golden_retriever (84.78%)". How did it know this?**
   - **Answer:** The model learned to recognize golden retrievers because "golden_retriever" is one of the 1000 classes in the ImageNet dataset it was trained on. During training on millions of images, VGG16 learned the visual features that distinguish golden retrievers (fur color, body shape, facial features) from other dog breeds and objects.

7. **What is "fine-tuning"? How is it different from what we did here?**
   - **Answer:** Fine-tuning is another type of transfer learning where we load a pre-trained model like VGG16, "freeze" most of the early layers (keeping their weights fixed), replace the final output layer with our own custom layer (e.g., 2 classes: cat vs. dog), and then retrain only the new layers on our specific dataset. This adapts the model to our task. In this lab, we used the model as-is without any retraining or modification.
