# Deep Learning Labs - Complete Viva Questions with Answers

## 📚 General Deep Learning Concepts

### 1. What is Deep Learning?
**Answer:** Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to learn hierarchical representations of data. It automatically extracts features from raw data without manual feature engineering, making it highly effective for complex tasks like image recognition, natural language processing, and speech recognition.

### 2. What are the main differences between Machine Learning and Deep Learning?
**Answer:** 
- **Feature Engineering:** ML requires manual feature extraction; DL automatically learns features
- **Data Requirements:** ML works well with smaller datasets; DL requires large amounts of data
- **Hardware:** ML can run on standard CPUs; DL typically needs GPUs for efficient training
- **Interpretability:** ML models are more interpretable; DL models are often "black boxes"
- **Performance:** DL generally outperforms ML on complex, high-dimensional data

---

## 🔬 Assignment 1: Deep Learning Packages

### 3. What are various packages in Python for supporting Deep Learning?
**Answer:** The main Deep Learning packages are:
- **TensorFlow:** Google's comprehensive ML framework
- **Keras:** High-level API (now integrated with TensorFlow)
- **PyTorch:** Facebook's dynamic computation graph framework
- **Theano:** Early symbolic computation library (now discontinued)
- **MXNet, Caffe, CNTK:** Other specialized frameworks

### 4. Compare TensorFlow, Keras, PyTorch, and Theano.
**Answer:**

| Feature | TensorFlow | Keras | PyTorch | Theano |
|---------|-----------|-------|---------|--------|
| Developer | Google | François Chollet | Facebook | MILA (U. Montreal) |
| Computation Graph | Static & Dynamic | Static (via TF) | Dynamic | Static |
| Ease of Use | Moderate | Very Easy | Easy-Moderate | Difficult |
| Production Deployment | Excellent | Good (via TF) | Good (TorchServe) | Limited |
| Community Support | Very Large | Very Large | Very Large | Small (discontinued) |
| Best For | Production, Mobile | Beginners, Prototyping | Research, Flexibility | Legacy projects |

### 5. What is TensorFlow? Explain its key features.
**Answer:** TensorFlow is an open-source machine learning framework developed by Google. Key features include:
- **Flexible Architecture:** Deploy on CPUs, GPUs, TPUs
- **TensorBoard:** Visualization toolkit for monitoring training
- **TensorFlow Lite:** For mobile and embedded devices
- **TensorFlow.js:** ML in browsers
- **Eager Execution:** Immediate operation evaluation
- **Production-Ready:** TensorFlow Serving for deployment
- **Multi-language Support:** Python, C++, Java, JavaScript

### 6. What is Keras? Explain the Keras Ecosystem.
**Answer:** Keras is a high-level neural networks API that provides a user-friendly interface for building deep learning models. **Keras Ecosystem:**
- **Keras Tuner:** Hyperparameter optimization library
- **Keras NLP:** Natural language processing components
- **KerasCV:** Computer vision models and utilities
- **AutoKeras:** Automated machine learning (AutoML)
- **Model Optimization:** Techniques for model compression and quantization

### 7. Explain Sequential Model in Keras.
**Answer:** The Sequential model is a linear stack of layers in Keras. It's the simplest way to build a model where each layer has exactly one input tensor and one output tensor. Layers are added sequentially using `model.add()` or by passing a list to `Sequential()`. Best for simple architectures without branching or multiple inputs/outputs.

### 8. What is PyTorch? What are PyTorch Tensors?
**Answer:** PyTorch is a dynamic deep learning framework developed by Facebook. **PyTorch Tensors** are multi-dimensional arrays similar to NumPy arrays but with GPU acceleration and automatic differentiation capabilities. They are the fundamental data structure in PyTorch, used to encode inputs, outputs, and model parameters. Key features:
- GPU acceleration via `.cuda()`
- Automatic gradient computation via `.backward()`
- Dynamic computation graphs
- Seamless NumPy integration

### 9. What is a Virtual Environment and why is it important?
**Answer:** A virtual environment is an isolated Python environment that allows you to install packages specific to a project without affecting the system-wide Python installation. **Importance:**
- Prevents dependency conflicts between projects
- Ensures reproducibility across different machines
- Allows different Python versions for different projects
- Easy project cleanup by deleting the environment
- Better project dependency management

### 10. What is pip and how is it used?
**Answer:** pip (Pip Installs Packages) is Python's package manager for installing and managing software packages from the Python Package Index (PyPI). Common commands:
- `pip install package_name` - Install a package
- `pip install --upgrade package_name` - Update a package
- `pip uninstall package_name` - Remove a package
- `pip list` - List installed packages
- `pip show package_name` - Show package details
- `pip freeze > requirements.txt` - Export dependencies

---

## 🔬 Assignment 2: Feedforward Neural Networks (ANN/MLP)

### 11. What is an Artificial Neural Network (ANN)?
**Answer:** An ANN is a computational model inspired by biological neural networks. It consists of interconnected nodes (neurons) organized in layers: input layer, one or more hidden layers, and an output layer. Each connection has a weight, and each neuron applies an activation function to its weighted inputs. ANNs learn by adjusting these weights through backpropagation to minimize prediction errors.

### 12. What is a Multi-Layer Perceptron (MLP)?
**Answer:** MLP is a type of feedforward neural network with at least three layers: input, one or more hidden layers, and output. Unlike single-layer perceptrons, MLPs can learn non-linear relationships using activation functions in hidden layers. They use backpropagation for training and are fully connected, meaning every neuron in one layer connects to all neurons in the next layer.

### 13. What is the MNIST dataset?
**Answer:** MNIST (Modified National Institute of Standards and Technology) is a benchmark dataset for handwritten digit recognition containing:
- 70,000 grayscale images (60,000 training, 10,000 testing)
- Each image is 28×28 pixels
- 10 classes representing digits 0-9
- Commonly used for testing image classification algorithms
- Built into Keras/TensorFlow for easy access

### 14. Why do we normalize the images by dividing by 255?
**Answer:** Normalization scales pixel values from [0, 255] to [0, 1], which:
- **Prevents gradient explosion/vanishing:** Large input values can cause unstable gradients
- **Faster convergence:** Normalized inputs help optimization algorithms work more efficiently
- **Equal feature importance:** All features are on the same scale
- **Better numerical stability:** Prevents overflow/underflow in computations
- **Consistent with activation functions:** Many activations (sigmoid, tanh) expect inputs in specific ranges

### 15. What is the purpose of the Flatten layer?
**Answer:** The Flatten layer converts a multi-dimensional input (like a 28×28 image) into a 1D vector without changing the data values. For MNIST, it transforms the 2D (28×28) array into a 784-element vector (28×28=784). This is necessary because Dense (fully connected) layers expect 1D input, and it preserves all pixel information while making it compatible with subsequent layers.

### 16. What is an activation function? Why is it needed?
**Answer:** An activation function introduces non-linearity into the neural network, enabling it to learn complex patterns. Without activation functions, multiple layers would collapse into a single linear transformation, limiting the network to learning only linear relationships. **Purpose:**
- Introduces non-linearity
- Determines if a neuron should "fire" (activate)
- Helps with gradient flow during backpropagation
- Enables learning of complex, non-linear patterns

### 17. Explain different activation functions: ReLU, Sigmoid, Softmax, Tanh.
**Answer:**

**ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`
- Most popular for hidden layers
- Fast computation, helps with vanishing gradient
- Can cause "dead neurons" (always output 0)

**Sigmoid:** `f(x) = 1 / (1 + e^(-x))`
- Output range: [0, 1]
- Used for binary classification
- Suffers from vanishing gradient problem

**Softmax:** `f(xi) = e^(xi) / Σe^(xj)`
- Converts logits to probability distribution (sums to 1)
- Used in multi-class classification output layer
- Ensures all outputs are between 0 and 1

**Tanh:** `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Output range: [-1, 1]
- Zero-centered (better than sigmoid)
- Still suffers from vanishing gradient

### 18. What is the difference between sparse_categorical_crossentropy and categorical_crossentropy?
**Answer:**
- **sparse_categorical_crossentropy:** Used when labels are integers (e.g., 0, 1, 2, 3, 4 for classes)
- **categorical_crossentropy:** Used when labels are one-hot encoded (e.g., [0,0,0,1,0] for class 3)

Both compute the same loss, but differ in label format. Sparse is more memory-efficient as it doesn't require one-hot encoding.

### 19. What is an epoch?
**Answer:** An epoch is one complete pass through the entire training dataset. During one epoch, the model sees every training example once, and weights are updated based on the cumulative error. Multiple epochs are used to train a model, with the number chosen based on when the model converges or starts overfitting. Example: Training for 10 epochs means the model processes the entire dataset 10 times.

### 20. What is a batch and batch size?
**Answer:** A batch is a subset of the training data processed together before updating model weights. **Batch size** is the number of samples in each batch. 
- **Small batch (e.g., 32):** More frequent updates, noisier gradients, better generalization, slower training
- **Large batch (e.g., 256):** Less frequent updates, smoother gradients, faster training, may overfit
- **Full batch:** All training data at once (impractical for large datasets)
- **Mini-batch:** Compromise between stochastic (batch=1) and full batch

### 21. What is Stochastic Gradient Descent (SGD)?
**Answer:** SGD is an optimization algorithm that updates model weights using the gradient of the loss function. Unlike standard gradient descent (which uses all data), SGD updates weights after each batch:
- **Stochastic:** Updates based on one sample
- **Mini-batch SGD:** Updates based on small batches (most common)
- **Advantages:** Faster, can escape local minima, works with large datasets
- **Disadvantages:** Noisy updates, may oscillate around minimum
- **Variants:** SGD with momentum, Adam, RMSprop

### 22. What is backpropagation?
**Answer:** Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight. It works backward through the network using the chain rule of calculus:
1. Forward pass: Input flows through network to produce output
2. Calculate loss between prediction and actual
3. Backward pass: Gradient flows backward, computing partial derivatives
4. Update weights using gradient descent
This process minimizes the loss function iteratively.

### 23. How are parameters calculated in a Dense layer?
**Answer:** Parameters in a Dense layer = (Input neurons × Layer neurons) + Biases

**Example:** Dense layer with 784 inputs and 128 neurons:
- Weights: 784 × 128 = 100,352
- Biases: 128 (one per neuron)
- Total: 100,352 + 128 = 100,480 parameters

Each neuron receives input from all previous layer neurons (fully connected) plus one bias term.

### 24. What is overfitting and how can you prevent it?
**Answer:** **Overfitting** occurs when a model learns training data too well, including noise and outliers, resulting in poor performance on new data. The model has high training accuracy but low validation/test accuracy.

**Prevention techniques:**
- **Dropout:** Randomly deactivate neurons during training
- **L1/L2 Regularization:** Penalize large weights
- **Early Stopping:** Stop training when validation loss increases
- **Data Augmentation:** Create more training examples
- **Reduce Model Complexity:** Fewer layers/neurons
- **Cross-validation:** Better estimate of generalization

### 25. What is the difference between training accuracy and validation accuracy?
**Answer:**
- **Training Accuracy:** Measured on data the model has seen during training. High training accuracy doesn't guarantee good generalization.
- **Validation Accuracy:** Measured on unseen data reserved for validation. Better indicator of model performance on new data.
- **Gap between them:** Large gap indicates overfitting (model memorized training data but can't generalize)

### 26. What is model.compile() in Keras?
**Answer:** `model.compile()` configures the learning process before training. It specifies:
- **Optimizer:** Algorithm to update weights (e.g., 'adam', 'sgd')
- **Loss function:** Metric to minimize (e.g., 'sparse_categorical_crossentropy')
- **Metrics:** Additional measures to monitor (e.g., 'accuracy')

Example: `model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])`

### 27. What does model.fit() do?
**Answer:** `model.fit()` trains the neural network on training data. It:
- Feeds data through the network in batches
- Computes loss for each batch
- Performs backpropagation
- Updates weights using the optimizer
- Repeats for specified number of epochs
- Returns training history (loss, accuracy per epoch)

**Parameters:** training data, labels, epochs, batch_size, validation_data, callbacks

### 28. What is the purpose of model.evaluate()?
**Answer:** `model.evaluate()` assesses the trained model's performance on test/validation data that wasn't used during training. It:
- Returns loss and metrics (e.g., accuracy)
- Provides unbiased performance estimate
- Helps determine if model generalizes well
- Runs forward pass only (no weight updates)
- Essential for comparing different models

---

## 🔬 Assignment 3: Convolutional Neural Networks (CNN)

### 29. What is a Convolutional Neural Network (CNN)?
**Answer:** CNN is a specialized deep learning architecture designed for processing grid-like data (images, videos). It uses convolutional layers to automatically learn spatial hierarchies of features through:
- **Convolution:** Applies filters to detect features (edges, textures, shapes)
- **Pooling:** Reduces spatial dimensions while preserving important features
- **Parameter Sharing:** Same filter applied across entire image
- **Local Connectivity:** Neurons connect to small regions, not entire image
- **Translation Invariance:** Detects features regardless of position

### 30. What is the difference between ANN and CNN?
**Answer:**

| Aspect | ANN | CNN |
|--------|-----|-----|
| Architecture | Fully connected layers | Convolutional + Pooling + Dense layers |
| Input | 1D vectors | 2D/3D data (images) |
| Parameters | Very high | Fewer (parameter sharing) |
| Feature Learning | Manual feature extraction | Automatic feature learning |
| Spatial Information | Lost (flattening) | Preserved (convolution) |
| Best For | Tabular data, simple tasks | Images, spatial data |
| Accuracy on Images | Lower (~95% MNIST) | Higher (~99% MNIST) |

### 31. What is a convolution operation?
**Answer:** Convolution is a mathematical operation where a small filter (kernel) slides across an input image, performing element-wise multiplication and summing the results to create a feature map. 

**Example:** 3×3 filter on image:
1. Place filter on top-left of image
2. Multiply corresponding elements
3. Sum all products → single output value
4. Slide filter right by stride
5. Repeat across entire image

This detects specific features like edges, corners, or textures.

### 32. What is a filter/kernel in CNN?
**Answer:** A filter (or kernel) is a small matrix of learnable weights that slides across the input to detect specific features. 
- **Size:** Typically 3×3 or 5×5
- **Depth:** Matches input channels (1 for grayscale, 3 for RGB)
- **Learning:** Values are learned during training
- **Purpose:** Each filter detects one type of feature (e.g., vertical edges)
- **Multiple Filters:** CNN uses many filters to detect various features

### 33. What does Conv2D(32, (3, 3)) mean?
**Answer:**
- **32:** Number of filters (creates 32 feature maps)
- **(3, 3):** Filter size (3×3 kernel)
- Each filter learns to detect a different feature
- Output has 32 channels, one per filter
- More filters = more features detected = more parameters

**Example:** If input is 28×28×1, output might be 26×26×32 (depends on padding/stride)

### 34. Why do we need to reshape data to (28, 28, 1) for CNN?
**Answer:** Conv2D layers expect 4D input: `(batch_size, height, width, channels)`
- MNIST original shape: (60000, 28, 28) - 3D
- Need to add channel dimension: (60000, 28, 28, 1) - 4D
- **1** represents single channel (grayscale)
- RGB images would have 3 channels: (height, width, 3)
- This tells CNN the number of input feature maps

### 35. What is parameter sharing in CNN?
**Answer:** Parameter sharing means the same filter (with its learned weights) is applied across the entire image, rather than learning separate weights for each position. 

**Benefits:**
- **Drastically reduces parameters:** 3×3 filter = 9 weights reused thousands of times
- **Translation invariance:** Same feature detected anywhere in image
- **Efficient learning:** Learn once, apply everywhere
- **Example:** ANN needs 100,000+ parameters; CNN needs only ~300 for first layer

### 36. What is MaxPooling? Why is it used?
**Answer:** MaxPooling reduces spatial dimensions by taking the maximum value from each region (typically 2×2).

**MaxPooling2D(2, 2):**
- Divides input into 2×2 blocks
- Takes maximum value from each block
- Output is half the input size (height/2, width/2)

**Purpose:**
- **Dimension reduction:** Decreases computation
- **Translation invariance:** Small shifts don't change output
- **Feature preservation:** Keeps most important information
- **Overfitting prevention:** Reduces parameters

### 37. What is stride and padding in CNN?
**Answer:**
**Stride:** Number of pixels the filter moves at each step
- Stride=1: Move one pixel at a time (more computation, larger output)
- Stride=2: Move two pixels (less computation, smaller output)

**Padding:** Adding zeros around input border
- **Valid (no padding):** Output shrinks (28×28 → 26×26 with 3×3 filter)
- **Same padding:** Output same size as input (adds zeros to maintain dimensions)
- Purpose: Control output size, preserve edge information

### 38. Why does a 28×28 image become 26×26 after a 3×3 convolution?
**Answer:** Without padding, a 3×3 filter cannot be centered on edge pixels:
- Filter needs one pixel on each side
- Loses 1 pixel on each edge (top, bottom, left, right)
- Formula: Output = Input - Filter + 1
- 28 - 3 + 1 = 26

With 32 filters: output shape is (None, 26, 26, 32)

### 39. What is translation invariance?
**Answer:** Translation invariance means the network can recognize a feature regardless of its position in the image. For example, it can detect a cat whether it's in the top-left or bottom-right corner. 

**Achieved through:**
- Pooling layers (local invariance)
- Convolutional filters (scan entire image)
- Data augmentation (training on shifted images)

This makes CNNs robust to object position changes.

### 40. What is a feature map?
**Answer:** A feature map is the output of applying one filter to an input image or previous layer. 
- Each filter produces one feature map
- Contains activation values showing where features were detected
- Higher values = strong feature presence
- Deeper layers create more abstract feature maps
- **Example:** First layer detects edges, deeper layers detect faces

### 41. What is the Flatten layer in CNN?
**Answer:** Flatten converts multi-dimensional feature maps from convolutional/pooling layers into a 1D vector for input to Dense layers. 

**Example:** If CNN produces 9×9×128 feature maps:
- Flatten reshapes to 1D: 9×9×128 = 10,368 neurons
- This vector feeds into Dense layers for final classification
- No learning happens in Flatten (just reshaping)

### 42. Explain a complete CNN architecture for image classification.
**Answer:** **Typical CNN Architecture:**

```
Input Image (28×28×1)
    ↓
Conv2D(32, 3×3) + ReLU → Feature maps (26×26×32)
    ↓
MaxPooling(2×2) → Reduced size (13×13×32)
    ↓
Conv2D(64, 3×3) + ReLU → More features (11×11×64)
    ↓
MaxPooling(2×2) → Further reduction (5×5×64)
    ↓
Flatten → 1D vector (1600 neurons)
    ↓
Dense(128) + ReLU → Hidden layer
    ↓
Dense(10) + Softmax → Output (class probabilities)
```

**Flow:** Conv layers detect features → Pooling reduces size → Dense layers classify

### 43. What is data augmentation?
**Answer:** Data augmentation artificially increases training data by creating modified versions of existing images. 

**Common techniques:**
- Rotation, flipping, cropping
- Brightness/contrast adjustment
- Scaling, shifting
- Adding noise

**Benefits:**
- Prevents overfitting
- Improves generalization
- Makes model robust to variations
- Especially useful with limited data

---

## 🔬 Assignment 4: Autoencoders for Anomaly Detection

### 44. What is an Autoencoder?
**Answer:** An autoencoder is an unsupervised neural network that learns to compress data into a lower-dimensional representation (encoding) and then reconstruct the original data (decoding). 

**Architecture:**
- **Encoder:** Compresses input to bottleneck (e.g., 30 → 14 → 7 → 4)
- **Bottleneck:** Smallest layer containing compressed representation
- **Decoder:** Reconstructs from bottleneck (4 → 7 → 14 → 30)

**Training:** Input = Target (learns to reproduce its own input)

### 45. Why is Autoencoder called "unsupervised" learning?
**Answer:** Autoencoders are unsupervised because they don't require labeled data. The model learns by trying to reconstruct its own input, using the input data as both the feature (X) and target (Y). No human annotation is needed—the network discovers patterns automatically by learning efficient data representations.

### 46. How are Autoencoders used for anomaly detection?
**Answer:** **Training Strategy:**
1. Train autoencoder ONLY on normal data
2. Model learns to reconstruct normal patterns accurately
3. When given anomalous data, reconstruction fails
4. High reconstruction error = anomaly detected

**Example:** Credit card fraud:
- Train on legitimate transactions
- Fraudulent transactions have high reconstruction error
- Set threshold: error > threshold = fraud

### 47. What is the bottleneck layer?
**Answer:** The bottleneck is the smallest layer in the middle of the autoencoder that contains the compressed (encoded) representation of the input. 

**Example:** Dense(4) in 30 → 14 → 7 → **4** → 7 → 14 → 30
- Forces network to learn most important features
- Lower dimension = more compression
- Too small: loses information
- Too large: no compression benefit
- Acts as information bottleneck forcing efficient encoding

### 48. What is reconstruction error?
**Answer:** Reconstruction error measures how different the autoencoder's output is from its input. Calculated as:

**Mean Squared Error (MSE):** Average squared difference between input and output
```
MSE = mean((input - reconstructed)²)
```

**For anomaly detection:**
- Low error: Normal data (model reconstructed well)
- High error: Anomaly (model couldn't reconstruct)
- Used to set detection threshold

### 49. Why do we train Autoencoder only on normal transactions for fraud detection?
**Answer:** Training only on normal data ensures the autoencoder becomes an "expert" at reconstructing normal patterns but "ignorant" of fraudulent patterns. 

**Reasoning:**
- Fraud data is rare and varied
- Model learns what "normal" looks like
- Anything significantly different = high error = potential fraud
- If trained on both: might learn to reconstruct fraud too, defeating the purpose
- Works well for imbalanced datasets

### 50. What loss function is used for Autoencoders and why?
**Answer:** **Mean Squared Error (MSE)** is most commonly used because:
- Measures reconstruction quality
- Penalizes large errors more heavily (squared term)
- Differentiable (needed for backpropagation)
- Intuitive: smaller MSE = better reconstruction

**Formula:** `MSE = (1/n) * Σ(input - output)²`

Other options: Mean Absolute Error, Binary Crossentropy (for binary data)

### 51. What is a threshold in anomaly detection?
**Answer:** A threshold is a cutoff value for reconstruction error that separates normal from anomalous data.

**Setting threshold:**
- Analyze error distribution on normal data
- Choose based on business needs:
  - Low threshold: High sensitivity, many false positives
  - High threshold: Low sensitivity, fewer false positives
- Example: threshold = 52 means error > 52 → classify as fraud

**Trade-off:** Precision vs Recall

### 52. If Precision and Recall are both 0.0, what does this mean?
**Answer:** This means the model predicted "Normal" for EVERY transaction—it didn't detect a single anomaly.

**Causes:**
- Threshold too high (no errors exceed it)
- Model not trained properly
- Insufficient feature learning

**Solution:**
- Lower the threshold
- Retrain with better architecture
- Try different features/preprocessing
- Analyze error distribution

### 53. What is the trade-off between Precision and Recall?
**Answer:**
**Precision:** Of predicted frauds, how many are actually fraud?
- High precision: Few false alarms, but might miss some fraud

**Recall:** Of actual frauds, how many did we catch?
- High recall: Catch most fraud, but many false alarms

**Trade-off:**
- Lower threshold → High Recall, Low Precision (catch more, more false positives)
- Higher threshold → Low Recall, High Precision (catch less, fewer false positives)

**Balance:** Use F1-score or ROC curve to find optimal threshold

### 54. Why is the credit card dataset called "highly imbalanced"?
**Answer:** Highly imbalanced means extreme disparity between class frequencies. In credit card fraud:
- Normal transactions: 99.8% (thousands)
- Fraudulent transactions: 0.2% (very few)

**Challenges:**
- Model can achieve 99.8% accuracy by predicting everything as normal
- Standard metrics (accuracy) are misleading
- Difficult for supervised learning
- Perfect use case for autoencoders (train on abundant normal data)

---

## 🔬 Assignment 5: Word Embeddings (CBOW)

### 55. What is Natural Language Processing (NLP)?
**Answer:** NLP is a field of AI focused on enabling computers to understand, interpret, and generate human language. 

**Applications:**
- Machine translation (Google Translate)
- Sentiment analysis
- Chatbots, virtual assistants
- Text summarization
- Named entity recognition
- Speech recognition

**Challenges:** Ambiguity, context, grammar, idioms, multiple languages

### 56. What is Word Embedding?
**Answer:** Word embedding represents words as dense, low-dimensional vectors (e.g., 50-300 dimensions) where semantically similar words have similar vectors.

**Example:**
- "king" vector close to "queen" vector
- vector("king") - vector("man") + vector("woman") ≈ vector("queen")

**Advantages over one-hot encoding:**
- Captures semantic meaning
- Reduces dimensionality (10,000 words → 300 dimensions)
- Similar words grouped together
- Enables arithmetic operations on words

### 57. What is one-hot encoding and why is it problematic for NLP?
**Answer:** One-hot encoding represents each word as a sparse vector with 1 at the word's index and 0s elsewhere.

**Example:** Vocabulary = {cat, dog, bird}
- cat = [1, 0, 0]
- dog = [0, 1, 0]
- bird = [0, 0, 1]

**Problems:**
- High dimensionality (vocabulary size)
- No semantic information (all words equally different)
- Sparse (mostly zeros)
- Can't capture word relationships
- Huge memory requirements for large vocabularies

### 58. What is Word2Vec?
**Answer:** Word2Vec is a technique to learn word embeddings from large text corpora using shallow neural networks. Developed by Google (Mikolov et al., 2013).

**Two architectures:**
- **CBOW (Continuous Bag of Words):** Predict target word from context
- **Skip-gram:** Predict context words from target

**Output:** Dense vector representations where similar words have similar vectors

**Training:** Unsupervised learning on large text

### 59. What is CBOW (Continuous Bag of Words)?
**Answer:** CBOW predicts a target word based on its surrounding context words.

**Example:** Sentence: "We are about to study"
- Context: [we, are, to, study]
- Target: about
- Model learns: given context words → predict "about"

**Architecture:**
- Input: One-hot encoded context words
- Embedding layer: Convert to dense vectors
- Average context vectors
- Output layer: Predict target word (softmax)

### 60. What is Skip-gram?
**Answer:** Skip-gram is opposite of CBOW—it predicts context words from a single target word.

**Example:** Target: "about"
- Predict context: we, are, to, study

**Comparison:**
- **CBOW:** Faster, better for frequent words, averages context
- **Skip-gram:** Slower, better for rare words, treats each context separately

**Use case:** Skip-gram preferred for smaller datasets; CBOW for larger datasets

### 61. What is context window size?
**Answer:** Context window size determines how many surrounding words are considered as context for a target word.

**Window size = 2:**
- Consider 2 words before and 2 words after target
- Total context = 4 words

**Example:** "The cat sat on the mat"
- Target: "sat"
- Context (window=2): [cat, on] (if only right context)
- Context (window=2, both sides): [the, cat, on, the]

**Impact:**
- Larger window: Captures broader context, topic-level relationships
- Smaller window: Captures syntax, close word relationships

### 62. Explain the CBOW model architecture.
**Answer:** **CBOW Architecture:**

```
Context Words (one-hot) → [Input Layer]
    ↓
Embedding Matrix (W1) → [Convert to dense vectors]
    ↓
Average Context Vectors → [Merge context]
    ↓
Linear Layer (W2) → [Compute scores]
    ↓
Softmax → [Probability distribution]
    ↓
Predicted Target Word
```

**Two weight matrices:**
- **W1 (Embeddings):** Vocabulary_size × Embedding_dim
- **W2 (Output):** Embedding_dim × Vocabulary_size

**Training:** Adjust weights to predict target from context

### 63. What is a Tokenizer?
**Answer:** A tokenizer breaks text into smaller units (tokens), typically words or subwords.

**Functions:**
- Split text into words
- Assign unique integer ID to each word
- Build vocabulary
- Handle unknown words
- Create word-to-index and index-to-word mappings
