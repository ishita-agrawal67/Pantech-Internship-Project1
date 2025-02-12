# Image Recognition using CIFAR10 Dataset
This project is about building a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**. The CIFAR-10 dataset consists of **60,000 color images** of size **32x32 pixels**, divided into **10 different classes** (e.g., airplanes, cars, birds, cats, etc.). The project involves **loading, preprocessing, training, evaluating**, and **visualizing** the performance of the model.

## **1. Load Dataset**
- The dataset used is **CIFAR-10**, which is available in TensorFlow/Keras.
- It is divided into:
  - **Training set**: 50,000 images
  - **Testing set**: 10,000 images
- Each image has a **corresponding label** indicating its class (e.g., dog, truck, etc.).
- The dataset shape confirms it has **RGB images (32x32x3).**

## **2. Data Preprocessing**
- **Reshape Labels:**  
  - The labels are reshaped to be in a **1D format** instead of a column vector.
- **Class Names:**  
  - A list of **10 class names** is defined to interpret the numerical labels.
- **Visualizing Images:**  
  - A function is created to display **sample images** with their class names using Matplotlib.

## **3. Normalizing the Dataset**
- **Pixel values** in the dataset range from **0 to 255** (since they are RGB images).
- To **improve model performance**, the pixel values are scaled between **0 and 1** by **dividing by 255**.
- This normalization helps in:
  - **Faster training**
  - **Better accuracy**
  - **Avoiding large weight updates**

## **4. Building the CNN Model**
A **CNN (Convolutional Neural Network)** is built using Keras with the following layers:

### **Layers Explanation**
1. **Input Layer**  
   - Takes images of size **(32,32,3)**
   
2. **First Convolutional Layer**  
   - **32 filters** of size **3x3**
   - Uses **ReLU activation** (to introduce non-linearity)
   
3. **Max Pooling Layer**  
   - Reduces the **image size by half** (2x2 filter)
   
4. **Second Convolutional Layer**  
   - **64 filters** of size **4x4**
   - Uses **ReLU activation**
   
5. **Second Max Pooling Layer**  
   - Further reduces size (2x2 filter)
   
6. **Flatten Layer**  
   - Converts the **2D feature maps** into a **1D vector**
   
7. **Fully Connected (Dense) Layer**  
   - **34 neurons** with **ReLU activation**
   
8. **Output Layer**  
   - **10 neurons (one for each class)**
   - Uses **Softmax activation** (to output class probabilities)

## **5. Compiling and Training the Model**
- The model is compiled using:
  - **Adam optimizer**: Adjusts learning rates automatically.
  - **Sparse categorical cross-entropy loss**: Suitable for multi-class classification.
  - **Accuracy as a metric**.
  
- The model is trained for **5 epochs**, where:
  - **Epoch 1:** Initial accuracy is **~33%**
  - **Epoch 5:** Final accuracy reaches **~67%**
  
- **Validation accuracy (on test set) is ~67%**, showing that the model generalizes decently.

## **6. Making Predictions**
- Predictions are made on the **test set**.
- The **argmax function** is used to find the predicted class for each image.
- A **sample test image** is displayed along with its predicted label.

## **7. Evaluating the Model**
- The model is evaluated on the test set, yielding:
  - **Loss:** ~0.95
  - **Accuracy:** ~67%

- A **classification report** is generated, displaying:
  - **Precision, Recall, and F1-score** for each class.
  - The model performs well on some classes but struggles on others.

## **8. Confusion Matrix & Visualization**
- The **confusion matrix** compares true labels vs. predicted labels.
- A **heatmap** visualization is created to:
  - Identify misclassified images.
  - Show which classes are confused with each other.

## Conclusion
This project successfully: Loads the CIFAR-10 dataset.
- Normalizes the dataset for better performance.
- Builds a CNN model with two Conv2D layers and two pooling layers.
- Trains the model for 5 epochs, achieving 67% accuracy.
- Makes predictions and evaluates performance using classification reports and a confusion matrix.
- Visualizes misclassifications using heatmaps.

## Potential Improvements
- Train for more epochs (e.g., 20-50).
- Use data augmentation to increase dataset diversity.
- Increase model complexity (more Conv2D layers, dropout, batch normalization).
- Experiment with different optimizers (SGD, RMSprop).
