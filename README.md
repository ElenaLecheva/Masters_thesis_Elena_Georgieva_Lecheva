# Master's thesis 2025

**Malignant Melanoma Diagnosis with Classification Convolutional Neural Networks**

**Elena Georgieva Lecheva** 

Student Number: 202007457

MSc in Business Intelligence, June 2025

Supervisor: Julie Norlander

Aarhus University

Aarhus School of Business and Social Sciences
<p></p>
<br>


## Libraries used
The code script for the thesis uses Python 3.11.11 and the following libraries and their versions:
- 	os
-	random
-	keras 3.8.0 
- tensorflow 2.18.0 
-	pandas 2.2.2
-	numpy 1.26.4
-	PIL 11.1.0
-	matplotlib 3.10.0
<p></p>
<br>

## Code explanation
The overall structure of the code is: import libraries, load and pre-process data, create the model architecture and specify hyperparameters, train the model, plot results and evaluate the model on the validation set, retrain the model on the training and validation data, test the model on the test set.

The following text briefly outlines the main functions and hyperparameters used to pre-process the data, construct, run and evaluate the models as well as the motivation behind the choices.
<p></p>
<br>

### Metadata pre-processing
For the pre-processing of the metadata, the metadata variables are first inspected. Both the age and the sex variable are categorical as age is rounded to 5-year intervals and has 16 or 17 categories (i.e. age groups), and sex has 2 categories (male and female). All age and sex values are valid and there are no errors.<br>
Since the missing values are represented by the string "unknown”, “unknown” is replaced with `nan` so that python can recognize them as missing values.<br>
The imputation of the missing values in age and sex in the training and the validation set is based on the ground truth.<br>
The missing sex values are imputed with "male" if the person has melanoma, and with "female" if the person does not have melanoma.<br>
The missing age values are imputed with the age with the highest number of melanoma cases if the person has melanoma, and with the age with the highest number of benign cases if the person does not have melanoma.<br>
This is done independetly for the training and the validation set.<br>
Since for the training set there are two ages with the same highest number of melanoma cases - age 65 and age 70, the missing values are imputed with the higher age, i.e. 70, based on evidence that the number of cases of melanoma increases with age. For the validation set, the age with the highest number of melanoma cases is also 70.<br>
The ages that have the most benign cases are 45 for the training set and 15 for the validation set.<br>
Since the training data has one more age group than the validation and the test set (in total 17 categories), namely age 5, all ages 5 are relabelled as age 10 to ensure consistency in the metadata across the 3 sets. This can potentially affect the validity of the model, but since there are only 4 people aged 5 (i.e. 4/2000 = 0.2%), this is not expected to significantly affect the result.<br>
Regarding the test set, since we cannot use the labels as we assume this is the unseen data the model will be applied to in production and the test labels are unknown, the missing values in sex and age are simply imputed with their mode in the test set.<br>
In pandas, the `object` data type typically means that the column contains text strings or mixed types (e.g. strings and numbers). ANN models cannot handle string variables, but need numeric or Boolean variables. For this reason, after the imputation, the metadata variables in all 3 datasets are one-hot encoded, which gives 18 dummy variables - 2 for sex (male and female) and 16 for age (age 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85) after age 5 has been relabelled as age 10 in the training set.<br>
The metadata for each set is combined with the ground truth, i.e. the label, and the corresponding image in a dataframe.<br>
For each dataframe, the image path and a .jpg file extension are added to each image. In this way the model can access the images during training and inference.
<p></p>
<br>

### Image branch (CNN)
ResNet50 and ResNet101 are used as base models with the weights pre-trained on ImageNet loaded by specifying `weights = ‘imagenet’`.<br>
To exclude the fully connected layers, we need to select `include_top = False`. This is because only the convolutional base is used to do feature extraction on the input images, which are then fed into a new fully connected classifier. For this reason, the base layers are frozen. However, as the networks are also fine-tuned by allowing the weights of some of its layers to be updated, the last 10 or 7 layers (see section 3.6 Experiments) are unfrozen by setting them to `trainable = True`.<br>
The `preprocess_input` function in Keras is specifically designed to prepare the image data for input into the pre-trained ResNet models. It performs pre-processing operations that match the pre-processing steps used during the training of these models. The function converts the input images from RGB to BGR format and zero-centers each color channel with respect to the ImageNet dataset. However, the function does not perform scaling.<br>
For the head of the model, i.e. the customized fully connected classifier, which is to be trained together with the unfrozen layers, a global average pooling layer is used. The choice of global average pooling over for flattening is because unlike flattening, global average pooling does not keep positional information. We can argue that in the classification of images of skin lesions the positional information is not essential. The only thing that is important is _whether_ the model detects the class of the lesion in the image, and not _where_ the model detects the lesion in the image.<br>
After the global average pooling layer, several densely connected layers are added (see the exact architectures in section 3.6 Experiments).<br>
The shape of the network should be either a tunnel, i.e. the number of neurons in each layer is the same, or a funnel, i.e. the number of neurons in early layers is larger than the one in later layers. However, the shape should not be a bottleneck, i.e. the number of neurons in early layers should not be smaller than the one in later layers. In case of a bottleneck, the model cannot compress all of the information from the data as the representations are too low-dimensional. In our case, the network has a funnel shape.<br>
All layers use a ReLU activation function to remove the linearity in the model.<br>
A dropout layer to control overfitting is also included with a dropout rate of 0.5 or 0.3 (see section 3.6 Experiments).<br>
This network serves as a general-purpose feature extractor and a backbone for the classification task. For this reason, no classification function such as sigmoid or softmax is used in it.
<p></p>
<br>

### Metadata branch (ANN)
The metadata branch is a basic dense ANN, i.e. a Multi-Layer Perceptron (MLP), where the layers are built sequentially. There is no need to explicitly specify `Sequential()` because this model is already defined as such by using Keras' Functional API (`Input`, `Dense` and `Model`).<br>
First, the input layer is defined where the input shape is specified, which is the number of metadata dimensions after one-hot encoding, i.e. 18 (2 for sex and 16 for age).<br>
The model then consists of several fully connected layers (see section 3.6 Experiments). It again has a funnel shape and each layer uses a ReLU activation function. The classification output is not specified yet.
<p></p>
<br>

### Concatenated, i.e. combined, model
The outputs of the CNN and the ANN network are concatenated into a single input.<br>
After the concatenation of the CNN and ANN models and before the final output layer, additional dense layers and dropout regularization layers are defined. The dense layers learn complex patterns from the combined features (the output from both the CNN and ANN models). The number of neurons in each hidden layer should be at least as big as the number of outcomes. Other values that can also be used are determined by raising 2 to the power of some number. This raised value has to be either equal to or bigger than the number of outcomes. For example, for 46 outcomes, the minimum number should be 2^6 = 64 > 46, but bigger numbers such as 2^7 = 128 can also work. In our case, each hidden layer should have at least 2 neurons.<br>
A dropout layer that randomly drops 30% or 50% of the neurons during training is also added (see section 3.6 Experiments).<br>
The output layer is a fully connected layer modified to match the number of training classes in the classification task. Thus, it has 1 neuron and sigmoid activation as the classification is binary.
<p></p>
<br>

### Image data preparation
As a first step, all 2000, 150 and 600 super-pixel images in the training, validation and test set, respectively, are removed so that only the jpg images are left in the directories.<br>
To pre-process the image data, an `ImageDataGenerator` for both the training and validation set is created. Since the `preprocess_input` function does not scale the images, scaling is applied here to both sets. Because the images are represented by pixels with colour values between 0 and 255, when passing the images as input to a CNN, these values should be normalized, i.e. rescaled by diving them by 255, to values in the same range, i.e. between 0 and 1, because having input data in different ranges will cause problems for the network (Goodfellow et al. 2016).<br>
Data augmentation is applied only to the training data because its primary purpose is to artificially increase the size and diversity of the training dataset, which helps the model generalize better to unseen data. The validation set is used to evaluate the model’s performance during training. It should reflect the real-world data distribution so that we can accurately measure how well the model will perform on unseen data. Thus, augmentation is active only during training but not during inference.<br>
The `fill_mode` determines how to fill in pixels that get introduced due to data augmentation transformations (like rotation, zoom or shift). The options include `constant` (pad with a constant value), `wrap` (copy the top part and put it on the bottom and vice versa, and copy the right side and put it on the left and vice versa), `reflect` (fill with a reflection of the image's border), `nearest` (replicate the border). The different fill modes can be seen in the figure below.

![image](https://github.com/user-attachments/assets/0545b6b0-7cf0-411c-9872-c0c8210e7a59)
![image](https://github.com/user-attachments/assets/57085d9a-5640-47a6-822a-0f07af1466ad)
![image](https://github.com/user-attachments/assets/4629b74c-b28d-4cc0-a9dc-ffb4722ceb4e)
![image](https://github.com/user-attachments/assets/4e16693d-6cbe-46df-a3ad-66aac2c091c3)

Figure: fill mode from left to right: constant, wrap, reflect, nearest <p></p> <br>

The `nearest` mode seems to introduce the least distortion in the images, for which reason it is chosen as a fill mode.
<p></p>
<br>

After this, the actual training and validation generators are created using the `flow_from_dataframe` method from `ImageDataGenerator`, specifying the dataframes with the directory paths where the images can be found, the image data as the input (`image_id`), the labels and metadata as the output (`columns[1:]`), the target size of the images and the batch size. Since all images, both within and between the 3 datasets, are of varying size (width and height), they were resized to a common size of 224x224.<br>
The `own_train_generator_func` and `own_validation_generator_func` are custom-defined Python generator functions that yield a tuple of inputs (images and metadata) and targets (the labels).<br>
Then the training and validation datasets are created, which are flat map datasets, to be used for training by using the `from_generator` method, the custom-defined generator functions and the `output_signature` argument, which specifies the shapes and data types of the inputs and output defined by `tf.TensorSpec`. The `output_signature` argument is needed in order to ensure that the output of the custom generator functions matches the expected structure and data types.
<p></p>
<br>

### Loss function
In a dataset with class imbalance, the model can become biased toward predicting the majority class. The `focal_loss` function defines a custom loss function based on the focal loss concept, which is designed to address class imbalance in tasks like binary classification (Lin et al. 2017).<br>
Alpha is a balancing factor to adjust the importance of positive vs. negative classes (the default is 0.25). Gamma is a modulating factor that helps the model focus more on hard-to-classify examples (the default is 2.0). The final focal loss is a combination of the binary cross entropy loss, alpha factor and modulating factor. The loss is averaged across all instances in the batch.<br>
In this thesis, the focal loss is used with its default alpha and gamma values and no further experimentation with these hyperparameters was conducted.<br>
The defined `focal_loss` is used in place of the `binary_crossentropy` loss in the `compile` method.<br>
<p></p>
<br>

### Optimizer and learning rate
The `optimizer` implements a specific variant of the gradient descent (GD) and specifies exactly how the gradient of the loss function will be used to update the parameters. Commonly used optimizers are Root Mean Square Propagation (RMSProp) and Adaptive moment estimation (ADAM).<br>
In this thesis, ADAM is used as an optimizer because the models seem to converge a little bit faster than with RMSprop. Furthermore, ADAM is a variant of gradient descent that combines momentum and adaptive learning rate.<br>
Instead of considering only the current value of the gradients, momentum also takes into account previous weight updates when computing the next update. Thus, momentum represents not only the current slope value (i.e. the current gradient value), but also the current velocity, which comes from past weight updates (i.e. the accumulated past gradients) and prevents optimization from getting stuck in a local minimum.<br>
The adaptive learning rate means that ADAM adapts the learning rate for each parameter based on estimates of the first and second moments of the gradients, which helps in handling different gradients and speeds up convergence.<br>
When fine-tuning a pre-trained model, we want to make smaller updates to the weights so that the representations learnt by the pre-trained model are not destroyed. For this reason, the learning rate is lowered from the default value of 1e-3 to 1e-5 in order to take smaller steps to find the minimum loss. Furthermore, reducing the learning rate makes the validation loss less wiggly and noisy.
<p></p>
<br>

### Training the model
A `callbacks` list with a `ModelCheckpoint` callback is created to save the model with the lowest validation loss so that it can be loaded later to be retrained and used for inference on the test set. The `ModelCheckpoint` callback saves both the model's configuration (i.e. its architecture with the layers the model contains and the connections between them) and its state (i.e. the weights and compilation information).<br>
The model is trained by passing the `train_dataset` and `validation_dataset`, which yield the training and validation inputs and targets.<br>
A relatively high number of epochs is needed in order to make sure we will find the epoch where the model fits the data. The models are trained for 30 or 40 epochs (see section 3.6 Experiments).<br>
The `steps_per_epoch` argument defines how many training steps (i.e. batches) the model should go through in one epoch. The default value is the number of samples in the training dataset divided by the batch size.<br>
Similar to `steps_per_epoch`, `validation_steps` defines how many validation steps (batches) should be run in each epoch during the validation phase. The default value is the number of samples in the validation dataset divided by the batch size.<br>
If `steps_per_epoch` and `validation_steps` are omitted, TensorFlow assumes that the generator will eventually raise a `StopIteration` exception, which happens when the generator naturally ends.<br>
However, for infinite generators, like the custom-defined ones here, this can lead to an infinite training loop because the generator will never stop unless we explicitly define the steps, i.e. it will keep calling `next()` on the generator indefinitely. For this reason, when passing an infinitely repeating dataset, these arguments have to be specified, otherwise the training or validation will run indefinitely.<br>
If the total number of samples in the datasets is not a perfect multiple of the batch size, Keras will still process the dataset in full, but the last batch will contain fewer samples than the specified batch size. For example, if there are 105 samples in the training dataset and a batch size of 32, Keras will process 3 full batches of 32 samples each (96 samples in total), and the last batch will have the remaining 9 samples. The same applies to the `validation_steps`.<br>
However, since the expected shape of the input has already been specified in `output_signature`, which includes the batch size, the `fit` method cannot accept batches of other size. For this reason, the batch size must be the same for all batches and it must be an exact divisor of the training, validation and test set, i.e. of 2000, 150 and 600, so that there is no remainder when the dataset sizes are divided by the batch size. The exact divisors of 2000, 150 and 600 are 1, 2, 5, 10, 25, 50. Selecting a good batch size is a matter of finding the right training configuration. However, we want a batch size big enough so that the model can be exposed to enough samples to learn the data patterns. Besides, bigger batches lead to gradients that are more informative and less noisy, i.e. that have lower variance (Chollet 2021). Based on this, a batch size of 50 was selected. Furthermore, experimentation showed that a batch size of 50 is the optimal one and smaller batch sizes do not improve performance by much.<p></p><br>

The models were also trained with class weights in order to potentially alleviate the effect of class imbalance. To train the model with class weights, the `fit` method together with the same arguments is used, but in this case a `class_weight` argument is added as well with class weights, which are calculated beforehand.
<p></p>
<br>

### Retraining the model
Following the universal workflow of Machine Learning devised by Chollet (2021), when a model configuration with satisfactory performance has been developed, the model can be trained on all the available data (training and validation) and evaluated one final time on the test dataset. Therefore, before being assessed on the test data, the best performing ResNet50 model (Model ResNet50_3) and ResNet101 model (Model ResNet101_6) were trained on all the training and validation data for 30 epochs each.<br>
To do this, the models are first loaded where the `preprocess_input` and the `focal_loss` function have to be explicitly defined. This is because the model contains a `Lambda` layer with a pre-processing function from a library, but Keras cannot find it during loading. The same happens with the custom loss function `focal_loss`, which Keras is unable to find when loading the model. For this reason, these functions need to be explicitly defined as `custom_objects` and passed to `load_model` to give Keras access to the definition of these custom objects.<br>
The number of training steps is defined again because the combined training dataset runs infinitely.<br>
The model is then retrained with the `fit` method again.<br>
In this case, there is no need to compile the models again because the same optimizer, loss and a set of metrics (all defined by previously compiling the model) are used.<br>
The retrained models are saved again in a callbacks list with a `ModelCheckpoint` callback.
<p></p>
<br>

### Test image data
For the test images, again an `ImageDataGenerator` without data augmentation is created. <br>
Then the image generator is specified by using the `flow_from_dataframe` method from `ImageDataGenerator`, specifying the dataframe with the directory paths where the test images can be found, the image data as the input, the labels and metadata as the output, the target size of the images and batch size, the last two being the same as the ones for the training and validation data (224x224 width x height, and 50 images per batch). (For pre-processing of the test metadata, refer back to section **Metadata pre-processing**).<br>
Then the `own_test_generator_func` is created, similar to the other custom generator functions, which is a Python generator function yielding a tuple of inputs (images and metadata) and targets (the labels).<br>
Then the test dataset is specified, which is again a flat map datasets, by using the `from_generator` method, the custom-defined generator function and the `output_signature` argument, which specifies the shapes and data types of the inputs and output defined by `tf.TensorSpec`, which are the same as for the training and validation data.<br>
The model to be used for inference is loaded where the `preprocess_input` and the `focal_loss` function are passed to the model.<br>
For the `evaluate` method, the number of steps (i.e. number of batches) that Keras will use to evaluate the model need to be specified again because the test dataset created is also an infinitely repeating dataset. Since `output_signature` has already specified the expected shape of the input, which includes the batch size of 50 images, the `evaluate` method cannot accept batches of other size and this batch size (50) is used to determine the number of test steps. 
<p></p>
<br>

## References and acknowledgements
The pre-processing of the data, the model architecture, the custom generator functions and the focal loss function are adapted from the following notebook by Narek HM found on Kaggle:<br>
https://www.kaggle.com/code/nhm1440/image-metadata-with-keras-imagedatagenerator
<br>

The class weights are calculated by using a method found in a tutorial about classification on imbalanced data by TensorFlow:<br>
https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
<br>

The training and validation loss and accuracy plots as well as the custom-defined function "print_best_val_metrics" have been borrowed from exercise ML4BI_E5_solution from the course Machine Learning for Business Intelligence 2:<br>
https://colab.research.google.com/drive/1pjX4cPtXvnbjSeasNf0Z4Sl-Ii-jPM0L?usp=sharing
<br>

In addition, the Keras and TensorFlow documentation has been used:<br>
Keras documentation: https://keras.io/api/ <br>
TensorFlow documentation: https://www.tensorflow.org/guide <br>
Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Rafal Jozefowicz, Yangqing Jia, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mané, Mike Schuster, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viégas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org.
