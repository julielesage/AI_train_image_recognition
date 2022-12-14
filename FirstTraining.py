from imageai.Classification.Custom import ClassificationModelTrainer

# instance of the model training class
model_trainer = ClassificationModelTrainer()

# set the model type to ResNet50 
# (there are four model types available which are MobileNetv2, ResNet50, InceptionV3 and DenseNet121)
model_trainer.setModelTypeAsResNet50()

# set up data file
model_trainer.setDataDirectory("idenprof")

#call the trainModel function and specified the following values:
model_trainer.trainModel(
	# • number_objects : This refers to the number of different types of professionals in the IdenProf dataset.
	num_objects=10,
	# • num_experiments : This is the number of times the model trainer will study all the images in the idenprof dataset
	# in order to achieve maximum accuracy.
	num_experiments=200,
	# • Enhance_data (Optional) : This is to tell the model trainer to create modified copies of the images in the IdenProf dataset
	# to ensure maximum accuracy is achieved.
	enhance_data=True,
	# • batch_size: This refers to the number of images the set that the model trainer will study at once,
	# until it has studied all the images in the IdenProf dataset.
	batch_size=32,
	# • Show_network_summary (Optional) : This is to show the structure of the model type you are using
	# to train the artificial intelligence model.
	show_network_summary=True)

# RUN THIS FILE WITH : python3 FirstTraining.py

# The line Epoch 00000: saving model to C:\Users\User\PycharmProjects\FirstTraining\idenprof\models\model_ex-000_acc-0.100000.h5 refers to the model saved after the present training.
#  The ex_000 represents the experiment at this stage while the acc0.100000 and valacc: 0.1000 represents the accuracy of the model on the test images
#  after the present experiment (maximum value value of accuracy is 1.0). 
# This result helps to know the best performed model you can use for custom image prediction.

# exemple :
# Epoch 1/200
# 281/281 [==============================] - ETA: 0s - loss: 1.9653 - accuracy: 0.3753
# Epoch 1: accuracy improved from -inf to 0.37533, saving model to idenprof/models/model_ex-001_acc-0.375335.h5
# 281/281 [==============================] - 3122s 11s/step - loss: 1.9653 - accuracy: 0.3753 - val_loss: 74.3496 - val_accuracy: 0.1346 - lr: 0.0010
# Epoch 2/200
# 281/281 [==============================] - ETA: 0s - loss: 1.4442 - accuracy: 0.5014

# the accuracy increase >>> 0.75 it becomes good.
# copy the last model 

# Next, create another Python file and give it a name, for example FirstCustomImageRecognition.py .
# Copy the artificial intelligence model you downloaded above or the one you trained that achieved the highest accuracy and paste it to the folder where your new python file (e.g FirstCustomImageRecognition.py ) . Also copy the JSON file you downloaded or was generated by your training and paste it to the same folder as your new python file. Copy a sample image(s) of any professional that fall into the categories in the IdenProf dataset to the same folder as your new python file.