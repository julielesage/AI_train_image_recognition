from imageai.Classification.Custom import CustomImageClassification
import os

# creates a variable which holds the reference to the path that contains your python file
# (in this example, your FirstCustomImageRecognition.py)
# and the ResNet50 model file you downloaded or trained yourself.
execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
# best model copied name
prediction.setModelPath("idenprof_061-0.7933.h5")
prediction.setJsonPath("idenprof_model_class.json")
prediction.loadModel(num_objects=10)

# launch with the image in the folder to try
predictions, probabilities = prediction.predictImage("image.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)