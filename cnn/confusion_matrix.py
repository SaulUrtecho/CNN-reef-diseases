from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

testPath = "./cnn/data/test"  # It contains 150 images
modelo = load_model("./cnn/model/model.h5")

# We use the class ImageDataGenerator since our model was trained using generators
valGenerated = ImageDataGenerator(rescale=1.0 / 255)

valGenerator = valGenerated.flow_from_directory(
    testPath,
    target_size=(256, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False,
)

predict = modelo.predict(
    valGenerator, verbose=1
)  # This line returns a 2D matrix with the probabilities of each prediction

print("El tamaño del generador es: ", len(predict))  # Show the array's predictions size
print(predict)


resultPredicted = []
for i in predict:
    maxValueindex = np.argmax(i)
    resultPredicted.append(
        maxValueindex
    )  # We obtain the index of the predictions in a 2D array

print(len(resultPredicted))
print(resultPredicted)

# We created a array with the labels [0,1,2,3,4,5] to simulate the yTRUE
real = []

# In the ranges we took the initial value and the final value -1
# and then we use it for evaluate the next iteration through if sentence.
# In each range we must sum the total of images in each folder in this case it's 25
for fila in range(0, 25):
    real.append(0)
    if fila == 24:
        for fila in range(24, 49):
            real.append(1)
            if fila == 48:
                for fila in range(48, 73):
                    real.append(2)
                    if fila == 72:
                        for fila in range(72, 97):
                            real.append(3)
                            if fila == 96:
                                for fila in range(96, 121):
                                    real.append(4)
                                    if fila == 120:
                                        for fila in range(120, 145):
                                            real.append(5)


print(real)


##------------------------ SKLEARN MATRIX------------------------
print("-----------------SKLEARN MATRIX")
print()


cm = confusion_matrix(real, resultPredicted)
print(cm)
clases = ["Five: 0", "Four: 1", "None: 2", "One: 3", "Three: 4", "Two: 5"]
report = classification_report(real, resultPredicted, target_names=clases)
print(report)

print("-----------------CONFUSION MATRIX METRICS (SKLEARN)--------------")
print()
ac = accuracy_score(real, resultPredicted)
print("Accuracy Score: ", ac)
rc = recall_score(real, resultPredicted, average=None)
print("Recall Score: ", rc)
ps = precision_score(real, resultPredicted, average=None)
print("Precision Score", ps)
f1 = f1_score(real, resultPredicted, average=None)
print("F1 Score: ", f1)


######### Saving metrics in a text file ##########
scoresFile = open("./cnn/metrics/Scores.txt", "w")
scoresFile.write("Accuracy Score: " + str(ac) + "\n")
scoresFile.write("\n")
scoresFile.write("Recall Score: " + str(rc) + "\n")
scoresFile.write("\n")
scoresFile.write("Precision Score TP/(TP + FP): " + str(ps) + "\n")
scoresFile.write("\n")
scoresFile.write("F1 Score: " + str(f1))
scoresFile.close()

print("------------- Graphic Matrix -----------------")

plt.get_current_fig_manager().set_window_title("Classification Metrics")
plt.title("Confusion Matrix")
cax = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
colorbar = plt.colorbar(cax)
colorbar.set_label("Samples", rotation=0, labelpad=25)
classNames = ["Five", "Four", "None", "One", "Three", "Two"]
plt.ylabel("True", rotation=0, labelpad=20)
plt.xlabel("Predict", labelpad=10)
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames)
plt.yticks(tick_marks, classNames)
for fila in range(6):
    for columna in range(6):
        plt.text(columna, fila, str(cm[fila][columna]))
plt.savefig("./cnn/metrics/MatrizConfusion.png")
plt.show()
