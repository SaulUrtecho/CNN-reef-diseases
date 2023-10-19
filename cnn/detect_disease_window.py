import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.initializers import glorot_uniform
from tkinter import Tk, Label, Button
import re
from logic_window_manager import LogicWindowManager
from os import path

# Image size used
width, height = 256, 256

# We define all the paths
itmLogoPath = "./cnn/logo.jpg"
modelPath = "/home/saul/Documents/CNN-reef-diseases/cnn/model/model.h5"
weightsPath = "/home/saul/Documents/CNN-reef-diseases/cnn/model/weights.h5"
testImagesPath = "/home/saul/Documents/CNN-reef-diseases/cnn/test_images"
count = 0  # With this count we manage the elimination of select another image button


class DetectDiseaseWindow(LogicWindowManager):
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.mainWindow.title("Detection of diseases in marine corals")
        self.mainWindow.geometry("600x500+450+100")
        self.mainWindow.resizable(False, False)
        self.selectImageLabel = Label(self.mainWindow, text="Select image to evaluate: ")
        self.selectImageLabel.pack()
        self.selectImageButton = Button(self.mainWindow, text="Select image", command=self.selectImage)
        self.selectImageButton.pack()
        self.configuredLogo = super().configureImage(itmLogoPath)
        self.itmLogoWindowImage = Label(self.mainWindow, image=self.configuredLogo)
        self.itmLogoWindowImage.pack(side="bottom")
        self.mainWindow.mainloop()

    # Select image method
    def selectImage(self):
        global count
        mainWindowImage = True
        self.selectedImage = super().openFileDialog()
        splitImage = path.split(self.selectedImage)
        if self.selectedImage is not None:
            if splitImage[0] != testImagesPath:  # we evaluate if the given path is different to default
                super().alertWindow()
            else:
                # we evaluate that the selected file is an image
                if re.search("\.(jpg|jpeg|JPG|png|bmp|tiff)$", self.selectedImage):
                    super().deleteWidgets(self.selectImageLabel, self.selectImageButton, self.itmLogoWindowImage)
                    # this sentence manage the lifecycle of the select-other-image-button
                    if count > 0:
                        super().deleteWidgets(self.selectAnotherImageButton)
                    self.configuredImage = super().configureImage(self.selectedImage)
                    self.makePredictionButton = Button(self.mainWindow,text="Detect health status",command=self.makePrediction)
                    self.makePredictionButton.pack()

                    self.selectAnotherImageButton = Button(self.mainWindow,text="Select another image",command=self.selectAnotherImage)
                    self.selectAnotherImageButton.pack()
                    if mainWindowImage is True:
                        self.mainWindowImage = Label(self.mainWindow, image=self.configuredImage)
                        self.mainWindowImage.pack(side="bottom")
                # if the given file is not a valid image format throw a alert
                else:
                    super().alertWindow()

    # Method to select another image
    def selectAnotherImage(self):
        global count
        count = count + 1  # when we select the image the count is incremented by 1 each time
        super().deleteWidgets(self.mainWindowImage, self.makePredictionButton)
        self.selectImage()

    # Method to make a prediction
    def makePrediction(self):
        secondaryWindowImage = True
        super().deleteWidgets(self.makePredictionButton, self.selectAnotherImageButton, self.mainWindowImage)
        # initializing weights using GlorotUniform
        with tf.keras.utils.custom_object_scope({"GlorotUniform": glorot_uniform()}):
            cnn = load_model(modelPath)
        cnn.load_weights(weightsPath)
        image = load_img(self.selectedImage, target_size=(width, height), color_mode="grayscale")
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # we add 1 more channel in the index 0 (1, 256, 256, 1)
        answer = cnn.predict(image)
        result = answer[0]
        result = np.argmax(result)
        if result == 0:
            print(result)
            super().makePrediction("Number five detected","The fifth number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 1:
            print(result)
            super().makePrediction("Number four detected","The fourth number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 2:
            print(result)
            super().makePrediction("None", "there is no number", secondaryWindowImage, self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 3:
            print(result)
            super().makePrediction("Number one detected","The first number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 4:
            print(result)
            super().makePrediction("Number three detected","The third number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 5:
            print(result)
            super().makePrediction("Number two detected","The second number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.mainWindow, text="New detection", command=self.cleanMainWindow)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.mainWindow, text="Exit", command=self.closeMainWindow)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        return result

    def cleanMainWindow(self):
        super().deleteWidgets(self.newDetectionButton, self.closeButton)
        super().insertWidgets(self.selectImageLabel, self.selectImageButton)

    def closeMainWindow(self):
        self.mainWindow.destroy()

# We create an instance of our class
if __name__ == "__main__":
    DetectDiseaseWindow(Tk())
