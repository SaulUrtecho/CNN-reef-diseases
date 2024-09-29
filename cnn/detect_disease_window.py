import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.initializers import glorot_uniform
from tkinter import Tk, Label, Button, ttk, Toplevel, Text, INSERT
import tkinter as tk
import re
from logic_window_manager import LogicWindowManager
from os import path
import pandas as pd


# Image size used
width, height = 256, 256

# We define all the paths
itmLogoPath = "./cnn/logo.jpg"
modelPath = "/home/saul/Documents/CNN-reef-diseases/cnn/model/model.h5"
weightsPath = "/home/saul/Documents/CNN-reef-diseases/cnn/model/weights.h5"
testImagesPath = "/home/saul/Documents/CNN-reef-diseases/cnn/test_images"
curvesPath = '/home/saul/Documents/CNN-reef-diseases/cnn/model/results_graph.png'
matrixPath = '/home/saul/Documents/CNN-reef-diseases/cnn/metrics/Confusion_matrix.png'
xlsxPath = '/home/saul/Documents/CNN-reef-diseases/cnn/model/results.xlsx'
textPath = '/home/saul/Documents/CNN-reef-diseases/cnn/metrics/Scores.txt'
count = 0  # With this count we manage the elimination of select another image button


class DetectDiseaseWindow(LogicWindowManager):
    def __init__(self, mainWindow):
        self.mainWindow = mainWindow
        self.mainWindow.title("Detection of diseases in marine corals")
        self.mainWindow.geometry("600x500+450+100")
        self.mainWindow.resizable(False, False)
        style = ttk.Style()
        style.theme_use('clam')
        style.map('TNotebook.Tab', background=[('selected', 'gold1'), ('!selected', 'gold3')])
        self.tabManager = ttk.Notebook(self.mainWindow)
        self.tab1 = tk.Frame(self.tabManager, bg='firebrick4', highlightbackground='gold1', highlightthickness=4)
        self.tab2 = tk.Frame(self.tabManager, bg='firebrick4', highlightbackground='gold1', highlightthickness=4)
        self.tab3 = tk.Frame(self.tabManager, bg='firebrick4', highlightbackground='gold1', highlightthickness=4)
        self.tabManager.add(self.tab1, text='Testing')
        self.tabManager.add(self.tab2, text='Results about model')
        self.tabManager.add(self.tab3, text='About')
        self.tabManager.pack(expand=1, fill="both")
        self.firstTabWidgets()
        self.secondTabWidgets()
        self.thirdTabWidgets()
        self.mainWindow.mainloop()

    # we define the initial widgets for first tab
    def firstTabWidgets(self):
        self.selectImageLabel = Label(self.tab1, text="Select image to evaluate: ", bg='firebrick4')
        self.selectImageLabel.config(fg='white')
        self.selectImageLabel.pack()
        self.selectImageButton = Button(self.tab1, text="Select image", command=self.selectImage, width=20)
        self.selectImageButton.pack()
        self.configuredLogo = super().configureImage(itmLogoPath)
        self.itmLogoWindowImage = Label(self.tab1, image=self.configuredLogo)
        self.itmLogoWindowImage.pack(side="bottom")

    # we define the initial widgets for second tab
    def secondTabWidgets(self):
        self.curvesLabel = Label(self.tab2, text="Loss & accuracy curves: ", bg='firebrick4')
        self.curvesLabel.config(fg='white')
        self.curvesLabel.place(x=50, y=50)
        self.curvesButton = Button(self.tab2, text="Show", command=self.showCurves, width=20)
        self.curvesButton.place(x=50, y=75)
        self.matrixLabel = Label(self.tab2, text="Confusion matrix graph (25 samples used): ", bg='firebrick4')
        self.matrixLabel.config(fg='white')
        self.matrixLabel.place(x=300, y=50)
        self.matrixButton = Button(self.tab2, text="Show", command=self.showConfusionMatrix, width=20)
        self.matrixButton.place(x=360, y=75)
        self.recordsLabel = Label(self.tab2, text="Records for each epoch (.xlsx): ", bg='firebrick4')
        self.recordsLabel.config(fg='white')
        self.recordsLabel.place(x=50, y=150)
        self.recordsButton = Button(self.tab2, text="Show", command=self.showTrainingRecords, width=20)
        self.recordsButton.place(x=50, y=175)
        self.finalResultsLabel = Label(self.tab2, text="Final results: ", font='sans 12 bold', bg='firebrick4')
        self.finalResultsLabel.config(fg='white')
        self.finalResultsLabel.place(x=360, y=150)
        self.finalResultsButton = Button(self.tab2, text="Show", command=self.showFinalResults, width=20)
        self.finalResultsButton.place(x=360, y=175)
        self.logoSecondTab = super().configureImage(itmLogoPath)
        self.itmLogoSecondTab = Label(self.tab2, image=self.logoSecondTab)
        self.itmLogoSecondTab.pack(side="bottom")

    def thirdTabWidgets(self):
        self.recordsLabel = Label(self.tab3, text="Version: 1.0.0\nLanguage: python 3.10.12\nTensorflow: 2.8.0\nKeras: 2.8.0\nOS: Ubuntu 22.04.5 LTS\nCodename: jammy\nDeveloper: saúl urtecho\nemail: saul.urtecho93@gmail.com\ngithub: https://github.com/SaulUrtecho", bg='firebrick4')
        self.recordsLabel.config(fg='white')
        self.recordsLabel.place(x=175, y=150)

    def showCurves(self):
        super().showImagesResult(curvesPath, 'Loss & accuracy curves')

    def showConfusionMatrix(self):
        super().showImagesResult(matrixPath, 'Confusion matrix')

    def showTrainingRecords(self):
        global xlsxPath
        xlsxWindow = Toplevel()
        xlsxWindow.geometry('1100x350')
        xlsxWindow.wm_title('Records per epoch')
        xlsxFrame = tk.Frame(xlsxWindow, pady=35)
        xlsxFrame.pack(pady=20)
        self.treeView = ttk.Treeview(xlsxFrame)
        xlsxPath = r'{}'.format(xlsxPath)
        df = pd.read_excel(xlsxPath)
        self.treeView["column"] = list(df.columns)
        self.treeView["show"] = "headings"
        for col in self.treeView["column"]:
            self.treeView.heading(col, text=col)
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.treeView.insert("", "end", values=row)
        self.treeView.pack()


    def showFinalResults(self):
        global textPath
        resultsWindow = Toplevel()
        resultsWindow.geometry("400x300+300+150")
        resultsWindow.wm_title("Final results")
        resultFrame = tk.Frame(resultsWindow, pady=35)
        resultFrame.pack(pady=20)
        text = Text(resultFrame, width=40, height=12)
        with open(textPath, 'r') as f:
            text.insert(INSERT, f.read())
        text.pack(fill="none", expand=1)

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
                    self.makePredictionButton = Button(self.tab1,text="Detect health status",command=self.makePrediction, width=20)
                    self.makePredictionButton.pack()

                    self.selectAnotherImageButton = Button(self.tab1,text="Select another image",command=self.selectAnotherImage, width=20)
                    self.selectAnotherImageButton.pack()
                    if mainWindowImage is True:
                        self.mainWindowImage = Label(self.tab1, image=self.configuredImage)
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
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 1:
            print(result)
            super().makePrediction("Number four detected","The fourth number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 2:
            print(result)
            super().makePrediction("None", "there is no number", secondaryWindowImage, self.configuredImage)
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 3:
            print(result)
            super().makePrediction("Number one detected","The first number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 4:
            print(result)
            super().makePrediction("Number three detected","The third number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
            self.closeButton.pack()
            super().insertWidgets(self.itmLogoWindowImage)
            self.mainWindow.mainloop()
        elif result == 5:
            print(result)
            super().makePrediction("Number two detected","The second number",secondaryWindowImage,self.configuredImage)
            self.newDetectionButton = Button(self.tab1, text="New detection", command=self.cleanMainWindow, width=20)
            self.newDetectionButton.pack()
            self.closeButton = Button(self.tab1, text="Exit", command=self.closeMainWindow, width=20)
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
    DetectDiseaseWindow(mainWindow=Tk())
