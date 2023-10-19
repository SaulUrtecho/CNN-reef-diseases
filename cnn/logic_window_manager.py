"""This class help us to manage the logic of the main window"""

import cv2
import tkinter
from tkinter import filedialog as fd
from PIL import ImageTk, Image
from tkinter import Label, Button, Text, Toplevel


class LogicWindowManager:
    # Method that provide us the image configuration
    def configureImage(self, path):
        image = cv2.imread(path)
        # if the image is the same size of the logo.jpg size, then it stays
        if (image.shape[0] and image.shape[1]) == (108 and 468):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
        # if the image is any other size different to logo.jpg
        elif (image.shape[0] and image.shape[1]) != (108 and 468):
            image = cv2.resize(image, (200, 200))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
        return image

    # Method for erase n quantity of widgets than aren't necessary
    def deleteWidgets(self, *args):
        for arg in range(len(args)):
            deletedWidgets = args[arg].pack_forget()
        return deletedWidgets

    # Method for insert n quantity of widgets on GUI
    def insertWidgets(self, *args):
        for item in range(len(args)):
            # if the widget is equals to self.itmLogoWindowImage then it going to GUI bottom
            if args[item] == self.itmLogoWindowImage:
                args[item].pack(side="bottom")
            else:
                args[item].pack()

    # Method for open a window file dialog
    def openFileDialog(self):
        filePath = fd.askopenfilename()
        return filePath

    # Method for throw an alert window if the user select a wrong file
    def alertWindow(self):
        def closeWarningWindow():  # local function for destroy itself
            alertWindowWidget.destroy()

        alertWindowWidget = Toplevel()
        alertWindowWidget.geometry("300x100+500+250")
        alertWindowWidget.wm_title("WARNING!!!")
        alertWindowWidget.focus_set()
        alertWindowWidget.grab_set()
        alertLabel = Label(alertWindowWidget, text="Invalid file\nTry again!!")
        alertLabel.pack()
        alertLabel.config(fg="black")
        closeAlertWindowButton = Button(
            alertWindowWidget, text="OK", command=closeWarningWindow
        )
        closeAlertWindowButton.pack()

    # Method for make any prediction
    def makePrediction(self, firstText, secondText, windowImage, configuredImage):
        def closeSecondaryWindow():
            secondaryWindow.quit()
            secondaryWindow.destroy()

        secondaryWindow = Toplevel()
        secondaryWindow.geometry("600x300+300+150")
        secondaryWindow.wm_title("Information about health status")
        secondaryWindow.focus_set()
        secondaryWindow.grab_set()
        predictionLabel = Label(secondaryWindow, text=firstText)
        predictionLabel.grid(row=0, column=0)
        predictionLabel.config(fg="black", font=("verdana", 12))
        closeSecondaryWindowButton = Button(
            secondaryWindow, text="Close", command=closeSecondaryWindow
        )
        closeSecondaryWindowButton.grid(row=2, column=3)
        if windowImage == True:
            windowImage = Label(secondaryWindow, image=configuredImage)
            windowImage.grid(row=1, column=0)

            text = Text(secondaryWindow, width=40, height=12)
            text.insert(tkinter.END, secondText)
            text.grid(row=1, column=1)
