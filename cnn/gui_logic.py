import cv2
from PIL import ImageTk
from PIL import Image as Img
from tkinter import Toplevel
from tkinter import filedialog as fd
from tkinter import Label, Button, Text
import tkinter


class tools:
    # Method that provide us the image configuration
    def configurar_img(self, ubicacion):
        self.ubicacion = ubicacion
        self.imagen = cv2.imread(self.ubicacion)  # we read the image
        if (self.imagen.shape[0] and self.imagen.shape[1]) == (
            108 and 468
        ):  # if the image is the same size of the logo.jpg size, then it stays
            self.imagen = cv2.cvtColor(
                self.imagen, cv2.COLOR_BGR2RGB
            )  # image will convert to RGB
            self.imagen = Img.fromarray(
                self.imagen
            )  # it is convert from matrix to image
            self.imagen = ImageTk.PhotoImage(
                self.imagen
            )  # it is convert to PhotoImage to set in a label
        elif (self.imagen.shape[0] and self.imagen.shape[1]) != (
            108 and 468
        ):  # if the image is any other size different to logo.jpg
            self.imagen = cv2.resize(
                self.imagen, (200, 200)
            )  # we change the size to 200x200
            self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)
            self.imagen = Img.fromarray(self.imagen)
            self.imagen = ImageTk.PhotoImage(self.imagen)
        return self.imagen  # return the configured image


    def loadFile(self, defaultPath):
        self.givenPath = self.abrir_ventana()
        if self.givenPath != defaultPath:
            self.ventana_alerta()
        return self.givenPath

    # Method for erase n quantity of widgets than aren't necessary
    def eliminar_widgets(self, *args):
        for arg in range(len(args)):
            self.widgets_borrados = args[arg].pack_forget()
        return self.widgets_borrados

    # Method for insert n quantity of widgets on GUI
    def insertar_widgets(self, *args):
        for item in range(len(args)):
            if (
                args[item] == self.panelLogo
            ):  # if the widget is equals to self.panel then it going to GUI bottom
                args[item].pack(side="bottom")
            else:
                args[item].pack()

    # Method for open a window file dialog
    # Return a file path
    def abrir_ventana(self):
        self.ruta = fd.askopenfilename()
        return self.ruta

    # Method for throw an alert window if the user select a wrong file
    def ventana_alerta(self):
        def cerrar_ventana_warning():  # local function for destroy itself
            self.ventana_aviso.destroy()

        self.ventana_aviso = Toplevel()
        self.ventana_aviso.geometry("300x100+500+250")
        self.ventana_aviso.wm_title("WARNING!!!")
        self.ventana_aviso.focus_set()
        self.ventana_aviso.grab_set()
        self.aviso = Label(self.ventana_aviso, text="Invalid file\nTry again!!")
        self.aviso.pack()
        self.aviso.config(fg="black")
        self.botonCerrar = Button(
            self.ventana_aviso, text="OK", command=cerrar_ventana_warning
        )
        self.botonCerrar.pack()

    # Method for make any prediction
    def makePrediction(self, firstText, secondText, imageLabel, imageSelected):
        def closeSecondaryWindow():
            self.subVentana.quit()
            self.subVentana.destroy()

        self.subVentana = Toplevel()
        self.subVentana.geometry("600x300+300+150")
        self.subVentana.wm_title("Information about health status")
        self.subVentana.focus_set()
        self.subVentana.grab_set()
        self.pred = Label(self.subVentana, text=firstText)
        self.pred.grid(row=0, column=0)
        self.pred.config(fg="black", font=("verdana", 12))
        buttonCerrar = Button(
            self.subVentana, text="Close", command=closeSecondaryWindow
        )
        buttonCerrar.grid(row=2, column=3)

        if imageLabel == None:
            self.imageLabel = Label(self.subVentana, image=imageSelected)
            self.imageLabel.grid(row=1, column=0)

            self.text = Text(self.subVentana, width=40, height=12)
            self.text.insert(tkinter.END, secondText)
            self.text.grid(row=1, column=1)
