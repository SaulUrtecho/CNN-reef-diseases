from tkinter import *
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.initializers import glorot_uniform
from tkinter import Tk, Label, Button, ttk
import tkinter
import re
from gui_logic import tools
from os import path

longitud, altura = 256, 256 # Image size used

# We define all the paths
ruta_logo_itm = './cnn/logo.jpg'
ruta_modelo_default = '/home/saul/Documents/CNN-reef-diseases/cnn/model/model.h5'
ruta_pesos_default = '/home/saul/Documents/CNN-reef-diseases/cnn/model/weights.h5'
ruta_img_prueba = '/home/saul/Documents/CNN-reef-diseases/cnn/test_images'
cont = 0 # With this cont we manage the elimination of select another image button


class UserInterface(tools):
    def __init__(self, master):
        self.master = master
        self.master.title("Detection of diseases in marine corals")
        self.master.geometry("500x400+450+100")
        self.master.resizable(False, False)
        self.etiqueta_modelo = Label(self.master, text="Select the model: ")
        self.etiqueta_modelo.pack()
        self.boton_cargar_modelo = Button(self.master, text="Load model", command=self.cargar_modelo)
        self.boton_cargar_modelo.pack()
        self.etiqueta_pesos = Label(self.master, text="Select the weights: ")
        self.etiqueta_pesos.pack()
        self.boton_cargar_pesos = Button(self.master, text="Load weights", command=self.cargar_pesos)
        self.boton_cargar_pesos.pack()
        self.etiqueta_imagen = Label(self.master, text="Select image to evaluate: ")
        self.etiqueta_imagen.pack()
        self.boton_cargar_imagen = Button(self.master, text="Load Image", command=self.seleccionar_imagen)
        self.boton_cargar_imagen.pack()
        self.var_img = super().configurar_img(ruta_logo_itm)
        self.panelLogo = Label(self.master, image=self.var_img) # El logo se coloca en una etiqueta_imagen
        self.panelLogo.pack(side="bottom")
        self.master.mainloop()

    # Load model method
    # Return the model's path
    def cargar_modelo(self):
        self.currentModelPath = super().loadFile(ruta_modelo_default)
        super().eliminar_widgets(self.etiqueta_modelo, self.boton_cargar_modelo)
        return self.currentModelPath

    # Load weights method
    # Return the weights' path
    def cargar_pesos(self):
        self.currentWeightsPath = super().loadFile(ruta_pesos_default)
        super().eliminar_widgets(self.etiqueta_pesos, self.boton_cargar_pesos)
        return self.currentWeightsPath

    # Select image method
    def seleccionar_imagen(self):
        global cont
        panel_img_principal = None
        self.ruta_imagen_user = super().abrir_ventana()
        self.ruta_imagen_split = path.split(self.ruta_imagen_user)

        if self.ruta_imagen_user is not None:
            if self.ruta_imagen_split[0] == ruta_img_prueba: # we evaluate if the given path is the same to default
                if re.search("\.(jpg|jpeg|JPG|png|bmp|tiff)$", self.ruta_imagen_user): # we evaluate that the selected file is an image
                    super().eliminar_widgets(self.etiqueta_imagen, self.boton_cargar_imagen, self.panelLogo) # clean GUI
                    if cont > 0: # this sentence manage the lifecycle of the select-other-image-button
                        super().eliminar_widgets(self.boton_selec_otra_img)

                    self.imagen_configurada = super().configurar_img(self.ruta_imagen_user)
                    self.botonPredict = Button(text="Detect health status", command=self.prediccion) # button to do the prediction is created
                    self.botonPredict.pack()
                    self.boton_selec_otra_img = Button(text="Select another image", command=self.abrir_otra_img) # button to select other image
                    self.boton_selec_otra_img.pack()

                    if panel_img_principal is None:
                        self.panel_img_principal = Label(self.master, image=self.imagen_configurada)  # the selected image is placed
                        self.panel_img_principal.pack(side="bottom")
                else: # if the given file is not a valid image format throw a alert
                    super().ventana_alerta()
                self.master.mainloop()
            else:
                super().ventana_alerta()
            self.master.mainloop()

    # Method to select other image
    def abrir_otra_img(self):
        global cont
        cont = cont + 1  # when we select the image the cont is incremented by 1 each time
        super().eliminar_widgets(self.panel_img_principal, self.botonPredict) # we must clean the previous data
        self.seleccionar_imagen()

    # Method to do a prediction
    def prediccion(self):
        super().eliminar_widgets(self.botonPredict, self.boton_selec_otra_img, self.panel_img_principal)
        with tf.keras.utils.custom_object_scope({'GlorotUniform':glorot_uniform()}): # initializing weights using GlorotUniform
            cnn = load_model(self.currentModelPath)
        cnn.load_weights(self.currentWeightsPath)
        x = load_img(self.ruta_imagen_user, target_size = (longitud, altura), color_mode='grayscale')
        x = img_to_array(x)
        x = np.expand_dims(x, axis = 0) # we add 1 more channel in the index 0 (1, 256, 256, 1)
        answer = cnn.predict(x)
        respuesta = answer[0]
        respuesta = np.argmax(respuesta)

        panel_ventana_prediccion = None
        if respuesta == 0:
            print(respuesta)
            super().makePrediction('Number five detected', 'The fifth number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        elif respuesta == 1:
            print(respuesta)
            super().makePrediction('Number four detected', 'The fourth number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        elif respuesta == 2:
            print(respuesta)
            super().makePrediction('None', 'there is no number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        elif respuesta == 3:
            print(respuesta)
            super().makePrediction('Number one detected', 'The first number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        elif respuesta == 4:
            print(respuesta)
            super().makePrediction('Number three detected', 'The third number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        elif respuesta == 5:
            print(respuesta)
            super().makePrediction('Number two detected', 'The second number', panel_ventana_prediccion, self.imagen_configurada)
            self.botonNvaDeteccion = Button(self.master, text="New detection", command=self.Limpiar)
            self.botonNvaDeteccion.pack()
            self.botonSalir = Button(self.master, text="Salir", command=self.Salir)
            self.botonSalir.pack()
        return respuesta

    def Limpiar(self):
            self.eliminar_widgets(self.botonNvaDeteccion, self.botonSalir)
            self.insertar_widgets(self.etiqueta_imagen, self.boton_cargar_imagen, self.panelLogo)

    def Salir(self):
        self.master.destroy()

if __name__ == "__main__":
    root = Tk()
    UserInterface(root)