from PIL import Image
import os
import re

#print('this is the PATH-->',os.getcwd())
dirName = os.path.join(os.getcwd(), './cnn/data/test/FIVE/')
totalImages = []
count = 0
size = (500,500)

print("Reading images from: ", dirName)
for root, dirNames, fileNames in os.walk(dirName):
    for fileName in fileNames:
        if re.search("\.(jpg|jpeg|JPG|png|bmp|tiff)$", fileName):
            filePath = os.path.join(root, fileName)
            imagen = Image.open(filePath)
            imagen = imagen.resize(size)
            rotatedImage90 = imagen.rotate(90)
            rotatedImage180 = imagen.rotate(180)
            rotatedImage270 = imagen.rotate(270)
            rotatedImage90 = rotatedImage90.convert("RGB")
            rotatedImage180 = rotatedImage180.convert("RGB")
            rotatedImage270 = rotatedImage270.convert("RGB")
            rotatedImage90.save('./cnn/test_augmentation_images/' + str(count) + "_90_degrees_.jpg")
            rotatedImage180.save('./cnn/test_augmentation_images/' + str(count) + "_180_degrees_.jpg")
            rotatedImage270.save('./cnn/test_augmentation_images/' + str(count) + "_270_degrees_.jpg")
            count = count + 1
            b = "Reading..." + str(count)
            print(b,end="\r")

totalImages.append(count)
print("Number of images from each directory", totalImages)
print("Total images amount in subdirs", sum(totalImages))