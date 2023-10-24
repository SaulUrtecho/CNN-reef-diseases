import os
import random
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt


dataDir = "./cnn/data/training/"
imageSize = (300,300)

fig, ax = plt.subplots(1, 6, figsize=(10, 4))
fig.canvas.manager.set_window_title("Classes")

fiveSample = random.choice(os.listdir(dataDir + "FIVE"))
image = load_img(dataDir + "FIVE/" + fiveSample)
image = image.resize(size=imageSize)
ax[0].imshow(image)
ax[0].set_title("five")
ax[0].axis("Off")

fourSample = random.choice(os.listdir(dataDir + "FOUR"))
image = load_img(dataDir + "FOUR/" + fourSample)
image = image.resize(size=imageSize)
ax[1].imshow(image)
ax[1].set_title("four")
ax[1].axis("Off")

noneSample = random.choice(os.listdir(dataDir + "NONE"))
image = load_img(dataDir + "NONE/" + noneSample)
image = image.resize(size=imageSize)
ax[2].imshow(image)
ax[2].set_title("none")
ax[2].axis("Off")

oneSample = random.choice(os.listdir(dataDir + "ONE"))
image = load_img(dataDir + "ONE/" + oneSample)
image = image.resize(size=imageSize)
ax[3].imshow(image)
ax[3].set_title("one")
ax[3].axis("Off")

threeSample = random.choice(os.listdir(dataDir + "THREE"))
image = load_img(dataDir + "THREE/" + threeSample)
image = image.resize(size=imageSize)
ax[4].imshow(image)
ax[4].set_title("three")
ax[4].axis("Off")

twoSample = random.choice(os.listdir(dataDir + "TWO"))
image = load_img(dataDir + "TWO/" + twoSample)
image = image.resize(size=imageSize)
ax[5].imshow(image)
ax[5].set_title("two")
ax[5].axis("Off")

plt.show()