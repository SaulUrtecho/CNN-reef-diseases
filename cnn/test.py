import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

model = './cnn/model/model.h5'
weights = './cnn/model/weights.h5'
cnn = load_model(model)
cnn.load_weights(weights)


def predict(file):
    x=load_img(file, target_size=(256,256), color_mode='grayscale')
    x=img_to_array(x)
    x=np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    print(array)
    result = array[0]
    result = np.argmax(result)

    if result == 0:
        print('five')
    elif result == 1:
        print('four')
    elif result == 2:
        print('none')
    elif result == 3:
        print('one')
    elif result == 4:
        print('three')
    elif result == 5:
        print('two')
    return result

predict('./cnn/none.jpg')
predict('./cnn/one.png')
predict('./cnn/two.jpg')
predict('./cnn/three.jpg')
predict('./cnn/four.jpeg')
predict('./cnn/five.jpg')