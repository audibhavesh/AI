import numpy as np
import matplotlib.pyplot as plt
from src.TraningPyFile.Cifar_dataset import Cifar_dataset
from PIL import Image


def conRGBtoGray(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2990 * r + 0.5870 * g + 0.1440 * b
    # plt.imshow(gray,cmap="gray")
    # plt.show()
    return gray
    # return np.dot(image[...,:3], [0.299, 0.587, 0.144])
    # return gray


def makeblack(image):
    slider = np.zeros((5, 5, 3))
    tempimg = image
    m=0
    n=5
    p=0
    q=5
    print(image.shape)
    for i in range(0,66):
        for i in range(0,96):
            slider = image[p:q,m:n]
            m=n
            n=n+5

        p=q
        q=q+5
        m = 0
        n = 5
        dig=0
    for i in range(5,0,-1):



            pass
    plt.imshow(slider)
    plt.show()



if __name__ == '__main__':
    x_train, y_train, x_test, y_test = Cifar_dataset().cifar10_data_unpickling("/AI//venv//src//cifar-10-batches-py")
    yid = 3
    yd = np.where(y_train == yid)[0]
    image = np.random.choice(yd)
    image = Image.open("/AI/venv/src/res/cat.jpg")
    # image = image.rotate(90)
    img = np.asarray(image)
    plt.imshow(img, cmap="gray")
    plt.show()
    makeblack(img)
