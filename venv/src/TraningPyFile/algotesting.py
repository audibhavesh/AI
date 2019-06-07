import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
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
    filter = np.zeros((5, 5, 3))
    img=np.zeros((330,480,3))
    tempimg = image
    m = 0
    n = 5
    p = 0
    q = 5
    print(image.shape)
    a =col.rgb_to_hsv(image)
    plt.imshow(a)
    plt.show()
    # f,s=plt.subplots(10,10)
    for i in range(0,66):
        for j in range(0, 96):
            # print(i,j)
            slider = a[p:q, m:n]
            hue=9.66
            print(hue)
            # print(slider)
            for k in range(0,5):
                for l in range(0,5):
                    if hue !=np.max(slider[k][l][0]):
                        filter[k][l] = 255
            # s[i,j].imshow(filter)
            img[p:q,m:n]=filter
            m = n
            n = n + 5
            filter=np.zeros((5, 5, 3))
            break
        p = q
        q = q + 5
        m = 0
        n = 5
        dig = 0
        hue=slider[0][0][0]


    # print(slider.shape)
    plt.imshow(img)
    plt.show()

def makeblack_cifar(image):
    slider = np.zeros((4, 4, 3))
    filter = np.zeros((4, 4, 3))
    img=np.zeros((32,32,3))
    tempimg = image
    m = 0
    n = 4
    p = 0
    q = 4
    print(image.shape)
    a =image
    plt.imshow(a)
    plt.show()
    # f,s=plt.subplots(10,10)
    for i in range(0,8):
        for j in range(0, 8):
            # print(i,j)
            slider = a[p:q, m:n]
            hue=slider[0][0][0]
            # print(hue)
            # print(slider)
            for k in range(0,4):
                for l in range(0,4):
                    if hue !=slider[k][l][0]:
                        filter[k][l] = 255
            # s[i,j].imshow(filter)
            img[p:q,m:n]=filter
            m = n
            n = n + 5
            filter=np.zeros((4, 4, 3))
            break
        p = q
        q = q + 4
        m = 0
        n = 4
        dig = 0
        hue=slider[0][0][0]


    # print(slider.shape)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = Cifar_dataset().cifar10_data_unpickling("/AI//venv//src//cifar-10-batches-py")
    yid = 3
    yd = np.where(y_train == yid)[0]
    image = np.random.choice(yd)
    image = Image.open("/AI/venv/src/res/cat.jpg")
    # image = image.rotate(90)
    img =np.asarray(image)

    plt.imshow(img, cmap="gray")
    plt.show()
    for i in x_train:
        makeblack_cifar(i)
    # makeblack(img)
