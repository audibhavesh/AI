import numpy as np
from src.TraningPyFile.split_glass_csv import split_glass
#
x, y, x_test, y_test = split_glass().split_glass_data()
# x=np.random.rand(150,9)
w=np.random.rand(9,7)
b=np.random.rand(1,7)
a=np.dot(x,w)
d=np.add(a,b)
n = np.exp(d- np.max(d))
d = np.sum(n, axis=0)
sf = n/ d
print(sf.shape)
p=np.argmax(sf,axis=1)
c=np.sum(p==y)
