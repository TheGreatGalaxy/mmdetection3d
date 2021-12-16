import numpy as np
import matplotlib.pyplot as plt

# source_data = np.load('curtain_0088.npy')[:,0:3]  #10000x3
# source_data = np.random.randint(-100, high=100,size=(1000, 2))

# plt.plot(source_data[:, 0], source_data[:, 1])
# plt.show()

np.empty(shape=(2, 2)).tofile('test.txt')
a = np.empty(shape=(0, 0))
print("a: \n", a)
a.tofile('test1.bin')
print("a.shape: \n", a.shape)


b = np.fromfile("test1.bin", dtype=np.float)
print("b: \n", b)
print("b.shape: \n", b.shape)
