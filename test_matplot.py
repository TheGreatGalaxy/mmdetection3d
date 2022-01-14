import numpy as np
import matplotlib.pyplot as plt

# source_data = np.load('curtain_0088.npy')[:,0:3]  #10000x3
source_data = np.random.randint(-100, high=100,size=(1000, 2))

plt.plot(source_data[:, 0], source_data[:, 1])
plt.show()

