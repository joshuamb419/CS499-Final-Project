import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

starting_Q = {}
with open('everyvisitmc.dict', 'rb') as file:
    starting_Q = pickle.load(file)

heatmap = [[0 for y in range(19)] for x in range(19)]

for key in starting_Q.keys():
    # print(key)
    image_arr = np.frombuffer(key[1], dtype=np.int64)
    image_arr = image_arr.reshape(19, 19, 3)
    for x in range(len(image_arr)):
        for value in image_arr[x]:
            if value[2] == 10:
                heatmap[value[0]][value[1]] = starting_Q[key].max()
                break

print(heatmap)
plt.imshow(heatmap, cmap='viridis')
plt.colorbar()
plt.title("MC Every Visit Expected Reward Heatmap")
plt.show()
