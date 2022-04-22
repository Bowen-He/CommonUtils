import matplotlib.pyplot as plt
import re
import numpy as np

def get_data(filename):
    number = []
    with open(filename, "r+") as f:
        while True:
            line = f.readline()
            if line:
                content = re.match("Eval num_timesteps=\d*, episode_reward=(-|)\d*.\d*", line)
                if content:
                    content = content.group().split("=")[-1]
                    number.append(float(content))
            else:
                break
    number = np.convolve(number, np.ones(10)/10, mode='valid')
    return number.ravel().tolist()

data = []
for i in [1, 2, 3, 4, 5]:
    filename = "Bipedal_Walker_2510154_" + str(i) +".o"
    data.append(get_data(filename))
data = np.array(data)
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
axis = np.arange(0, len(mean)) * 10000
plt.plot(axis, mean, label="Bipedal Walker_SAC")
plt.fill_between(x=axis,y1=mean-std, y2=mean+std, alpha=0.2)
plt.legend()
plt.show()
