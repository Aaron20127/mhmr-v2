
import matplotlib.pyplot as plt
import numpy as np

fig,ax=plt.subplots()

num_vector = 5

x_point_list = np.array([np.sin(i) for i in np.linspace(0, np.pi, num_vector + 1)])
y_point_list = np.array([np.cos(i) for i in np.linspace(0, np.pi, num_vector + 1)])

x_vector_start = x_point_list[:-1]
y_vector_start = y_point_list[:-1]

x_vector = x_point_list[1:] - x_point_list[:-1]
y_vector = y_point_list[1:] - y_point_list[:-1]


ax.quiver(x_vector_start,
          y_vector_start,
          x_vector,
          y_vector,
          angles='xy', scale_units='xy', scale=1, color='r')

ax.set_xlim([-1, 2])
ax.set_ylim([-1, 2])
plt.show()