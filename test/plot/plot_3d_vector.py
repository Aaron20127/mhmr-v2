
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

num_vector = 6

x_point_list = np.array([np.sin(i) for i in np.linspace(0, np.pi, num_vector + 1)])
y_point_list = np.array([np.cos(i) for i in np.linspace(0, np.pi, num_vector + 1)])
z_point_list = np.linspace(0, 2, num_vector + 1)

x_vector_start = x_point_list[:-1]
y_vector_start = y_point_list[:-1]
z_vector_start = z_point_list[:-1]

x_vector = x_point_list[1:] - x_point_list[:-1]
y_vector = y_point_list[1:] - y_point_list[:-1]
z_vector = z_point_list[1:] - z_point_list[:-1]


ax.quiver(x_vector_start,
          y_vector_start,
          z_vector_start,
          x_vector,
          y_vector,
          z_vector,
          arrow_length_ratio=0.3,
          color='r')

ax.quiver(x_vector_start,
          y_vector_start,
          z_vector_start + 1,
          x_vector,
          y_vector,
          z_vector,
          arrow_length_ratio=0.3,
          color='b')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 3])
plt.show()


