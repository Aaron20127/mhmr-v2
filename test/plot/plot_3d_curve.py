# import necessary module
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

# load data from file
# you replace this using with open
x_list = np.array([np.sin(i) for i in np.linspace(0, 1.5*np.pi, 10)])
y_list = np.array([np.cos(i) for i in np.linspace(0, 1.5*np.pi, 10)])
z_list = np.ones(10)


# new a figure and set it into 3d
fig = plt.figure()
ax = fig.gca(projection='3d')

# set figure information
ax.set_title("3D_Curve")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# draw the figure, the color is r = read
figure1 = ax.plot(x_list, y_list, z_list, c='r')
plt.show()