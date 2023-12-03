import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
import sympy as sp
import math

steps=1000
t_fin=20
t=np.linspace(0,t_fin,steps)

phi = t * 2 * np.pi / t_fin
xx =1.8* np.sin(t)+3

fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111, projection='3d')
ax.axis('equal')
ax.grid(0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([0, 6])
ax.set_ylim([0, 6])
ax.set_zlim([0, 6])
plate_coordinates = np.array([[1, 1, 0], [5, 1, 0], [5, 1, 5], [1, 1, 5], [1, 1, 0]])
x_plate = plate_coordinates[:, 0]
y_plate = plate_coordinates[:, 1]
z_plate = plate_coordinates[:, 2]
plate = ax.plot(x_plate, y_plate, z_plate, color='b')[0]

x = np.linspace(1, 5, 100)  # Создание массива значений x от 1 до 5
y = np.zeros_like(x)  # Задаем y как нули, чтобы парабола была в плоскости XOZ
z = 1.3 * (x - 3)**2   # Уравнение параболы
z = np.minimum(z, 5)
parabola_1=ax.plot(x, y+1, z,  color='r')[0]  # Отрисовка параболы

x_1 = np.linspace(1, 5, 100)  # Создание массива значений x от 1 до 5
y_1 = np.zeros_like(x_1)  # Задаем y как нули, чтобы парабола была в плоскости XOZ
z_1 = 1.6 * (x_1 - 3)**2+0.3 # Уравнение параболы
z_1 = np.minimum(z_1, 5)
parabola_2=ax.plot(x_1, y_1+1, z_1 ,color='r')[0]  # Отрисовка параболы

pointX=3
pointY=1
pointZ=0.2
point=ax.plot(pointX,pointY,pointZ,color='orange', marker='o', markeredgewidth=3)[0]

# Ось вращения
axis_start = np.array([3, 1, 0])
axis_start_array = np.array([[3, 1, 0]] * 100)
axis_end = np.array([3, 1, 5])

def update(frame):
    rotation_matrix = np.array([[np.cos(phi[frame]), -np.sin(phi[frame]), 0],
                                [np.sin(phi[frame]), np.cos(phi[frame]), 0],
                                [0, 0, 1]])
    rotated_plate = np.dot(rotation_matrix, (plate_coordinates - axis_start).T).T + axis_start
    
    plate.set_data(rotated_plate[:, 0:2].T)
    plate.set_3d_properties(rotated_plate[:, 2].T)
    rotated_points = np.dot(rotation_matrix, ((np.array([x, y+1, z]))-axis_start_array.T))+axis_start_array.T
    parabola_1.set_data(rotated_points[0], rotated_points[1])  
    parabola_1.set_3d_properties(rotated_points[2]) 
    
    rotated_points_2 = np.dot(rotation_matrix, np.array([x_1, y_1+1, z_1]-axis_start_array.T))+axis_start_array.T

    parabola_2.set_data_3d(rotated_points_2[0], rotated_points_2[1], rotated_points_2[2])



    new_x = xx[frame]
    new_y = pointY
    new_z=1.45* (new_x-3)**2+0.15


    point.set_data_3d([new_x], [new_y], [new_z]) 
    rotated_point = np.dot(rotation_matrix, (np.array([new_x, new_y, new_z]) - axis_start)) + axis_start


    # point.set_data_3d([new_x], [new_y], [new_z])
    point.set_data(rotated_point[0], rotated_point[1])
    point.set_3d_properties(rotated_point[2])


    # return [plate,parabola_1,parabola_2,point]

ani = FuncAnimation(fig, update, frames=len(t), interval=50)

plt.show()
