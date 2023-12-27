import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# построение системы дифференциальных уравнений
def odesys(ya,t, M,m,J,p,alpha,g):
    dy = np.zeros(4)
    dy[0] = ya[2]
    dy[1] = ya[3]

    a11 = 0
    a12= J+m*(ya[0]**2)
    a21 = 1+((ya[0]**2)/(p**2))
    a22 = 0

    b1=M-alpha*ya[3]-2*m*ya[0]*ya[2]*ya[3]
    b2 = -(ya[0]*(ya[2]**2))/(p**2) - ya[0]*(g/p-(ya[3]**2))
    

    dy[2] = (b1*a22 - b2*a12)/(a11*a22 - a12*a21)
    dy[3] = (b2*a11 - b1*a21)/(a11*a22 - a12*a21)

    return dy

# константы
M=0
m = 5
J = 3
p = 1
alpha = 5
g = 9.81

steps=1000
t_fin=20
t=np.linspace(0,t_fin,steps)

x0 = 1
phi0 = 0
dx0 = 0
dphi0 = 0
y0 = [x0, phi0, dx0, dphi0]

# решение системы
Y = odeint(odesys, y0, t, (M,m,J,p,alpha,g))

x =Y[:,0]
phi = Y[:,1]
dx = Y[:,2]
dphi = Y[:,3]
ddphi = [odesys(y,t,M,m,J,p,alpha,g)[3] for y,t in zip(Y,t)]

Ny=m*(2*dphi*dx+ddphi*x)

# рисунок для графиков
fig_for_graphs = plt.figure(figsize=[10,7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, x, color='Blue')
ax_for_graphs.set_title("x(t)")

ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(0)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, phi, color='Red')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(0)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, Ny, color='green')
ax_for_graphs.set_title("Ny(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(0)

fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111, projection='3d')
ax.axis('equal')
ax.grid(0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-3, 4])
ax.set_ylim([-3, 4])
ax.set_zlim([0, 6])
plate_coordinates = np.array([[-2, 0, 0], [2, 0, 0], [2, 0, 5], [-2, 0, 5], [-2, 0, 0]]) #координаты пластины
# Извлекаем координаты X, Y и Z пластины из массива координат пластины
x_plate = plate_coordinates[:, 0]  
y_plate = plate_coordinates[:, 1]  
z_plate = plate_coordinates[:, 2] 
plate = ax.plot(x_plate, y_plate, z_plate, linewidth=3,color='b')[0]

x_1 = np.linspace(-2, 2, 100)  # Создание массива значений x от -2 до 2
y_1 = np.zeros_like(x_1)  # Задаем y как нули, чтобы парабола была в плоскости XOZ
z_1 = 1.3 * x_1**2   # Уравнение первой параболы
z_1 = np.minimum(z_1, 5)
parabola_1=ax.plot(x_1, y_1, z_1, linewidth=2, color='g')[0]  

x_2 = np.linspace(-2, 2, 100)  # Создание массива значений x от -2 до 2
y_2 = np.zeros_like(x_2)  # Задаем y как нули, чтобы парабола была в плоскости XOZ
z_2 = 1.6 * x_2 **2+0.3 # Уравнение  второй параболы
z_2 = np.minimum(z_2, 5)
parabola_2=ax.plot(x_2, y_2, z_2 ,linewidth=2,color='g')[0]  

# координаты точки
pointX=0
pointY=0
pointZ=0.2
point=ax.plot(pointX,pointY,pointZ,color='orange', marker='o', markeredgewidth=3)[0]


def update(i):
    # матрица поворота
    rotation_matrix = np.array([[np.cos(phi[i]), -np.sin(phi[i]), 0],
                                [np.sin(phi[i]), np.cos(phi[i]), 0],
                                [0, 0, 1]])
    
    rotated_plate = np.dot(rotation_matrix, (plate_coordinates ).T).T 
    plate.set_data_3d(rotated_plate[:, 0], rotated_plate[:, 1], rotated_plate[:, 2])
    
    rotated_parabola_1 = np.dot(rotation_matrix, (np.array([x_1, y_1, z_1])))
    parabola_1.set_data_3d(rotated_parabola_1[0], rotated_parabola_1[1], rotated_parabola_1[2])
    
    rotated_parabola_2 = np.dot(rotation_matrix, np.array([x_2, y_2, z_2]))
    parabola_2.set_data_3d(rotated_parabola_2[0], rotated_parabola_2[1], rotated_parabola_2[2])

    # новые координаты точки 
    new_x = x[i]
    new_y = pointY
    new_z=1.45* (new_x)**2+0.15
    
    rotated_point = np.dot(rotation_matrix, np.array([new_x, new_y, new_z]))
    point.set_data_3d([rotated_point[0]], [rotated_point[1]], [rotated_point[2]])


    return plate,parabola_1,parabola_2,point

ani = FuncAnimation(fig, update, frames=len(t), interval=50)

plt.show()

