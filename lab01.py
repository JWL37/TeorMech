import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# Константа для уменьшения размера стрелок
decrease = 7

# Функция поворота
def rotation2D(x, y, a):
    Rx = x * np.cos(a) - y * np.sin(a)
    Ry = x * np.sin(a) + y * np.cos(a)
    return Rx, Ry


T = np.linspace(0, 15, 2000)
t = sp.Symbol('t')

# Заданные функции
r = 1 + 1.5 * sp.sin(12 * t)
phi = 1.25 * t + 0.2 * sp.cos(12 * t)

x = r * sp.cos(phi)
y = r * sp.sin(phi)
Vx = sp.diff(x, t)/ decrease
Vy = sp.diff(y, t)/ decrease
Ax = sp.diff(Vx, t)/ decrease
Ay = sp.diff(Vy, t)/ decrease

# Заполнение массивов нулями
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
AX = np.zeros_like(T)
AY = np.zeros_like(T)

# Заполнение массивов значениями заменяющие нули
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    AX[i] = sp.Subs(Ax, t, T[i])
    AY[i] = sp.Subs(Ay, t, T[i])

# Настройка графика
fig = plt.figure()

ax = fig.add_subplot()
ax.axis('equal')
ax.set(xlim=[-5, 5], ylim=[-5, 5])

# Построение траектории
ax.plot(X, Y, color="gray")

# Настройка отображаемых значений в начальный момент времени
ArrowX = np.array([-0.2, 0, -0.2])
ArrowY = np.array([0.1, 0, -0.1])

point, = ax.plot(X[0], Y[0], marker = 'o',color="#00FFFF")

radius_vector, = ax.plot([0, X[0]], [0, Y[0]],  color="red")
velocity_vector, = ax.plot([X[0], X[0] + VX[0]], [Y[0], Y[0] + VY[0]],  color="yellow")
acceleration_vector, = ax.plot([X[0], X[0] + AX[0]], [Y[0], Y[0] + AY[0]], color="green")

rot_arrowX, rot_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(Y[0], X[0]))
rot_vel_arrowX, rot_vel_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
rot_acc_arrowX, rot_acc_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(AY[0], AX[0]))

radius_vector_arrow, = ax.plot(rot_arrowX + VX[0] + X[0], rot_arrowY + VY[0] + Y[0],  color="red")
velocity_vector_arrow, = ax.plot(rot_vel_arrowX + VX[0] + X[0], rot_vel_arrowY + VY[0] + Y[0],  color="yellow")
acceleration_vector_arrow, = ax.plot(rot_acc_arrowX + X[0], rot_acc_arrowY + Y[0], color="green")

# Функция для анимации, которая изменяет все значения i раз
def anime(i):

    point.set_data(X[i], Y[i])
    
    radius_vector.set_data([0, X[i]], [0, Y[i]])
    
    velocity_vector.set_data([X[i] , (X[i] + VX[i]) ], [Y[i] , (Y[i] + VY[i]) ])
    
    acceleration_vector.set_data([X[i] , (X[i] + AX[i]) ], [Y[i] , (Y[i] + AY[i]) ])
    
    rot_arrowX, rot_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(Y[i], X[i]))
    radius_vector_arrow.set_data(rot_arrowX + X[i], rot_arrowY + Y[i])
    
    rot_vel_arrowX, rot_vel_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    velocity_vector_arrow.set_data(rot_vel_arrowX + (X[i] + VX[i]) , rot_vel_arrowY + (Y[i] + VY[i]))
    
    rot_acc_arrowX, rot_acc_arrowY = rotation2D(ArrowX, ArrowY, math.atan2(AY[i], AX[i]))
    acceleration_vector_arrow.set_data(rot_acc_arrowX + (X[i] + AX[i]) , rot_acc_arrowY + (Y[i] + AY[i]) )
    
    return point, velocity_vector, velocity_vector_arrow, acceleration_vector, acceleration_vector_arrow, radius_vector, radius_vector_arrow

# Создание анимации
anim = FuncAnimation(fig, anime, frames=1000, interval=5,repeat = False)
plt.show()
