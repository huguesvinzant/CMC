#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:34:15 2019

@author: huguesvinzant
"""

import numpy as np
from matplotlib import pyplot as plt


a = [1,1]
tetas = [-np.pi/4, -np.pi/8, -np.pi/16, 0, np.pi/16, np.pi/8, np.pi/4]
force = 1500

lengths = []
moments = []
torques = []

for teta in tetas:
    a1 = a[0]
    a2 = a[1]
    length = np.sqrt(np.power(a1,2)+np.power(a2,2)+2*a1*a2*np.sin(teta))
    moment = (a1*a2*np.cos(teta))/length
    torque = force * moment
    lengths.append(length)
    moments.append(moment)
    torques.append(torque)

plt.figure(1)
plt.plot(tetas, lengths)
plt.plot(tetas, moments)
plt.title('Muscle length and moment arm as a function of pendulum angular position')
plt.xlabel('Pendulum angular position (rad)')
plt.ylabel('Length (m)')
plt.legend(['Muscle length','Muscle moment arm'])
plt.show()

plt.figure(2)
plt.plot(tetas, torques)
plt.title('Torque as a function of pendulum angular position (F = 1500N)')
plt.xlabel('Pendulum angular position (rad)')
plt.ylabel('Torque (N.m)')
plt.show()