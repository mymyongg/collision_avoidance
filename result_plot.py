# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:18:23 2016

@author: steven
"""
import matplotlib.pyplot as plt
import numpy as np
lines = np.loadtxt("9999_train_result.txt", comments="#", delimiter="\n", unpack=False)

plt.plot(lines)
plt.show()
