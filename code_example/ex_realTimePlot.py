# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:28:58 2024
@author: subin
Draw realtime data using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 초기 데이터 설정
x_data = np.linspace(0, 100, 101)
y_data = [0 for i in range(101)]

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')  # 초기 라인 객체 생성
ax.set_xlim(0, 100)  # X축 범위
ax.set_ylim(0, 100)  # Y축 범위

# 경계선 추가
ax.axhline(20, color='red', linewidth=2)
ax.axhline(80, color='blue', linewidth=2)

# 텍스트 레이블 추가
ax.text(50, 10, 'Muscle', horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)
ax.text(50, 50, 'Fat', horizontalalignment='center', verticalalignment='center', color='green', fontsize=12)
ax.text(50, 90, 'Vein', horizontalalignment='center', verticalalignment='center', color='blue', fontsize=12)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    #x_data = np.linspace(0, 100, 101)
    y_data.append(np.random.randint(0, 100))  # 예시 데이터: 무작위로 생성된 Y값
    
    y_data.pop(0)
    
    line.set_data(x_data, y_data)
    return line,

# 애니메이션 설정
ani = FuncAnimation(fig, update, frames=1, init_func=init, blit=True, interval=100)

plt.show()