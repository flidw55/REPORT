# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from simple_linear_regression import r_squared

#測試用數值
num = 20    #資料筆數
residual = 3 #殘差範圍

#設定一匿名函數，作用為將丟入值加上隨機正負號
random_sign = np.vectorize(lambda x: x if np.random.sample() > 0.5 else -x)

original_X = np.linspace(1,9, num) #產生num個，從 1到9 的List
original_Y = original_X
fitted_Y = random_sign(np.random.sample(num) * residual) + original_X


#取得輸入用之方程式係數
A = np.vstack([original_X, np.ones(num)]).T #.T=transpose theself
#取得beta , alpha之值
beta, alpha = np.linalg.lstsq(A, fitted_Y)[0]
print(np.linalg.lstsq(A, fitted_Y))
#印出數值
print("y = {:.4f} * x + ({:.4f})".format(beta, alpha))
print "r-squared", r_squared(alpha, beta, original_X, fitted_Y)

#劃出圖表
plt.plot(original_X, original_Y, 'g-',label="Original Line")
plt.plot(original_X, fitted_Y, 'ro',label="Original Data")
plt.plot(original_X,  beta * original_X + alpha, 'b-',label="Fitted Data") # render blue line
plt.legend()
plt.show()