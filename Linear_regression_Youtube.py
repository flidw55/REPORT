# -*- coding: utf8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#使用sklearn中的 Linear_model
from sklearn import linear_model
from simple_linear_regression import r_squared


#抓取資料
url = 'http://www.statsheep.com/UCIdhd_1spj49unBWx1fjS2A/28days'
TablePd=pd.read_html(url)[0]

#抽取日期
Date = pd.to_datetime(TablePd[0][3:],format="%m%d", errors='ignore')
count = len(Date)
#print Date

#整理資料格式
data_Date = pd.DataFrame(pd.Series(np.arange(count), index=Date))
data_SubsTotal = pd.Series(TablePd[2][3:],dtype=np.uint,name="Total")
#print("Date\n{},\nTotal\n{}".format(data_Date[0],data_SubsTotal))



#取得輸入用之方程式係數
A = np.vstack([np.arange(count), np.ones(count)]).T #.T=transpose theself
#取得beta , alpha之值
beta, alpha = np.linalg.lstsq(A, data_SubsTotal)[0]

print("y = {:.4f} * x + ({:.4f})".format(beta, alpha))
print "r-squared", r_squared(alpha, beta, data_Date, data_SubsTotal)
print ("估計14天後的訂閱人數：{}".format(beta*(count+14)+alpha))

regr = linear_model.LinearRegression()
regr.fit(data_Date,data_SubsTotal)
print("估計14天後的訂閱人數：{0[0]}".format(regr.predict(count+14)))

#畫出圖表
plt.plot(data_Date[0],data_SubsTotal, 'go',label="Original Line")
plt.plot(data_Date[0],  beta * data_Date[0] + alpha, 'b-',label="Fitted Data") # render blue line
plt.xticks(data_Date[0],[x[-5:] for x in data_Date[0].index])
plt.show()