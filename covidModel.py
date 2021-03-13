import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import integrate
from scipy.optimize import fmin
import plotly
import plotly.express as px
                                                                
#Data
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
data1 = pd.read_csv(url) # coronavirus data
data_time_series = data1.iloc[244,30:60]#30 days
# interactive line visualization
fig2 = px.line(x=data_time_series.index, y=data_time_series.values)
plotly.offline.plot(fig2, filename='covid_time_series.html') #save the interactive plot to html.

Td=range(0,30)
Id=list(data_time_series.values)
# parameters   
N = 328200000 # Total population of USA
beta = 0.2
gamma = 0.001  
rates=(beta,gamma)
# initial conditions

I0 = 1 # infected
R0 = 0 #revovered
S0 = N - I0 - R0 # suceptible          
y0 = [S0, I0, R0]# initial condition vector

start_time=0.0
end_time=365 #days
intervals=1000
mt=np.linspace(start_time,end_time,intervals)

# model index compare to data
findindex=lambda x:np.where(mt>=x)[0][0]
mindex=list(map(findindex, Td))

def eq(N,par,initial_cond,start_t,end_t,incr):
     #-time grid 
     t  = np.linspace(start_t, end_t,incr)
     #differential eq system
     def funct(y,t):
        S=y[0]
        I=y[1]
        R=y[2]
        beta,gamma=par
        # the model equations
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]
     #integrate 
     ds = integrate.odeint(funct,initial_cond,t)
     return (ds[:,0],ds[:,1],ds[:,2],t)                                  
# Score Fit                           
def score(parms):
    #Solution to system
    F0,F1,F2,T=eq(N,parms,y0,start_time,end_time,intervals)
    #Pick of Model Points to Compare
    Im=F1[mindex]
    #Score Difference between model and data points
    ss=lambda data,model:((data-model)**2).sum()
    return ss(Id,Im)
                                 
# Optimize Fit 
fit_score=score(rates)
answ=fmin(score,(rates),full_output=1,maxiter=1000000)
bestrates=answ[0]
bestscore=answ[1]
beta,gamma=answ[0]
newrates=(beta,gamma)

#Generate Solution to System                               
F0,F1,F2,T=eq(N,newrates,y0,start_time,end_time,intervals)
Im=F1[mindex]
Tm=T[mindex]                                  

#Plot Solution to System                                     
plt.figure()
plt.plot(T,F1,'b-',Tm,Im,'ro',Td,Id,'go')
plt.xlabel('days')
plt.ylabel('Infected population')
title='Covid in USA  Fit Score: '+str(bestscore)
plt.title(title)
plt.show()
                                    