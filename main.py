import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import linregress


#implement Lorenz equations
def f(t, y):
    x, y1, z = y
    return np.array([10 * (y1 - x), x * (28 - z) - y1, x * y1 - (8 / 3) * z])   


#create random initial conditions for x, y, and z enclosed by -10 and 10
np.random.seed(1)
y0_list=np.random.uniform(low=-10, high=10, size=(10,3))

#create a list of slopes for the line of best fits
xslopes=[]
yslopes=[]
zslopes=[]

#create arrays for the start times and end times for each trial
start=np.array([5,0,25,10,5,15,10,5,0,0])
end=np.array([30,35,55,32,35,40,30,33,26,32])



#loop through all the random initial conditions
for i, y0 in enumerate(y0_list):


    #changing x, y, and z for each inital condition
    xstart=y0+np.array([1e-10,0,0])
    ystart=y0+np.array([0,1e-10,0])
    zstart=y0+np.array([0,0,1e-10])


    #set time frame
    tf=65
    t_eval=np.linspace(0,tf,5000)


    #get solutions
    solution=solve_ivp(f,(0,tf),y0,t_eval=t_eval)
    xchange=solve_ivp(f,(0,tf),xstart,t_eval=t_eval)
    ychange=solve_ivp(f,(0,tf),ystart,t_eval=t_eval)
    zchange=solve_ivp(f,(0,tf),zstart,t_eval=t_eval)


    #plot the change in x, y, and z vs time
    plt.figure()
    plt.plot(solution.t, solution.y[0], label='x')
    plt.plot(solution.t, solution.y[1], label='y')
    plt.plot(solution.t, solution.y[2], label='z')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()


    #plot 3d graph of control solution vs x, y, and z experimental solutions
    fig=plt.figure()


    #x experimental solution
    ax1=fig.add_subplot(221, projection='3d')
    ax1.plot(solution.y[0],solution.y[1],solution.y[2], label='control')
    ax1.plot(xchange.y[0], xchange.y[1], xchange.y[2],label='x change')
    ax1.set_title('changing x')
    ax1.legend()
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')


    #y experimental solution
    ax2=fig.add_subplot(222, projection='3d')
    ax2.plot(solution.y[0],solution.y[1],solution.y[2], label='control')
    ax2.plot(ychange.y[0], ychange.y[1], ychange.y[2],label='y change')
    ax2.set_title('changing y')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')


    #z experimental solution
    ax3=fig.add_subplot(223, projection='3d')
    ax3.plot(solution.y[0],solution.y[1],solution.y[2], label='control')
    ax3.plot(zchange.y[0], zchange.y[1], zchange.y[2],label='z change')
    ax3.set_title('changing z')
    ax3.legend()
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')


    plt.tight_layout(h_pad=2)
    plt.show()




    #create linear regression for x, y, and z
    ax1=plt.subplot(221)


    #measure error
    delta=np.linalg.norm(solution.y-xchange.y, axis=0, ord=2)

    #filtering the time to evaluate the linear regression
    mask=np.logical_and(t_eval>start[i],t_eval<end[i])

    #create linear regression
    a,b,r,p,s=linregress(t_eval[mask], np.log(delta)[mask])


    #plot data
    ax1.semilogy(t_eval[mask], delta[mask], label='Actual Error')
    ax1.semilogy(t_eval[mask], np.exp(a*t_eval+b)[mask],label='Regression')
    ax1.set_title('X Change differences; Lyapunov Exponent $\lambda${}'.format(a), fontsize=5.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Separation')


    #add Lyapunov Exponent to xslopes list
    xslopes.append(a)






    ax2=plt.subplot(222)

    #meausre error
    delta=np.linalg.norm(solution.y-ychange.y, axis=0, ord=2)

    #filtering the time to evaluate the linear regression
    mask=np.logical_and(t_eval>start[i],t_eval<end[i])


    #create linear regression
    a,b,r,p,s=linregress(t_eval[mask], np.log(delta)[mask])


    #plot data
    ax2.semilogy(t_eval[mask], delta[mask], label='Actual Error')
    ax2.semilogy(t_eval[mask], np.exp(a*t_eval+b)[mask],label='Regression')
    ax2.set_title('Y Change differences; Lyapunov Exponent $\lambda${}'.format(a), fontsize=5.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Separation')


    #add Lyapunov Exponent to yslopes list
    yslopes.append(a)






    ax3=plt.subplot(223)

    #measure error
    delta=np.linalg.norm(solution.y-zchange.y, axis=0, ord=2)

    #filtering the time to evaluate the linear regression
    mask=np.logical_and(t_eval>start[i],t_eval<end[i])

    #create linear regression
    a,b,r,p,s=linregress(t_eval[mask], np.log(delta)[mask])

    #plot data
    ax3.semilogy(t_eval[mask], delta[mask], label='Actual Error')
    ax3.semilogy(t_eval[mask], np.exp(a*t_eval+b)[mask],label='Regression')
    ax3.set_title('Z Change differences; Lyapunov Exponent $\lambda${}'.format(a), fontsize=5.7)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Separation')


    #add Lyapunov Exponent to zslopes list
    zslopes.append(a)


    plt.tight_layout(h_pad=2)
    plt.show()


#get average of Lyapunov Exponents
x_avg=np.mean(xslopes)
y_avg=np.mean(yslopes)
z_avg=np.mean(zslopes)

#get median of Lyapunov Exponents
x_median=np.median(xslopes)
y_median=np.median(yslopes)
z_median=np.median(zslopes)

#get standard deviations of Lyapunov Exponents
x_std=np.std(xslopes)
y_std=np.std(yslopes)
z_std=np.std(zslopes)

print('x avg={}, y avg={}, z avg={}\nx med={}, y med={}, z med={}\nx std={}, y std={}, z std={}'.format(x_avg,y_avg,z_avg,x_median,y_median,z_median,x_std,y_std,z_std))

