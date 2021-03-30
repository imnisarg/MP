def convection(nt, nx, tmax, xmax, c):
    dt = tmax/(nt-1)
    dx = xmax/(nx-1)

   # Initialise data structures
    import numpy as np
    u = np.zeros((nx,nt))
    x = np.zeros(nx)

   
    
    u[0,:] = 0.4
    for i in range(1,nx-1):
        

        u[i,0] = 0.4 - (0.15*i/(nx-1))
        

   # Loop
    for n in range(0,nt-1):
        for i in range(1,nx-1):

            u[i,n+1] = u[i,n]-u[i,n]*(dt/dx)*(u[i,n]-u[i-1,n])

   # X Loop
    for i in range(0,nx):
        x[i] = i*dx
    #print(u)
    return u, x

def plot_convection(u,x,nt,title):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    plt.figure()
    
    import matplotlib.animation as animation
    
    for i in range(0,nt,10):
        plt.plot(x,u[:,i])
        plt.xlabel('x (m)')
        plt.ylabel('u Dimensionless Number giving vehicle density INDICATION')
        plt.ylim([0,2.2])
        plt.title(title + str(i/nt))
        plt.show()
    
        
    
    

u,x = convection(151, 51, 1, 1, 0.5)
plot_convection(u,x,151,'Plot at time = ')
