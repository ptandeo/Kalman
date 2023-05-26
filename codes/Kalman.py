#!/usr/bin/env python

""" Kalman.py: Apply the linear and Gaussian Kalman filter and smoother. """

__author__ = "Pierre Tandeo"
__version__ = "0.1"
__date__ = "2022-03-09"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@imt-atlantique.fr"

from pylab import *
from numpy import *
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def Kalman_filter(y, x0, P0, M, Q, H, R):
    """ Apply the linear and Gaussian Kalman filter. """
    
    # shapes
    n = len(x0)
    T, p = shape(y)

    # Kalman initialization
    x_f = zeros((T,n))   # forecast state
    P_f = zeros((T,n,n)) # forecast error covariance matrix
    x_a = zeros((T,n))   # analysed state
    P_a = zeros((T,n,n)) # analysed error covariance matrix
    loglik = zeros((T))  # log-likelihood
    K_a = zeros((T,n,p)) # analysed Kalman gain
    x_a[0,:]   = x0
    P_a[0,:,:] = P0
        
    # apply the Kalman filter
    for k in range(1,T):
    
        # prediction step
        x_f[k,:]   = M @ x_a[k-1,:]
        P_f[k,:,:] = M @ P_a[k-1,:,:] @ M.T + Q
        
        # Kalman gain
        K_a[k,:,:] = P_f[k,:,:] @ H.T @ inv(H @ P_f[k,:,:] @ H.T + R)
        
        # update step
        x_a[k,:]   = x_f[k,:] + K_a[k,:,:] @ (y[k,:] - H @ x_f[k,:])
        P_a[k,:,:] = P_f[k,:] - K_a[k,:,:] @ H @ P_f[k,:,:]
        
        # stock the log-likelihood
        loglik[k] = -0.5*((y[k,:] - H @ x_f[k,:]).T @ inv(H @ P_f[k,:,:] @ H.T + R) @ (y[k,:] - H @ x_f[k,:])) - 0.5 * (n * log( 2 * np.pi)+ log(det(H @ P_f[k,:,:] @ H.T + R)))
    
    return x_f, P_f, x_a, P_a, loglik, K_a
    
def Kalman_smoother(y, x0, P0, M, Q, H, R):
    """ Apply the linear and Gaussian Kalman smoother. """
    
    # shapes
    n = len(x0)
    T, p = shape(y)
    
    # Kalman initialization    
    x_s = zeros((T,n))       # smoothed state
    P_s = zeros((T,n,n))     # smoothed error covariance matrix
    P_s_lag = zeros((T-1,n,n)) # smoothed lagged error covariance matrix
    
    # apply the Kalman filter
    x_f, P_f, x_a, P_a, loglik, K_a = Kalman_filter(y, x0, P0, M, Q, H, R)
    
    # apply the Kalman smoother
    x_s[-1,:]   = x_a[-1,:]
    P_s[-1,:,:] = P_a[-1,:,:]
    for k in range(T-2,-1,-1):
        K = P_a[k,:,:] @ M.T @ inv(P_f[k+1,:,:])
        x_s[k,:]   = x_a[k,:] + K @ (x_s[k+1,:] - x_f[k+1,:])
        P_s[k,:,:] = P_a[k,:,:] + K @ (P_s[k+1,:,:] - P_f[k+1,:,:]) @ K.T
        
    for k in range(0,T-1):
        A = (eye(n) - K_a[k+1,:,:] @ H) @ M @ P_a[k,:,:]
        B = (P_s[k+1,:,:] - P_a[k+1,:,:]) @ inv(P_a[k+1,:,:])
        P_s_lag[k,:,:] = A + B @ A
        
    return x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag

def Kalman_EM(y, xb, B, M, Q, H, R, nb_iter_EM):
    """ Apply the expectation-maximization algorithm. """
    
    # Kalman initialization
    x0 = xb[0,:]
    P0 = B
    
    # apply the Kalman smoother
    x_f, P_f, x_a, P_a, x_s, P_s, loglik = Kalman_smoother(y, x0, P0, M, Q, H, R)
    
    # copy x
    #x_s = xb.copy()
    
    # shapes
    n = shape(xb)[1]
    
    # tab to store the log-likelihood
    tab_loglik = []
    
    # loop on the SEM iterations
    for i in tqdm(arange(0, nb_iter_EM)):
        
        # Kalman parameters
        reg = LinearRegression(fit_intercept=False).fit(x_s[:-1,], x_s[1:,])
        M   = reg.coef_
        Q   = cov((x_s[1:,] - reg.predict(x_s[:-1,])).T)
        #R   = cov(y.T - H @ x.T)
        
        # Kalman initialization
        x0 = x_s[0,:]
        P0 = P_s[0,:,:,]
        
        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik = Kalman_smoother(y, x0, P0, M, Q, H, R)
        
        # store the log-likelihod
        tab_loglik = append(tab_loglik, sum(loglik))
        
    return x_s, P_s, M, tab_loglik

def Kalman_SEM(x, y, H, R, nb_iter_SEM):#, x_t, t):
    """ Apply the stochastic expectation-maximization algorithm. """
    
    # fix the seed
    random.seed(11)
    
    # copy x
    x_out = x.copy()
    
    # shapes
    n = shape(x_out)[1]
    
    # tab to store the log-likelihood
    tab_loglik = [] 
    
    # loop on the SEM iterations
    for i in tqdm(arange(0, nb_iter_SEM)):
        
        # Kalman parameters
        reg = LinearRegression(fit_intercept=False).fit(x_out[:-1,], x_out[1:,])
        M   = reg.coef_
        Q   = cov((x_out[1:,] - reg.predict(x_out[:-1,])).T)
        #R   = cov(y.T - H @ x.T)
        
        # Kalman initialization
        if i==0:
            x0 = zeros(n)
            P0 = eye(n)
        else:
            x0 = x_s[0,:]
            P0 = P_s[0,:,:,]
        
        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother(y, x0, P0, M, Q, H, R)
        
        # store the log-likelihod
        tab_loglik = append(tab_loglik, sum(loglik))
        
        # simulate the new x
        for k in range(len(x_s)):
            x_out[k,:] = random.multivariate_normal(x_s[k,:], P_s[k,:,:])
        
        '''
        if(shape(x_s)[1]==3):
            # plot
            figure()
            i_unobs_comp=0
            subplot(2,3,1)
            plot(x_s[:,0], x_s[:,2], 'C2')
            #plot(x_t[:,1], x_t[:,0], 'k--')
            title('$(z_1, x_2)$ plane', size=30)
            xlabel('$x_2$', size=20)
            ylabel('$z_1$', size=20)
            xlim([-24,22])
            ylim([-27,35])
            subplot(2,3,2)
            plot(x_s[:,1], x_s[:,2], 'C2')
            #plot(x_t[:,2], x_t[:,0], 'k--')
            title('$(z_1, x_3)$ plane', size=30)
            xlabel('$x_3$', size=20)
            #ylabel('$z_1$', size=20)
            xlim([8,43])
            ylim([-27,35])
            subplot(2,3,3)
            plot(tab_loglik[1:], 'C2')
            #plot(loglik_V0[1:], '--k')
            title('Log-likelihood', size=30)
            xlabel('Iterations', size=20)
            xlim([0,30])
            ylim([12000,30000])
            subplot(2,3,(4,6))
            # true components
            tab_labels = ['$x_1$', '$x_2$', '$x_3$', '$z_1$']
            plot(t, x_t[:,i_unobs_comp], '--k')
            plot(t, x_t[:,1], color='C0')
            plot(t, x_t[:,2], color='C1')
            plot(t, x_s[:,2], color='C2')
            legend(tab_labels, loc=1, fontsize='xx-large')
            ylim([-30,45])
            xlim([t[0],t[-1]])
            fill_between(t, x_s[:,2]-1.96*sqrt(P_s[:,2,2]), x_s[:,2]+1.96*sqrt(P_s[:,2,2]), facecolor='C2', alpha=0.25)
            xlabel('Time', size=20)
            ylabel('Lorenz components', size=20)
            ylim([-30,45])
            xlim([t[0],t[-1]])
            savefig('/home/administrateur/Dropbox/Applications/Overleaf/presentation_buenos_aires_2023_02_10/L63_' + format(i+1, '03d') + '.png', bbox_inches='tight', dpi=50)
            close()
        '''
    
    return x_s, P_s, M, tab_loglik, x_out, x_f, Q
    
def Kalman_SEM_bis(x, y, H, R, nb_iter_SEM, M_init, Q_init):
    """ Apply the stochastic expectation-maximization algorithm. """
    
    # fix the seed
    random.seed(11)
    
    # copy x
    x_out = x.copy()
    
    # shapes
    n = shape(x_out)[1]
    T, p = shape(y)
    
    # tab to store the log-likelihood
    tab_loglik = [] 
    
    # loop on the SEM iterations
    for i in tqdm(arange(0, nb_iter_SEM)):

        # Kalman initialization
        if i==0:
            x0 = zeros(n)
            P0 = eye(n)
            M  = M_init
            Q  = Q_init
        else:
            x0 = x_s[0,:]
            P0 = P_s[0,:,:,]
        
        # apply the Kalman smoother
        x_f, P_f, x_a, P_a, x_s, P_s, loglik, P_s_lag = Kalman_smoother(y, x0, P0, M, Q, H, R)
        
        # update the Kalman parameters
        A = zeros((n,n))
        for k in arange(0,T-1):
            A += P_s[k,:,:] + array([x_s[k,:]]).T @ array([x_s[k,:]])
        B = zeros((n,n))
        for k in arange(0,T-1):
            B += P_s_lag[k,:,:] + array([x_s[k+1,:]]).T @ array([x_s[k,:]])
        C = zeros((n,n))
        for k in arange(0,T-1):
            C += P_s[k+1,:,:] + array([x_s[k+1,:]]).T @ array([x_s[k+1,:]])
        M = B @ inv(A)
        Q = (C - M @ B.T) / (T-1)
                
        # store the log-likelihod
        tab_loglik = append(tab_loglik, sum(loglik))
        
        # simulate the new x
        for k in range(len(x_s)):
            x_out[k,:] = random.multivariate_normal(x_s[k,:], P_s[k,:,:])
        
    return x_s, P_s, M, tab_loglik, x_out, x_f
    
def ensemble_Kalman_filter(y, x0, P0, m, Q, H, R, Ne):
    """ Apply the ensemble Kalman filter (stochastic version). """
    
    # shapes
    n = shape(x0)[1] # n = len(x0)
    T, p = shape(y)

    # Kalman initialization
    x_f = zeros((T,n))   # forecast state
    P_f = zeros((T,n,n)) # forecast error covariance matrix
    x_a = zeros((T,n))   # analysed state
    P_a = zeros((T,n,n)) # analysed error covariance matrix
    loglik = zeros((T))  # log-likelihood
    x_f_tmp = zeros((n,Ne)) # members of the forecast
    y_f_tmp = zeros((p,Ne)) # members of the perturbed observations
    x_a_tmp = zeros((n,Ne)) # members of the analysis
    x_a_tmp = random.multivariate_normal(squeeze(x0), P0, Ne).T
    x_a[0,:]   = mean(x_a_tmp, 1)
    P_a[0,:,:] = cov(x_a_tmp)
        
    # apply the ensemble Kalman filter
    for k in range(1,T):
       
        # prediction step
        for i in range(Ne):
            x_f_tmp[:,i] = m(x_a_tmp[:,i]) + random.multivariate_normal(zeros(n), Q)
            y_f_tmp[:,i] = H @ x_f_tmp[:,i] + random.multivariate_normal(zeros(p), R)
        P_f[k,:,:] = cov(x_f_tmp)
        
        # Kalman gain
        K = P_f[k,:,:] @ H.T @ inv(H @ P_f[k,:,:] @ H.T + R)
        
        # update step
        if(sum(isfinite(y[k,:])) > 0): # if observations are available
            for i in range(Ne):
                x_a_tmp[:,i] = x_f_tmp[:,i] + K @ (y[k,:] - y_f_tmp[:,i])
            P_a[k,:,:] = cov(x_a_tmp)
        else:
            x_a_tmp = x_f_tmp
            P_a[k,:,:] = P_f[k,:,:]
        x_a[k,:] = mean(x_a_tmp, 1)
        
        # stock the log-likelihood
        loglik[k] = -0.5*((y[k,:] - H @ x_f[k,:]).T @ inv(H @ P_f[k,:,:] @ H.T + R) @ (y[k,:] - H @ x_f[k,:])) - 0.5 * (n * log( 2 * np.pi)+ log(det(H @ P_f[k,:,:] @ H.T + R)))

    return x_f, P_f, x_a, P_a, loglik
