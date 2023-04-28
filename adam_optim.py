import torch
import numpy as np
class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def updates(self, t, weight, b, dw, db):
        ## dw, db are from current minibatch
        #weights 
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)

        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db)

    
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)

    
        weight = weight - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return weight, b
def lossfunction(m):
       return m**2-2*m+1
## take derivative

def gradfunction(m):
    return 2*m-2
def checkconvergence(w0, w1):
    return (w0 == w1)
w_0 = 0
b0 = 0
adam = AdamOptim()
t = 1 
converged = False

while not converged:
    dw = gradfunction(w_0)
    db = gradfunction(b0)
    w_0_old = w_0
    w_0, b0 = adam.updates(t,weight=w_0, b=b0, dw=dw, db=db)
    if checkconvergence(w_0, w_0_old):
        print('converged after '+str(t)+' iterations')
        break
    else:
        print('iteration '+str(t)+': weight='+str(w_0))
        t+=1