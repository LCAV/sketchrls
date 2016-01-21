
from __future__ import division, print_function
import scipy.linalg as la
import numpy as np
# First the classic RLS (for real numbered signals)

class AdaptiveFilter:
    
    def __init__(self, length):
        
        # filter length
        self.length = length
        
        self.reset()
    
    def reset(self):
        
        # index
        self.n = 0
        
        # filter
        self.w = np.zeros((self.length))
        
        # system input signal
        self.x = np.zeros((self.length))
        
        # reference signal
        self.d = np.zeros((1))
        
    def update(self, x_n, d_n):
        
        self.n += 1
        
        # update buffers
        self.x[1:] = self.x[0:-1]
        self.x[0] = x_n
        self.d = d_n
        
    def name(self):
        
        return self.__class__.__name__


class RLS(AdaptiveFilter):
    
    def __init__(self, length, lmbd=0.999, delta=10):
        
        self.lmbd = lmbd
        self.lmbd_inv = 1/lmbd
        self.delta = delta
        self.x_buf_size = 10*length
        
        AdaptiveFilter.__init__(self, length)   
    
        self.reset()

    def reset(self):
        AdaptiveFilter.reset(self)
        
        if self.delta == 0:
            raise ValueError('Delta should be a positive constant.')
        else:
            self.P = np.eye(self.length)/self.delta

        self.x_buf = np.zeros(self.x_buf_size)
        self.x_buf_ind = self.length-1

        self.outer_buf = np.zeros((self.length, self.length))
        self.g = np.zeros(self.length)

    def update_buf(self, x_n):

        if self.x_buf_ind == self.x_buf_size:
            self.x_buf[:self.length-1] = self.x_buf[self.x_buf_size-self.length+1:]
            self.x_buf_ind = self.length-1

        self.x_buf[self.x_buf_ind] = x_n
        if self.x_buf_ind == self.length-1:
            self.x = self.x_buf[self.x_buf_ind::-1]
        else:
            self.x = self.x_buf[self.x_buf_ind:self.x_buf_ind-self.length:-1]
        self.x_buf_ind += 1
        
    
    def update(self, x_n, d_n):
        
        # update buffers
        #AdaptiveFilter.update(self, x_n, d_n)
        self.n += 1

        self.update_buf(x_n)
        self.d = d_n

        # a priori estimation error
        alpha = (self.d - np.inner(self.x, self.w))
        
        # update the gain vector
        np.dot(self.P, self.x*self.lmbd_inv, out=self.g)
        denom = 1 + np.inner(self.x, self.g)
        self.g /= denom
        
        # update the filter
        self.w += alpha*self.g
        
        # update P matrix
        np.outer(self.g, np.inner(self.x, self.P), out=self.outer_buf)
        self.P -= self.outer_buf
        self.P *= self.lmbd_inv

        
class NLMS(AdaptiveFilter):
    
    def __init__(self, length, mu=0.01):
        
        self.mu = mu
        AdaptiveFilter.__init__(self, length)
        
    def update (self, x_n, d_n):
        
        AdaptiveFilter.update(self, x_n, d_n)
        
        e = self.d - np.inner(self.x, self.w)
        self.w += self.mu*e*self.x/np.inner(self.x,self.x)
        

class BlockRLS(RLS):
    
    def __init__(self, length, lmbd=0.999, delta=10, L=1):
        
        # sketching parameters
        self.L = L # block size
        
        RLS.__init__(self, length, lmbd=lmbd, delta=delta) 
        self.reset()
        
    def reset(self):
        RLS.reset(self)
        # We need to redefine these two guys
        self.d = np.zeros((self.L))
        self.x = np.zeros((self.L+self.length-1))
        
        
    def update(self, x_n, d_n):
        
        # Update the internal buffers
        self.n += 1
        slot = self.L - ((self.n-1) % self.L) - 1
        self.x[slot] = x_n
        self.d[slot] = d_n
        
        # Block update
        if self.n % self.L == 0:
            
            # block-update parameters
            X = la.hankel(self.x[:self.L],r=self.x[self.L-1:])
            Lambda_diag = (self.lmbd**np.arange(self.L))
            lmbd_L = self.lmbd**self.L
            
            alpha = self.d - np.dot(X, self.w)
            
            pi = np.dot(self.P, X.T)
            g = np.linalg.solve((lmbd_L*np.diag(1/Lambda_diag) + np.dot(X,pi)).T, pi.T).T
            
            self.w += np.dot(g, alpha)
            
            self.P = self.P/lmbd_L - np.dot(g, pi.T)/lmbd_L
            
            # Remember a few values
            self.x[-self.length+1:] = self.x[0:self.length-1]

class BlockLMS(NLMS):
    
    def __init__(self, length, mu=0.01, L=1, nlms=False):
        
        self.nlms = nlms
        
        # sketching parameters
        self.L = L # block size
        
        NLMS.__init__(self, length, mu=mu)
        self.reset()
        
    def reset(self):
        NLMS.reset(self)
        # We need to redefine these two guys
        self.d = np.zeros((self.L))
        self.x = np.zeros((self.L+self.length-1))
        
    def update(self, x_n, d_n):
        
        # Update the internal buffers
        self.n += 1
        slot = self.L - ((self.n-1) % self.L) - 1
        self.x[slot] = x_n
        self.d[slot] = d_n
        
        # Block update
        if self.n % self.L == 0:
            
            # block-update parameters
            X = la.hankel(self.x[:self.L],r=self.x[self.L-1:])
            
            e = self.d - np.dot(X, self.w)
            
            if self.nlms:
                norm = np.linalg.norm(X, axis=1)**2
                if self.L == 1:
                    X = X/norm[0]
                else:
                    X = (X.T/norm).T
                
            self.w += self.mu*np.dot(X.T, e)
            
            # Remember a few values
            self.x[-self.length+1:] = self.x[0:self.length-1]
        
