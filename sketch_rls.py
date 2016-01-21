
from __future__ import division, print_function
import numpy as np
import scipy.linalg as la
from adaptive_filters import *

import matplotlib.pyplot as plt

class SketchRLS(AdaptiveFilter):
    
    def __init__(self, length, lmbd=0.999, delta=10, N=5, p=0.1, mem=0, sketch='RowSample'):
        
        # RLS parameters
        self.lmbd = lmbd
        self.delta = delta
        
        if delta == 0:
            raise ValueError('Delta should be a positive constant.')
        
        # sketching parameters
        self.sketch = sketch
        self.N = N # number of sketches
        self.p = p # update probability
        self.q = 1 - np.exp(np.log(1-self.p)/self.N)

        # buffer to keep old matrix update
        self.mem_buf_size = mem
        
        AdaptiveFilter.__init__(self, length) 
        self.reset()
        
    def reset(self):
        AdaptiveFilter.reset(self)
        
        self.P = np.zeros((self.N, self.length, self.length))
        for i in xrange(self.N):
            self.P[i,:,:] = np.eye(self.length)/self.delta

        self.last_update = np.zeros(self.N)
        
        self.R = self.delta*np.eye(self.length)
        self.r_dx = np.zeros((self.length))
        
        self.buf_len = 4*int(1/self.p)
        self.d = np.zeros((self.buf_len))
        self.x = np.zeros((self.length-1 + self.buf_len))
        self.buf_ind = 0
        
        # buffer to keep old matrix update
        self.mem_buf = np.zeros((self.mem_buf_size, self.length))
        self.out_buf = np.zeros((self.length,self.length))

        # precompute powers of lambda
        self.lmbd_pwr = self.lmbd**np.arange(2*int(1/self.p))


    def update_buf(self, x_n, d_n):
        
        # check for buffer overflow
        if self.buf_len == self.buf_ind:
            self.x = np.concatenate((self.x, np.zeros(int(1/self.p))))
            self.d = np.concatenate((self.d, np.zeros(int(1/self.p))))
            self.buf_len += int(1/self.p)
        
        # Update the internal buffers
        self.x[(self.length-1)+self.buf_ind] = x_n
        self.d[self.buf_ind] = d_n
        self.buf_ind += 1
        
        
    def update(self, x_n, d_n):

        self.n += 1
        self.update_buf(x_n, d_n)
        
        # get this randomness
        pr = np.random.uniform(size=(self.N)) 
        I = np.arange(self.N)[pr < self.q]
        
        if I.shape[0] > 0:
            
            # extract data from buffer
            L = self.buf_ind
            x = self.x[(self.length-1)+self.buf_ind-1::-1]
            d = self.d[self.buf_ind-1::-1]
            
            # compute more power of lambda if needed
            if L >= self.lmbd_pwr.shape[0]:
                self.lmbd_pwr = np.concatenate((self.lmbd_pwr, self.lmbd**np.arange(self.lmbd_pwr.shape[0], L+1)))
            
            # block-update parameters
            X = la.hankel(x[:L],r=x[L-1:])
            Lambda_diag = self.lmbd_pwr[:L]
            lmbd_L = self.lmbd_pwr[L]
            
            """ As of now, using BLAS dot is more efficient than the FFT based method toeplitz_multiplication.
            rhs = (self.lmbd**np.arange(L) * X.T).T
            self.R = (self.lmbd**L)*self.R + hankel_multiplication(x[:self.length], x[self.length-1:], rhs)
            """ # Use thus dot instead
            
            # block-update the auto-correlation matrix 
            rhs = (Lambda_diag * X.T).T
            self.R = lmbd_L*self.R + np.dot(X.T, rhs)
            
            # block-update of the cross-correlation vector here
            self.r_dx = lmbd_L*self.r_dx + np.dot(X.T, Lambda_diag*d)
            
            # update the sketches as needed
            for i in np.arange(self.N):

                if pr[i] < self.q:

                    self.last_update[i] = self.n


                    buf = np.zeros(self.length)
                    if self.sketch == 'RowSample':
                        buf = x[:self.length] 
                    elif self.sketch == 'Binary':
                        data = (np.sqrt(Lambda_diag) * X.T)
                        buf = np.dot(data, np.random.choice([-1,1], size=self.L))*np.sqrt(self.L)
                    x_up = buf.copy()

                    if self.mem_buf_size > 0:
                        for row in self.mem_buf:
                            x_up += np.random.choice([-1,1])*row*np.sqrt(lmbd_L)

                        # update the sketch buffer
                        self.mem_buf *= np.sqrt(lmbd_L)
                        self.mem_buf[1:,:] = self.mem_buf[0:-1,:]
                        self.mem_buf[0,:] = buf

                    # update the inverse correlation matrix
                    mu = 1/lmbd_L
                    pi = np.dot(self.P[i,:,:], x_up)

                    denom = (self.q*lmbd_L + np.inner(x_up, pi))
                    np.outer(pi, pi/denom, out=self.out_buf)
                    self.P[i,:,:] = self.P[i,:,:] - self.out_buf
                    self.P[i,:,:] *= mu
                    
                # do IHS
                self.w = self.w + \
                        np.dot(self.P[i,:,:], self.r_dx - np.dot(self.R, self.w))/self.n

            # Flush the buffers
            self.x[:self.length-1] = self.x[self.buf_ind:self.buf_ind+self.length-1]
            self.x[self.length-1:] = 0
            self.d[:] = 0
            self.buf_ind = 0


