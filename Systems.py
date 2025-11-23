import numpy as np
from smt.sampling_methods import LHS
from math import exp, sin
from utils import RK4
import torch

"""
Systems are implemented by defining 6 essential parameters:

function: The system function describing its dynamics.

output: The measureable outputs of the system.

input: Input to the system for non-autonomous systems. If the system is autonomous, input is None.

x_size: Dimension of the system.

y_size: Dimension of the output.

z_size: Dimension of the transformed system.

"""

def add_process_noise(method):
    def method_wrapper(self, t, u, x):
        x_dot = method(self, t, u, x)
        if self.add_noise:
            w, _ = self.gen_noise()
            return x_dot + w
        return x_dot

    return method_wrapper

def add_measurement_noise(method):
    def method_wrapper(self, x):
        y = method(self, x)
        if self.add_noise:
            _, v = self.gen_noise()
            return y + v
        return y

    return method_wrapper

class System:
    def __init__(self, function, output, add_noise, noise_process_mean, noise_process_std, noise_measurement_mean, noise_measurement_std):
        self.function = function
        self.output = output
        self.add_noise = add_noise
        self.noise = 0  
        self.noise_process_mean = noise_process_mean
        self.noise_process_std = noise_process_std
        self.noise_measurement_mean = noise_measurement_mean
        self.noise_measurement_std = noise_measurement_std
        
    # LHS Sampling
    def sample_ic(self, sample_space, samples, seed):
        return LHS(xlimits = sample_space, random_state = seed)(samples)
                     
    def simulate(self,a, b, N, v):
        x, t = RK4(self.function, a, b, N, v, self.input)
        return np.array(x), t
    
    def generate_data(self, ic, a, b, N):
        data = []
        output = []
        for i in range(0, np.size(ic, axis = 0)):
            x, time = self.simulate(a,b,N,ic[i])
            temp = []
            for j in x:
                temp.append(self.output(j))
            data.append(x)    
            output.append(np.array(temp))
      
        return np.array(data), np.array(output), time   

    def function_output(self, x):
        if torch.is_tensor(x[0]):
            return torch.tensor(x)
        else:
            return np.array(x)

    def gen_noise(self):
        # To generate process and measurement noise
        x_noise = np.random.normal(self.noise_process_mean, self.noise_process_std, (self.x_size))
        y_noise = np.random.normal(self.noise_measurement_mean, self.noise_measurement_std, (self.y_size))  
        if self.y_size == 1:
            y_noise = y_noise[0]

        return x_noise, y_noise

    def toggle_noise(self):
        if self.add_noise:
            self.add_noise = False
        else:
            self.add_noise = True
            
    def __contains__(self, x):
        y_variables = []
        for i in range(1, self.y_size + 1):
            y_variables.append(f'y{i}')

        if x in y_variables:
            return True
        return False

# --------------- Autonomous Systems --------------- 

# Reverse Duffing Oscillator
class RevDuff(System):
    def __init__(self, zdim, add_noise = False, noise_mean = 0, noise_std = 0.01, noise_measurement_mean = 0, noise_measurement_std = 0.01):
        self.y_size = 1
        self.x_size = 2
        if zdim == 5:
            self.z_size = self.y_size*(2*self.x_size + 1)
        if zdim == 3:
            self.z_size = self.y_size*(1*self.x_size + 1)           
        self.input = None
        super().__init__(self.function, self.output, add_noise, noise_mean, noise_std, noise_measurement_mean, noise_measurement_std)
        
    def function(self, t, u, x):
        x1 = x[0]
        x2 = x[1]
    
        x1_dot = x2**3
        x2_dot = -x1

        if self.add_noise:
           self.noise = self.gen_noise()[0]

        return self.function_output([x1_dot, x2_dot]) + self.noise

    def output(self, x):
        y = x[0]
        noise, process_noise = 0,0
        if self.add_noise:
            noise = self.gen_noise()[1]
            #process_noise = self.gen_noise()[0][0]

        return y + noise #+ process_noise
    
        
# Network SIS
class SIS(System):
    def __init__(self, sample_space, A, B, G, C):
        self.A = A
        self.B = B
        self.G = G
        self.C = C
        self.x_size = self.A.shape[0]
        self.y_size = self.C.shape[0]
        self.z_size = self.y_size*(self.x_size + 1)
        self.function = lambda u, x: (B@A - G)@x - np.diag(x)@B@A@x    # x = np.array([a, b, c,....]])
        self.output = lambda x: C@x
        self.input = None
        super().__init__(self.function, self.output, sample_space)
        
# Van der Pol Oscillator
class VdP(System):
    def __init__(self, zdim, my = 3, add_noise = False, noise_mean = 0, noise_std = 0.01, noise_measurement_mean = 0, noise_measurement_std = 0.01):
        self.x_size = 2
        self.y_size = 1
        if zdim == 5:
            self.z_size = self.y_size*(2*self.x_size + 1)
        if zdim == 3:
            self.z_size = self.y_size*(1*self.x_size + 1) 
        self.my = my
        self.input = None
        super().__init__(self.function, self.output, add_noise, noise_mean, noise_std, noise_measurement_mean, noise_measurement_std)
        
    def function(self, t, u, x):
        x1 = x[0]
        x2 = x[1]
            
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1

       # if self.add_noise:
         #   self.noise = self.gen_noise()[0]
            
        return self.function_output([x1_dot, x2_dot]) #+ self.noise
        
    def output(self, x):
        y = x[0]
        noise, process_noise = 0, 0
        if self.add_noise:
           noise = self.gen_noise()[1]
           process_noise = self.gen_noise()[0][0]

        return y + noise + process_noise
        
# VDP with changing mu
class Vdp_params(System):
    def __init__(self, add_noise = False, noise_mean = 0, noise_std = 0.01):
        self.x_size = 3
        self.y_size = 1
        self.param_size = 1
        self.z_size = self.y_size*(2*(self.x_size - self.param_size) + 1) + self.y_size*(2*self.param_size + 1) # dim = ny*(2*nx + 1) + ny*(2*p + 1)
        #self.z_size = 9
        self.input = None
        self.add_noise = add_noise
        self.noise = 0  
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        super().__init__(self.function, self.output)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        my = x[2]
            
        x1_dot = x2
        x2_dot = my*(1 - x1**2)*x2 - x1
        my_dot = 0

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
            
        #return np.array([x1_dot, x2_dot, my_dot]) + self.noise
        ###################################
        return self.function_output([x1_dot, x2_dot, my_dot])
        
    def output(self, x):
        y = x[0]

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]

        return y + self.noise

# Polynomial system
class Polynomial(System):
    def __init__(self):
        self.x_size = 2
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = None
        super().__init__(self.function, self.output)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        
        x1_dot = x1 - (1/3)*x1**3 - x1*x2**2
        x2_dot = x1 - x2 - (1/3)*x2**3 - x2*x1**2
        
        return np.array([x1_dot, x2_dot])
    
    def output(self, x):
        y = x[0]
            
        return y      
        
# Chua's Circuit
class Chua(System):
    def __init__(self, alpha, beta, gamma, a, b, add_noise = False, noise_mean = 0, noise_std = 0.01):
        super().__init__(self.function, self.output, add_noise, noise_mean, noise_std)
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.g = lambda x: 0.5*(a - b)*(np.abs(x[0] + 1) - np.abs(x[0] - 1))
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.a = a
        self.b = b
        self.input = None
        
    def function(self, t, u, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        x1_dot = self.alpha*(x2 - x1*(1 + self.b) - self.g(x))
        x2_dot = x1 - x2 + x3
        x3_dot = -self.beta*x2 - self.gamma*x3

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
        
        return self.function_output([x1_dot, x2_dot, x3_dot])

    def output(self, x):
        y = x[2]

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]

        return y + self.noise

# Smooth Chua's Circuit
class Chua_Smooth(System):
    """
    Source: https://www.math.spbu.ru/user/nk/PDF/2012-Physica-D-Hidden-attractor-Chua-circuit-smooth.pdf
    """
    def __init__(self, alpha, beta, gamma, m0, m1):
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.g = lambda x: m1*x + (m0 - m1)*np.tanh(x)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m0 = m0
        self.m1 = m1
        self.input = None
        super().__init__(self.function, self.output)  
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        x1_dot = self.alpha*(x2 - x1) - self.alpha*self.g(x1)
        x2_dot = x1 - x2 + x3
        x3_dot = -self.beta*x2 - self.gamma*x3
        
        return np.array([x1_dot, x2_dot, x3_dot])
    
    def output(self, x):
        y = x[2]
        return y
    
# Rössler's System
class Rossler(System):
    def __init__(self, a, b, c, add_noise = False, noise_mean = 0, noise_std = 0.1, noise_measurement_mean = 0, noise_measurement_std = 0.1):
        super().__init__(self.function, self.output, add_noise, noise_mean, noise_std, noise_measurement_mean, noise_measurement_std)
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.input = None
        self.a = a
        self.b = b
        self.c = c
        
    def function(self, t, u, x):
        x1 = x[0]
        x2 = x[1] 
        x3 = x[2]
        
        x1_dot = -(x2 + x3)
        x2_dot = x1 + self.a*x2
        x3_dot = self.b + x3*(x1 - self.c)
        
      #  if self.add_noise:
            #self.noise = self.gen_noise()[0]

        return self.function_output([x1_dot, x2_dot, x3_dot]) #+ self.noise
    
    def output(self, x):
        y = x[1]
        process_noise = 0
        if self.add_noise:
            self.noise = self.gen_noise()[1]
            process_noise =  self.gen_noise()[0]

        return y + self.noise #+ process_noise

# Rössler's System with changing parameters
class Rossler_params(System):
    def __init__(self,add_noise = False, noise_mean = 0, noise_std = 0.3):
        self.x_size = 6
        self.y_size = 2
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.input = None
        self.add_noise = add_noise
        self.noise = 0  
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        super().__init__(self.function, self.output)  
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1] 
        x3 = x[2]
        a0 = x[3]
        b0 = x[4]
        c0 = x[5]
        
        x1_dot = -(x2 + x3)
        x2_dot = x1 + a0*x2
        x3_dot = b0 + x3*(x1 - c0)
        a0_dot = 0 
        b0_dot = 0
        c0_dot = 0
        
        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]

        return np.array([x1_dot, x2_dot, x3_dot, a0_dot, b0_dot, c0_dot]) + self.noise
    
    def output(self, x):
        y = np.array([x[1], x[2]])

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[1]

        return y + self.noise

# SIR
class SIR(System):
    def __init__(self, beta, gamma, N):
        self.x_size = 3
        self.y_size = 2
        self.z_size = self.y_size*(self.x_size + 1)
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.input = None
        super().__init__(self.function, self.output)  
      
    def function(self, u, x):
        S = x[0]
        I = x[1]
        R = x[2]
        
        S_dot = -self.beta*I*S/self.N
        I_dot = self.beta*I*S/self.N - self.gamma*I
        R_dot = self.gamma*I
        
        return np.array([S_dot, I_dot, R_dot])
    
    def output(self, x):
        S = x[0]
        I = x[1]
        R = x[2]
        
        y = np.array([R, S+I+R])

        return y
    
class Network_SIR(System):
    def __init__(self, D, W, G, C):
        self.x_size = 10
        self.y_size = 5
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.D = D
        self.W = W
        self.G = G
        self.C = C
        self.input = None
        super().__init__(self.function, self.output)  

    def function(self,u ,x):
        S = x[0: int(self.x_size / 2)]
        I = np.expand_dims(x[int(self.x_size  / 2):], axis = 1)

        S_dot = -np.diag(S) @ self.D @ self.W @ I
        I_dot = np.diag(S) @ self.D @ self. W @ I - self.G @ I
        
        x_dot = np.concatenate((S_dot, I_dot), axis = 0)
        x_dot = np.squeeze(x_dot)

        return x_dot

    def output(self, x):
        return self.C @ x

# Lorenz system
class Lorenz(System):
    def __init__(self, rho, sigma, beta, add_noise = False, noise_process_mean = 0, noise_process_std = 0.01, noise_measurement_mean = 0, noise_measurement_std = 0.01):
        super().__init__(self.function, self.output, add_noise, noise_process_mean, noise_process_std, noise_measurement_mean, noise_measurement_std)
        self.x_size = 3
        self.y_size = 1
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.input = None
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.add_noise = add_noise
        self.noise = 0

    def function(self, t, u, x):
        x1 = x[0]
        x2 = x[1] 
        x3 = x[2]

        x1_dot = self.sigma*(x2 - x1)
        x2_dot = x1*(self.rho - x3) - x2
        x3_dot = x1*x2 - self.beta*x3

     #   if self.add_noise:
           # self.noise = self.gen_noise()[0]

        return self.function_output([x1_dot, x2_dot, x3_dot]) #+ self.noise

    def output(self, x):
        if self.add_noise:
            self.noise = self.gen_noise()[1]
            process_noise = self.gen_noise()[0]

        return x[1] + self.noise + process_noise[1]

class epidemic_SIR(System):
    def __init__(self, add_noise = False, noise_mean = 0, noise_std = 0.01):
        super().__init__(self.function, self.output)
        self.x_size = 4
        self.y_size = 1
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.input = None
        self.add_noise = add_noise
        self.noise = 0  
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def function(self, u, x):
        S = x[0]
        I = x[1]
        beta = x[2]
        gamma = x[3]

        S_dot = -beta*S*I
        I_dot = beta*S*I - gamma*I
        beta_dot = 0
        gamma_dot = 0

        if self.add_noise:
            self.noise = self.gen_noise(self.noise_mean, self.noise_std)[0]
        
        return np.array([S_dot, I_dot, beta_dot, gamma_dot]) + self.noise

    def output(self, x):
        I = x[1]
        gamma = x[3]
        y = gamma*I
        
        return y + self.noise

class Kuramoto(System):
    def __init__(self, nodes, K, adj_mat = None, add_noise = False, noise_process_mean = 0, noise_process_std = 0.01, noise_measurement_mean = 0, noise_measurement_std = 0.01):
        super().__init__(self.function, self.output, add_noise, noise_process_mean, noise_process_std, noise_measurement_mean, noise_measurement_std)
        self.x_size = nodes
        self.y_size = 5
        self.z_size = self.y_size*(2*self.x_size + 1)
        self.input = None
        self.K = K
        #self.w = np.random.normal(size = nodes)
        self.w = np.array([-0.9582865, -0.38394109, 1.33269242, -0.41366157, 1.44836661, 0.37739606, -0.21925155, -1.22455211 ,0.78344887, 1.00446236])
        self.adj_mat = adj_mat

   # @add_process_noise
    def function(self, t, u, x):
        if torch.is_tensor(x[0]):
            method = torch
            A = torch.tensor(self.adj_mat).cuda()
            w = torch.from_numpy(self.w).cuda()
        else:
            method = np
            A = self.adj_mat
            w = self.w

        theta_vec = x
        theta_i, theta_j = method.meshgrid(theta_vec, theta_vec)
        connections = A*method.sin(theta_j - theta_i)   
        theta_dot = w + (self.K/self.x_size)*method.sum(connections, axis = 0)

        return self.function_output(theta_dot)

   # @add_measurement_noise
    def output(self, x):
        y1 = x[0]
        y2 = x[1]
        y3 = x[2]
        y4 = x[3]
        y5 = x[4]

        return np.array([y1, y2, y3, y4, y5])
    
# New England Power System
class New_England_Power(System):
    def __init__(self, G, B, E, H, D, Pm, omega0, add_noise = False, noise_mean = 0, noise_std = 0.01, noise_measurement_mean = 0, noise_measurement_std = 0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_size = 10
        self.x_size = 20
        self.z_size = self.y_size*(2*self.x_size + 1)       
        self.n = self.x_size//2  
        # self.G = G.to(device)
        # self.B = B.to(device)
        # self.E = E.to(device)
        # self.H = H.to(device)
        # self.D = D.to(device)
        # self.Pm = Pm.to(device)
        # self.omega0 = omega0.to(device)

        self.G = G
        self.B = B
        self.E = E
        self.H = H
        self.D = D
        self.Pm = Pm
        self.omega0 = omega0
        self.input = None
        super().__init__(self.function, self.output, add_noise, noise_mean, noise_std, noise_measurement_mean, noise_measurement_std)
        
    def function(self, t, u, x):
        x_dot = np.zeros(self.x_size)
        omega = x[::2]
        delta = x[1::2]
        for i in range(self.n):
            omega_dot = self.omega0/(2*self.H[i])*(self.Pm - self.D*(omega[i] - self.omega0)/self.omega0 - 
                                               self.E[i]**2*self.G[i,i] - 
                                               np.sum([self.E[i]*self.E[j]*(self.B[i,j]*np.sin(delta[i] - delta[j]) + self.G[i,j]*np.cos(delta[i] - delta[j])) for j in range(self.n) if j != i]))
            
            delta_dot = omega[i] - self.omega0
            x_dot[i], x_dot[i+1] = omega_dot, delta_dot

        return self.function_output(x_dot)

    def output(self, x):
        y = x[:10]
        return y 


# --------------- Non-Autonomous Systems --------------- 

# Non-Autonomous Reverse Duffing Oscillator
class RevDuff_NA(System):
    def __init__(self, input):
        self.y_size = 1
        self.x_size = 2
        self.z_size = self.y_size*(self.x_size + 1)
        self.input = input
        super().__init__(self.function, self.output)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
    
        x1_dot = x2**3
        x2_dot = -x1 + u
    
        return np.array([x1_dot, x2_dot])
    
    def output(self, x):
        y = x[0]
        
        return y
    
    def add_train_input(self, train_input):
        self.add_train_input = train_input

# Non-Autonomous Van der Pol Oscillator
class VdP_NA(System):
    def __init__(self, input, my = 3):
        self.x_size = 2
        self.y_size = 1
        self.z_size = self.y_size*(self.x_size + 1)
        self.my = my
        self.input = input
        super().__init__(self.function, self.output)
        
    def function(self, u, x):
        x1 = x[0]
        x2 = x[1]
        x1_dot = x2
        x2_dot = self.my*(1 - x1**2)*x2 - x1 + u
            
        return np.array([x1_dot, x2_dot])
        
    def output(self, x):
        y = x[0]
        return y
        
    def add_train_input(self, train_input):
        self.add_train_input = train_input
        
        
        
        
        
        
        
        
        