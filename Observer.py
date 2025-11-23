import torch
import numpy as np
import utils as data
import control as ct
import matplotlib.pylab as plt
from torch.autograd.functional import jacobian
from smt.sampling_methods import LHS

class System_z:
    """
    Dynamics of the z-system for both autonomous and non-autonomous cases.
    Autonomous:
        z_dot = Mz(t) + Ky(t)
        
    Non-Autonomous:
        z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
        phi(t, z(t)) = dT/dx*g

    """
    def __init__(self, M, K, system):
        self.M = M
        self.K = K
        self.is_autonomous = True if system.input == None else False
        self.y_size = system.y_size

    def z_dynamics(self):
        if self.is_autonomous:
            if self.y_size > 1:
                z_dot = lambda y,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1))
            else:
                z_dot = lambda y,z: np.matmul(self.M,z) + self.K*y 
        
        else:
            if self.y_size > 1:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + np.matmul(self.K, np.expand_dims(y,1)) + q
            else:
                z_dot = lambda y,q,z: np.matmul(self.M,z) + self.K*y + q 

        return z_dot

class Observer:
    def __init__(self, system, M, K, net, a ,b ,N, init_z_zero = True):
        self.system = system
        self.z_system = System_z(M, K, system)
        #self.net = net #######
        self.f = self.z_system.z_dynamics()
       # self.T = net.net1
        self.T_inv = net
        self.a = a
        self.b = b
        self.N = N
        self.init_z_zero = init_z_zero
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def simulate_NA(self, ic, u0, g):
        """
        Online simulation of observer for non-autonomous input-affine systems by the following steps:
        1. Generate y data from the system with input u.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t) + phi(t, z(t))*(u(t) - u0(t))
            phi(t, z(t)) = dT/dx*g
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)

        """
        x, y, t = self.system.generate_data(ic, self.a, self.b, self.N)     # Generate y data
        x = torch.from_numpy(np.reshape(x, (self.N+1,self.system.x_size)))
        u = self.system.input
        size_z = self.z_system.M.shape[0]
        h = (self.b-self.a) / self.N

        z = [[0]*size_z]
        v = np.array(z).T

        y = np.squeeze(y)
        if y.ndim > 2:
            y = y[1:, :]
        else:
            y = np.delete(y,0)

        for idx, output in enumerate(y):
            with torch.no_grad():
                x_hat = self.T_inv(torch.tensor(z[-1]).float())
            u_sub_u0 = u(t[idx]) - u0(t[idx])
            dTdx = jacobian(self.T, x_hat).numpy()
            dTdx_mul_g = np.matmul(dTdx, g)

            q = dTdx_mul_g*u_sub_u0     # phi*(u(t) - u0(t))

            k1 = self.f(output,q, v)
            k2 = self.f(output,q, v + h/2*k1)
            k3 = self.f(output,q, v + h/2*k2)
            k4 = self.f(output,q, v + h*k3)
            
            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            z.append(np.ndarray.tolist(a)) 

        z = np.array(z)
        z = torch.from_numpy(z).float()
        with torch.no_grad():
            x_hat = self.T_inv(z)

        error = torch.abs(x-x_hat)

        return x, x_hat, t, error

    def simulate(self, ic, add_noise = False, faults = None):
        """
        Online simulation of observer for autonomous systems by the following steps:
        1. Generate y data from system.
        2. Use the y data to simulate observer dynamics:
            z_dot = Mz(t) + Ky(t)
        3. Use T_inv, the inverse of T to simulate:
            x_hat = T_inv(t, z)
        """
        fault_signals = 0
        if add_noise:
            self.system.add_noise = True
        else:
            self.system.add_noise = False
            
        x_truth, _, _ = self.system.generate_data(ic, self.a, self.b, self.N)

        _, y, time = self.system.generate_data(ic, self.a, self.b, self.N)
        
        # Inject faulty measurements
        if faults != None:
            fault_signals = []
            for fault in faults:
                y, fault_signal = self.inject_fault(y, fault, time)
                fault_signals.append(fault_signal)

        #x_truth = torch.from_numpy(np.reshape(x_truth, (self.N+1,self.system.x_size)))
        
        if self.init_z_zero:
            ic_z = np.zeros([1,self.system.z_size])
        else:
            with torch.no_grad():
                ic_z = self.T(torch.from_numpy(ic)).cpu().numpy()

        z = data.KKL_observer_data(self.z_system.M, self.z_system.K, y, self.a, self.b, ic_z, self.N)
        z = torch.from_numpy(z).view(self.N+1,self.system.z_size).float()
        z = z.to(self.device)

        with torch.no_grad():
            x_hat = self.T_inv(z).cpu().numpy()
        error = np.abs(x_truth-x_hat)

        return x_truth[0], x_hat, time, error
    
    def sim_multi(self, ic_samples, add_noise = False):
        """
        Simulate multiple trajectories at once.
        """
        avr_error = 0
        errors = []
        x_traj = []
        x_hat_traj = []
        for idx, ic in enumerate(ic_samples):
            x, x_hat, time, error, _, _ = self.simulate(ic, add_noise=add_noise)
            avr_error += error
            errors.append(error)
            x_traj.append(x)
            x_hat_traj.append(x_hat)
        avr_error = avr_error / idx

        return np.array(x_traj), np.array(x_hat_traj), np.array(errors), avr_error, time

    # def get_FDI_threshold(self, sample_space, samples, filter):
    #     ic = np.expand_dims(LHS(xlimits = sample_space)(samples), axis = 1)

    #     _, _, residuals, _, t = self.sim_multi(ic, add_noise = True)
    #     residuals = np.squeeze(residuals, axis=1)
    #     num_steps = residuals.shape[2]  
    #     dt = (b-a)/N
    #     err_deriv = np.abs(np.diff(err, axis = 0))/dt
    #     err_deriv = np.append(err_deriv, np.zeros([1,err.shape[1]]), axis = 0)
    #     max = -np.inf
    #     for residual in residuals:
    #         for j in range(5):
    #             _, filtered_res = ct.forced_response(filter, t[100:], residual[100:,j])
    #             max_current = np.max(filtered_res[100:])
    #             if max_current > max:
    #                 max = max_current
    #     return max
    
    def get_FDI_threshold(self, sample_space, samples):
        ic = np.expand_dims(LHS(xlimits = sample_space)(samples), axis = 1)
        dt = (self.b-self.a)/self.N
        residuals = self.sim_multi(ic, add_noise = True)[2]
        residuals = residuals.reshape(10,1001,10)
        residuals_deriv = np.diff(residuals, axis = 1)/dt
        
        return np.max(residuals_deriv[:,100:,:])

    def calc_gen_metric(self, train_ic, test_ic):
        GE = []
        GE_matrix = []
        p = len(train_ic)
        tau = self.N
        train_ic = np.expand_dims(train_ic, axis = 1)
        x_train, x_hat_train, _, _, time1 = self.sim_multi(train_ic)

        train_error = 0
        for true, est in zip(x_train, x_hat_train):
            sum = 0
            for x, x_hat in zip(true, est):
                error_norm = np.linalg.norm(x-x_hat)**2
                true_norm = np.linalg.norm(x)**2
                sum += error_norm/true_norm
            train_error += sum / tau
    
        train_error_av = train_error / p

        if self.system.x_size == 2:
            for circle in test_ic:
                x_test, x_hat_test, _, _, time2 = self.sim_multi(circle)
                sum_circle = 0
                GE_matrix_col = []
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis = 1)**2
                    true_norm = np.linalg.norm(true, axis = 1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_circle += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_circle / len(circle)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)
        elif self.system.x_size == 3:
            for sphere in test_ic:
                sum_sphere = 0
                GE_matrix_col = []
                ic = sphere.reshape(-1,1,3)
                x_test, x_hat_test, _, _, _ = self.sim_multi(ic)
                for true, est in zip(x_test, x_hat_test):
                    error_norm = np.linalg.norm(true - est, axis = 1)**2
                    true_norm = np.linalg.norm(true, axis = 1)**2
                    trajectory_average = np.sum(error_norm / true_norm) / tau
                    sum_sphere += trajectory_average
                    GE_matrix_col.append(trajectory_average)
                metric = np.abs((sum_sphere / len(ic)) - train_error_av)
                GE.append(metric)
                GE_matrix.append(GE_matrix_col)
                    
        return GE, np.array(GE_matrix)


        
