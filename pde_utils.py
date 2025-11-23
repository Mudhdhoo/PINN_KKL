import torch
from torch.autograd.functional import jacobian

# # PDE constrain loss for PINN from x --> z
def pde_loss(T_net, x, y, z_hat, time, system, M, K, device, reduction = 'mean'):
    M = torch.from_numpy(M).to(device)
    K = torch.from_numpy(K).to(device)
    x.requires_grad_()

    # Jacobian
    dTdx = calc_J(x, T_net)
    
    # Computation of f(x)
    f = x[0].unsqueeze(0)
    u = 0
    for state, t in zip(x[1:], time):
        f_out = system.function(t, u, state).unsqueeze(0)
        f_out = f_out.to(device)
        f = torch.cat((f, f_out), dim = 0)
    #f = torch.from_numpy(np.array(f)).float().to(self.device)
    # dT/dx * f(x)

    f = f.to(torch.float32)
    dTdx_mul_f = torch.bmm(dTdx, torch.unsqueeze(f,2))

    z_hat = torch.unsqueeze(z_hat, 2)
    M = M.to(torch.float32)
    M_mul_T = torch.matmul(M, z_hat)    # MT(x)
    
    # Check if y elements are scalar
    K = K.to(torch.float32)
    y = y.to(torch.float32)
    if y[0].shape == torch.Size([]):
        K_mul_h = torch.matmul(K, y.view(y.shape[0],1,1))    # Kh(x)
    else:
        y = torch.unsqueeze(y, 2)
        K_mul_h = torch.matmul(K, y)    # Kh(x)
        
    pde = dTdx_mul_f - M_mul_T - K_mul_h    # dT/dx*f(x) - MT(x) - Kh(x) = 0
    loss_batch = torch.linalg.norm(pde, dim = 1)    # Element-wise norm

    # Type of loss reduction
    if reduction == 'mean':
        samples = loss_batch.shape[0]
        loss_pde = torch.sum(loss_batch) / samples
    elif reduction == 'sum':
        loss_pde = torch.sum(loss_batch)
    
    return loss_pde

# Jacobian calculation
def calc_J(x, NN):
    m = x.shape[0]       
    dTdx = jacobian(NN, x, create_graph=False)    # dT/dx   
    # result is m* d_o * m * d_i
    ind = torch.arange(0, m)
    
    return dTdx[ind, :, ind, :]