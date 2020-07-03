import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]

        

def sample_unique(num_steps):
    rnd = torch.cat([torch.tensor([0.0]), torch.rand(num_steps).sort()[0], torch.tensor([1.0])])
    while torch.any(rnd[:-1] == rnd[1:]):
        rnd = torch.cat([torch.tensor([0.0]), torch.rand(num_steps).sort()[0], torch.tensor([1.0])])
    return rnd

def poly_reg_error(t, z, order):
    """
    Error = ((I-G(G^T G)^{-1}G^T)) @ y)^2
    """
    T = [torch.ones_like(t)]
    for i in range(order):
        T.append(T[-1] * (t-t[i]))
    G = torch.stack(T, -1)
    U = torch.svd(G)[0]
    mat = torch.eye(t.shape[0]).to(t) - U[:,:order+1]@U.t()[:order+1]
    return torch.sum(torch.einsum("ij,j...->i...", mat, z)**2)/t.shape[0]/z.shape[1]

class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, solver='dopri5', atol=1e-5, rtol=1e-5, test_atol=None, test_rtol=None, poly_num_sample=0, poly_order=0, adjoint=True):
        super(CNF, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        self.odefunc = odefunc
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = test_atol if test_atol else atol
        self.test_rtol = test_rtol if test_rtol else rtol
        assert poly_num_sample >= poly_order
        self.poly_num_sample = poly_num_sample
        self.poly_order = poly_order
        self.solver_options = {}
        self.odeint = odeint_adjoint if adjoint else odeint


    def forward(self, z, logpz=None, lec=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        return_last = integration_times is None
        lec = None if not self.training else lec

        if integration_times is None:
            end_time = self.sqrt_end_time **2
            if lec is None:
                integration_times = torch.tensor([0.0, end_time]).to(z)
            else:
                integration_times = sample_unique(self.poly_num_sample).to(z) * end_time

        if reverse:
            integration_times = _flip(integration_times, 0)

        self.odefunc.before_odeint()
        in_state = (z, _logpz)

        state_t = self.odeint(
            self.odefunc,
            in_state,
            integration_times,
            atol= self.atol if self.training else self.test_atol,
            rtol= self.rtol if self.training else self.test_rtol,
            method=self.solver if self.training else self.test_solver,
            options=self.solver_options if self.training else None,
        )
        
        z_t = state_t[0]
        if return_last:
            state_t = tuple(s[-1] for s in state_t)

        if lec is not None:
            lec = lec + poly_reg_error(integration_times, z_t, self.poly_order)
        return tuple(state_t) + (lec,)


    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
