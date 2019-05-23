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


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, solver='dopri5', atol=1e-5, rtol=1e-5, num_steps=None, adjoint=False):
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
        self.test_atol = atol
        self.test_rtol = rtol
        self.num_steps = num_steps
        self.solver_options = {}
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        return_last = integration_times is None

        if integration_times is None:
            end_time = self.sqrt_end_time **2
            if self.training and self.num_steps is not None:
                integration_times = sample_unique(self.num_steps).to(z) * end_time
            else:
                integration_times = torch.tensor([0.0, end_time]).to(z)

        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        in_state = (z,) if ( self.training or logpz is None ) else (z, logpz)

        if self.training:
            state_t = self.odeint(
                self.odefunc,
                in_state,
                integration_times.to(z),
                atol= self.atol,
                rtol= self.rtol,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = self.odeint(
                self.odefunc,
                in_state,
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
            )

        if self.training:
            z_t = state_t[0]
            if logpz is None:
                return z_t
            divs, accs = [], []

            #divs, accs = FuncWrapper.apply(self.odefunc.get_div_and_acc, integration_times[1:-1], z_t[1:-1], *self.odefunc.parameters())
            #time_length = integration_times[-1] - integration_times[0]
            #divergence = divs.mean(0)*time_length
            #acc_loss = accs.mean(0)*time_length
            for t, z in zip(integration_times[1:-1], z_t[1:-1]):
                div, acc = self.odefunc.get_div_and_acc(t, z)
                divs.append(div)
                accs.append(acc)
            time_length = integration_times[-1] - integration_times[0]
            divergence = torch.stack(divs).mean(0)*time_length
            acc_loss = torch.stack(accs).mean(0)*time_length

            self.regularization_states = (acc_loss,)
            return z_t[-1], (logpz if logpz is not None else 0) - divergence
        else:
            if return_last:
                state_t = tuple(s[-1] for s in state_t)
            if logpz is not None:
                return state_t[:2]
            else:
                return state_t[0]


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
