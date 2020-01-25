import torch
import torch.nn as nn

from .torchdiffeq import odeint_adjoint
from .torchdiffeq import odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]

        

def sample_unique(num_steps):
    rnd = torch.cat([torch.tensor([0.0]), torch.rand(num_steps).sort()[0], torch.tensor([1.0])])
    while torch.any(rnd[:-1] == rnd[1:]):
        rnd = torch.cat([torch.tensor([0.0]), torch.rand(num_steps).sort()[0], torch.tensor([1.0])])
    return rnd


class CNF(nn.Module):
    def __init__(self, odefunc, T=1.0, train_T=False, solver='dopri5', atol=1e-5, rtol=1e-5, num_sample=None, adjoint=True):
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
        self.num_sample = num_sample
        self.solver_options = {}
        self.odeint = odeint_adjoint if adjoint else odeint

    def _shrink(self, arr):
        out = []
        for a in arr:
            if a is not None:
                out.append(a)
        return tuple(out)

    def _expand(self, arr, values):
        out = list(arr)
        i = 0
        for v in values:
            while out[i] is None:
                i += 1
            out[i] = v
            i += 1
        return tuple(out)

    def _add(self, arr, values):
        out = list(arr)
        i = 0
        for v in values:
            while out[i] is None:
                i += 1
            out[i] = out[i] + v
            i += 1
        return tuple(out)


    def forward_solve(self, z, logpz=None, lacc=None, integration_times=None, reverse=False):
        return_last = integration_times is None
        if integration_times is None:
            end_time = self.sqrt_end_time **2
            integration_times = torch.tensor([0.0, end_time]).to(z)

        if reverse:
            integration_times = _flip(integration_times, 0)

        self.odefunc.before_odeint(output_div = logpz is not None, output_acc = lacc is not None)
        in_state = self._shrink([z, logpz, lacc])

        state_t = self.odeint(
            self.odefunc,
            in_state,
            integration_times,
            atol= self.atol if self.training else self.test_atol,
            rtol= self.rtol if self.training else self.test_rtol,
            method=self.solver if self.training else self.test_solver,
            options=self.solver_options if self.training else None,
        )
        
        if return_last:
            state_t = tuple(s[-1] for s in state_t)
        return self._expand([z,logpz,lacc], state_t)

    def forward_sample(self, z, logpz=None, lacc=None, integration_times=None, reverse=False):
        end_time = self.sqrt_end_time **2
        integration_times = sample_unique(self.num_sample).to(z) * end_time

        if reverse:
            integration_times = _flip(integration_times, 0)

        self.odefunc.before_odeint(output_div = False, output_acc = False)
        in_state = (z,)

        state_t = self.odeint(
            self.odefunc,
            in_state,
            integration_times.to(z),
            atol= self.atol,
            rtol= self.rtol,
            method=self.solver,
            options=self.solver_options,
        )

        z_t = state_t[0]

        samples = None
        for t, z in zip(integration_times[1:-1], z_t[1:-1]):
            output = self.odefunc._forward(t, z, output_div = logpz is not None, output_acc = lacc is not None)[1:]
            if not samples:
                samples = [[] for i in range(len(output))]
            for i in range(len(output)):
                samples[i].append(output[i])
        for i in range(len(samples)):
            samples[i] = torch.stack(samples[i]).mean(0)*end_time
        ret = (z_t[-1],) + self._add([logpz, lacc], samples)
        return ret


    def forward(self, z, logpz=None, lacc=None, integration_times=None, reverse=False):
        if self.training and self.num_sample is not None:
            return self.forward_sample(z, logpz, lacc, integration_times, reverse)
        else:
            return self.forward_solve(z, logpz, lacc, integration_times, reverse)


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
