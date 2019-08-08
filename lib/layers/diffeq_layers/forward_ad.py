import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return self.linear(x)

    def forward_AD(self, dx_dt):
        return F.linear(dx_dt, self.linear.weight)

class FunctionalLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, w, b=None):
        self.stored = (x, w)
        return F.linear(x, w, b)

    def forward_AD(self, dx_dt, dw_dt, db_dt=None):
        self.stored, (x, w) = None, self.stored
        return F.linear(dx_dt, w) + F.linear(x, dw_dt, db_dt)

class Addmul(nn.Module):
    """tensor1 + tensor2 * tensor3"""
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, x3):
        self.stored = (x2, x3)
        return x1 + x2 * x3

    def forward_AD(self, dx1_dt, dx2_dt, dx3_dt):
        (x2, x3), self.stored = self.stored, None
        dv_dt = dx1_dt + x2 * dx3_dt + dx2_dt * x3
        return dv_dt

class LinearInterpolate(nn.Module):
    """ x0*(1-t) + x1*t"""
    def __init__(self):
        super().__init__()

    def forward(self, t, x0, x1):
        inv_t = 1-t
        self.stored = (t, inv_t, x0, x1)
        return x0 * inv_t + x1 * t

    def forward_AD(self, dt_dt, dx0_dt, dx1_dt):
        (t, inv_t, x0, x1), self.stored = self.stored, None
        dv_dt = dx0_dt*inv_t + dx1_dt*t + (x1-x0)*dt_dt
        return dv_dt

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        variables = locals()
        self.settings = {v:variables[v] for v in ['stride', 'padding', 'dilation', 'groups']}
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias = bias, **self.settings)

    def forward(self, x):
        return self.conv2d(x)

    def forward_AD(self, dx_dt):
        return F.conv2d(dx_dt, self.conv2d.weight, **self.settings)

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding = 0, groups=1, bias=True, dilation=1):
        super().__init__()
        variables = locals()
        self.settings = {v:variables[v] for v in ['stride', 'padding', 'output_padding', 'dilation', 'groups']}
        self.layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, bias = bias, **self.settings)

    def forward(self, x):
        return self.layer(x)

    def forward_AD(self, dx_dt):
        return F.conv_transpose2d(dx_dt, self.layer.weight, **self.settings)


class Activation(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        self.stored = (x, self.func(x))
        return self.stored[1]

    def forward_AD(self, dx_dt):
        (x, out), self.stored = self.stored, None
        act_grad = torch.autograd.grad(out, x, torch.ones_like(out), create_graph=True)[0]
        return dx_dt * act_grad

