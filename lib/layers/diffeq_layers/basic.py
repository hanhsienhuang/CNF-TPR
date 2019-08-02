import torch
import torch.nn as nn
import torch.nn.functional as F

from . import forward_ad


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError

class IgnoreLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(IgnoreLinear, self).__init__()
        self._layer = forward_ad.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)

    def forward_AD(self, dt_dt, dx_dt):
        dv_dt = self._layer.forward_AD(dx_dt)
        return dv_dt

class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = forward_ad.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        #tt = torch.ones_like(x[:, :1]) * t
        tt = t.expand((x.shape[0],1))
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

    def forward_AD(self, dt_dt, dx_dt):
        #tt = torch.ones_like(dx_dt[:, :1]) * dt_dt
        tt = dt_dt.expand((dx_dt.shape[0], 1))
        ttx = torch.cat([tt, dx_dt], 1)
        return self._layer.forward_AD(ttx)

    def forward_AD2(self, d2t_dt2, d2x_dt2):
        #tt = torch.ones_like(d2x_dt2[:, :1]) * d2t_dt2
        tt = d2t_dt2.expand((d2x_dt2.shape[0], 1))
        ttx = torch.cat([tt, d2x_dt2], 1)
        return self._layer.forward_AD2(ttx)


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear, self).__init__()
        self._layer = forward_ad.Linear(dim_in, dim_out)
        self._hyper_bias = forward_ad.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1))

    def forward_AD(self, dt_dt, dx_dt):
        return self._layer.forward_AD(dx_dt) + self._hyper_bias.forward_AD(dt_dt.view(1, 1))


class SquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1)))

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatSquashLinear, self).__init__()
        self._layer = forward_ad.Linear(dim_in, dim_out)
        self._hyper_bias = forward_ad.Linear(1, dim_out, bias=False)
        self._hyper_gate = forward_ad.Linear(1, dim_out)
        self.sigmoid = forward_ad.Activation(torch.sigmoid)
        self.addmul = forward_ad.Addmul()

    def forward(self, t, x):
        t = t.view(1,1)
        return self.addmul(self._hyper_bias(t), self._layer(x), self.sigmoid(self._hyper_gate(t)))

    def forward_AD(self, dt_dt, dx_dt):
        dt_dt = dt_dt.view(1,1)
        dv_dt = self.addmul.forward_AD(
                self._hyper_bias.forward_AD(dt_dt),
                self._layer.forward_AD(dx_dt),
                self.sigmoid.forward_AD(self._hyper_gate(dt_dt))
                )
        return dv_dt

    def forward_AD2(self, d2t_dt2, d2x_dt2):
        d2t_dt2 = d2t_dt2.view(1,1)
        d2v_dt2 = self.addmul.forward_AD2(
                self._hyper_bias.forward_AD2(d2t_dt2),
                self._layer.forward_AD2(d2x_dt2),
                self.sigmoid.forward_AD2(self._hyper_gate(d2t_dt2))
                )
        return d2v_dt2


class HyperConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(HyperConv2d, self).__init__()
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        return self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class IgnoreConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(IgnoreConv2d, self).__init__()
        module = forward_ad.ConvTranspose2d if transpose else forward_ad.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        return self._layer(x)

    def forward_AD(self, dt_dt, dx_dt):
        return self._layer.forward_AD(dx_dt)


class SquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(SquashConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper = nn.Linear(1, dim_out)

    def forward(self, t, x):
        return self._layer(x) * torch.sigmoid(self._hyper(t.view(1, 1))).view(1, -1, 1, 1)

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = forward_ad.ConvTranspose2d if transpose else forward_ad.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

    def forward_AD(self, dt_dt, dx_dt):
        tt = torch.ones_like(dx_dt[:, :1, :, :]) * dt_dt
        ttx = torch.cat([tt, dx_dt], 1)
        return self._layer.forward_AD(ttx)


class ConcatConv2d_v2(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = forward_ad.ConvTranspose2d if transpose else forward_ad.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)

    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(1, 1)).view(1, -1, 1, 1)

    def forward_AD(self, dt_dt, dx_dt):
        return self._layer.forward_AD(dx_dt) + self._hyper_bias.forward_AD(dt_dt.view(1, 1)).view(1, -1, 1, 1)


class ConcatSquashConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatSquashConv2d, self).__init__()
        module = forward_ad.ConvTranspose2d if transpose else forward_ad.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._hyper_gate = forward_ad.Linear(1, dim_out)
        self._hyper_bias = forward_ad.Linear(1, dim_out, bias=False)
        self.sigmoid = forward_ad.Activation(torch.sigmoid)
        self.addmul = forward_ad.Addmul()

    def forward(self, t, x):
        t = t.view(1, 1)
        return self.addmul(
                self._hyper_bias(t).view(1, -1, 1, 1), 
                self._layer(x), 
                self.sigmoid(self._hyper_gate(t).view(1, -1, 1, 1))
                )

    def forward_AD(self, dt_dt, dx_dt):
        dt_dt = dt_dt.view(1,1)
        dv_dt = self.addmul.forward_AD(
                self._hyper_bias.forward_AD(dt_dt).view(1, -1, 1, 1),
                self._layer.forward_AD(dx_dt),
                self.sigmoid.forward_AD(self._hyper_gate.forward_AD(dt_dt).view(1, -1, 1, 1))
                )
        return dv_dt

class ConcatCoordConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatCoordConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 3, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        b, c, h, w = x.shape
        hh = torch.arange(h).to(x).view(1, 1, h, 1).expand(b, 1, h, w)
        ww = torch.arange(w).to(x).view(1, 1, 1, w).expand(b, 1, h, w)
        tt = t.to(x).view(1, 1, 1, 1).expand(b, 1, h, w)
        x_aug = torch.cat([x, tt, hh, ww], 1)
        return self._layer(x_aug)

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class GatedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
            groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g

    def forward_AD(self, dt_dt, dx_dt):
        raise NotImplementedError


class BlendLinear_v0(nn.Module):
    def __init__(self, dim_in, dim_out, **unused_kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = forward_ad.Linear(dim_in, dim_out)
        self._layer1 = forward_ad.Linear(dim_in, dim_out)
        self.linear_inter = forward_ad.LinearInterpolate()

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return self.linear_inter(t, y0, y1)

    def forward_AD(self, dt_dt, dx_dt):
        return self.linear_inter.forward_AD(dt_dt, 
                self._layer0.forward_AD(dx_dt),
                self._layer1.forward_AD(dx_dt)
                )

class BlendLinear(nn.Module):
    def __init__(self, dim_in, dim_out, **unused_kwargs):
        super().__init__()
        self._layer0 = nn.Linear(dim_in, dim_out)
        self._layer1 = nn.Linear(dim_in, dim_out)
        self._linear = forward_ad.FunctionalLinear()
        self.inter_w = forward_ad.LinearInterpolate()
        self.inter_b = forward_ad.LinearInterpolate()

    def forward(self, t, x):
        w = self.inter_w(t, self._layer0.weight, self._layer1.weight)
        b = self.inter_b(t, self._layer0.bias, self._layer1.bias)
        return self._linear(x, w, b)

    def forward_AD(self, dt_dt, dx_dt):
        return self._linear.forward_AD(dx_dt, 
                self.inter_w.forward_AD(dt_dt, 0, 0), 
                self.inter_b.forward_AD(dt_dt, 0, 0)
                )

class BlendConv2d(nn.Module):
    def __init__(
        self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(BlendConv2d, self).__init__()
        module = forward_ad.ConvTranspose2d if transpose else forward_ad.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self.linear_inter = forward_ad.LinearInterpolate()

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return self.linear_inter(t, y0, y1)

    def forward_AD(self, dt_dt, dx_dt):
        return self.linear_inter.forward_AD(dt_dt, 
                self._layer0.forward_AD(dx_dt),
                self._layer1.forward_AD(dx_dt)
                )

