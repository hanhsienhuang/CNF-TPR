import torch
import torch.nn as nn
from train_misc import build_model_tabular
import lib.layers as layers
from .VAE import VAE
import lib.layers.diffeq_layers as diffeq_layers
from lib.layers.diffeq_layers import forward_ad
from lib.layers.odefunc import NONLINEARITIES
from lib.layers.cnf import sample_unique
from train_misc import create_regularization_fns

from lib.layers.torchdiffeq import odeint_adjoint as odeint


def get_hidden_dims(args):
    return tuple(map(int, args.dims.split("-"))) + (args.z_size,)


def concat_layer_num_params(in_dim, out_dim):
    return (in_dim + 1) * out_dim + out_dim


class CNFVAE(VAE):

    def __init__(self, args):
        super(CNFVAE, self).__init__(args)

        # CNF model
        self.cnf = build_model_tabular(args, args.z_size)

        if args.cuda:
            self.cuda()

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)

        return mean_z, var_z

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        z_mu, z_var = self.encode(x)

        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)

        zero = torch.zeros(x.shape[0], 1).to(x)
        zk, delta_logp = self.cnf(z0, zero)  # run model forward

        x_mean = self.decode(zk)

        return x_mean, z_mu, z_var, -delta_logp.view(-1), z0, zk


class AmortizedBiasODEnet(nn.Module):

    def __init__(self, hidden_dims, input_dim, layer_type="concat", nonlinearity="softplus"):
        super(AmortizedBiasODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        self.input_dim = input_dim

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_dim
        for dim_out in hidden_dims:
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def _unpack_params(self, params):
        return [params]

    def forward(self, t, y, am_biases):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            this_bias, am_biases = am_biases[:, :dx.size(1)], am_biases[:, dx.size(1):]
            dx = dx + this_bias
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class AmortizedLowRankODEnet(nn.Module):

    def __init__(self, hidden_dims, input_dim, rank=1, layer_type="concat", nonlinearity="softplus"):
        super(AmortizedLowRankODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        self.input_dim = input_dim

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_dim
        self.output_dims = hidden_dims
        self.input_dims = (input_dim,) + hidden_dims[:-1]
        for dim_out in hidden_dims:
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(forward_ad.Activation(NONLINEARITIES[nonlinearity]))
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.rank = rank

    def _unpack_params(self, params):
        return [params]

    def _rank_k_bmm(self, x, u, v):
        xu = torch.bmm(x[:, None], u)
        xuv = torch.bmm(xu, v)
        return xuv[:, 0]

    def _prepare_params(self, am_params):
        self.am_params = []
        for in_dim, out_dim in zip(self.input_dims, self.output_dims):
            this_u, am_params = am_params[:, :in_dim * self.rank], am_params[:, in_dim * self.rank:]
            this_v, am_params = am_params[:, :out_dim * self.rank], am_params[:, out_dim * self.rank:]
            this_bias, am_params = am_params[:, :out_dim], am_params[:, out_dim:]
            this_u = this_u.view(-1, in_dim, self.rank)
            this_v = this_v.view(-1, self.rank, out_dim)
            self.am_params.append( (this_u, this_v, this_bias))

    def forward(self, t, y, am_params):
        dx = y
        self._prepare_params(am_params)
        for l, (layer, (u,v,bias)) in enumerate(zip(self.layers, self.am_params)):
            xw = layer(t, dx)
            xw_am = self._rank_k_bmm(dx, u, v)
            dx = xw + xw_am + bias
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        return dx

    def forward_AD(self, dt_dt, dy_dt):
        dv_dt = dy_dt
        for l, (layer, (u,v,bias)) in enumerate(zip(self.layers, self.am_params)):
            dvw = layer.forward_AD(dt_dt, dv_dt)
            dvw_am = self._rank_k_bmm(dv_dt, u, v)
            dv_dt = dvw + dvw_am 
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dv_dt = self.activation_fns[l].forward_AD(dv_dt)
        self.am_params = None
        return dv_dt


class HyperODEnet(nn.Module):

    def __init__(self, hidden_dims, input_dim, layer_type="concat", nonlinearity="softplus"):
        super(HyperODEnet, self).__init__()
        assert layer_type == "concat"
        self.input_dim = input_dim

        # build layers and add them
        activation_fns = []
        for dim_out in hidden_dims + (input_dim,):
            activation_fns.append(NONLINEARITIES[nonlinearity])
        self.activation_fns = nn.ModuleList(activation_fns[:-1])
        self.output_dims = hidden_dims
        self.input_dims = (input_dim,) + hidden_dims[:-1]

    def _pack_inputs(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return ttx

    def _unpack_params(self, params):
        layer_params = []
        for in_dim, out_dim in zip(self.input_dims, self.output_dims):
            this_num_params = concat_layer_num_params(in_dim, out_dim)
            # get params for this layer
            this_params, params = params[:, :this_num_params], params[:, this_num_params:]
            # split into weight and bias
            bias, weight_params = this_params[:, :out_dim], this_params[:, out_dim:]
            weight = weight_params.view(weight_params.size(0), in_dim + 1, out_dim)
            layer_params.append(weight)
            layer_params.append(bias)
        return layer_params

    def _layer(self, t, x, weight, bias):
        # weights is (batch, in_dim + 1, out_dim)
        ttx = self._pack_inputs(t, x)  # (batch, in_dim + 1)
        ttx = ttx.view(ttx.size(0), 1, ttx.size(1))  # (batch, 1, in_dim + 1)
        xw = torch.bmm(ttx, weight)[:, 0, :]  # (batch, out_dim)
        return xw + bias

    def forward(self, t, y, *layer_params):
        dx = y
        for l, (weight, bias) in enumerate(zip(layer_params[::2], layer_params[1::2])):
            dx = self._layer(t, dx, weight, bias)
            # if not last layer, use nonlinearity
            if l < len(layer_params) - 1:
                dx = self.activation_fns[l](dx)
        return dx


class LyperODEnet(nn.Module):

    def __init__(self, hidden_dims, input_dim, layer_type="concat", nonlinearity="softplus"):
        super(LyperODEnet, self).__init__()
        base_layer = {
            "ignore": diffeq_layers.IgnoreLinear,
            "hyper": diffeq_layers.HyperLinear,
            "squash": diffeq_layers.SquashLinear,
            "concat": diffeq_layers.ConcatLinear,
            "concat_v2": diffeq_layers.ConcatLinear_v2,
            "concatsquash": diffeq_layers.ConcatSquashLinear,
            "blend": diffeq_layers.BlendLinear,
            "concatcoord": diffeq_layers.ConcatLinear,
        }[layer_type]
        self.input_dim = input_dim

        # build layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_dim
        self.dims = (input_dim,) + hidden_dims
        self.output_dims = hidden_dims
        self.input_dims = (input_dim,) + hidden_dims[:-1]
        for dim_out in hidden_dims[:-1]:
            layer = base_layer(hidden_shape, dim_out)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])
            hidden_shape = dim_out

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)

    def _pack_inputs(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return ttx

    def _unpack_params(self, params):
        return [params]

    def _am_layer(self, t, x, weight, bias):
        # weights is (batch, in_dim + 1, out_dim)
        ttx = self._pack_inputs(t, x)  # (batch, in_dim + 1)
        ttx = ttx.view(ttx.size(0), 1, ttx.size(1))  # (batch, 1, in_dim + 1)
        xw = torch.bmm(ttx, weight)[:, 0, :]  # (batch, out_dim)
        return xw + bias

    def forward(self, t, x, am_params):
        dx = x
        for layer, act in zip(self.layers, self.activation_fns):
            dx = act(layer(t, dx))
        bias, weight_params = am_params[:, :self.dims[-1]], am_params[:, self.dims[-1]:]
        weight = weight_params.view(weight_params.size(0), self.dims[-2] + 1, self.dims[-1])
        dx = self._am_layer(t, dx, weight, bias)
        return dx


def construct_amortized_odefunc(args, z_dim, amortization_type="bias"):

    hidden_dims = get_hidden_dims(args)

    if amortization_type == "bias":
        diffeq = AmortizedBiasODEnet(
            hidden_dims=hidden_dims,
            input_dim=z_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
    elif amortization_type == "hyper":
        diffeq = HyperODEnet(
            hidden_dims=hidden_dims,
            input_dim=z_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
    elif amortization_type == "lyper":
        diffeq = LyperODEnet(
            hidden_dims=hidden_dims,
            input_dim=z_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
        )
    elif amortization_type == "low_rank":
        diffeq = AmortizedLowRankODEnet(
            hidden_dims=hidden_dims,
            input_dim=z_dim,
            layer_type=args.layer_type,
            nonlinearity=args.nonlinearity,
            rank=args.rank,
        )
    odefunc = layers.ODEfunc(
        diffeq=diffeq,
        divergence_fn=args.divergence_fn,
        residual=args.residual,
        rademacher=args.rademacher,
    )
    return odefunc


class AmortizedCNFVAE(VAE):
    h_size = 256

    def __init__(self, args):
        super(AmortizedCNFVAE, self).__init__(args)

        # CNF model
        self.odefuncs = nn.ModuleList([
            construct_amortized_odefunc(args, args.z_size, self.amortization_type) for _ in range(args.num_blocks)
        ])
        self.q_am = self._amortized_layers(args)
        assert len(self.q_am) == args.num_blocks or len(self.q_am) == 0

        if args.cuda:
            self.cuda()
        self.time_length = args.time_length

        self.atol = args.atol
        self.rtol = args.rtol
        self.solver = args.solver
        self.num_sample = args.num_sample
        self.coef_acc = args.coef_acc


    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        am_params = [q_am(h) for q_am in self.q_am]

        return mean_z, var_z, am_params

    def forward(self, x):

        self.log_det_j = 0.

        z_mu, z_var, am_params = self.encode(x)

        # Sample z_0
        z0 = self.reparameterize(z_mu, z_var)

        if self.training and self.num_sample is not None:
            z, delta_logp, lacc = self.sample(z0, am_params)
        else:
            z, delta_logp, lacc = self.solve(z0, am_params)

        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, -delta_logp.view(-1), z0, z, lacc

    def sample(self, z0, am_params):
        need_acc = self.coef_acc is not None
        z = z0
        laccs = []
        dlogps = []
        for odefunc, am_param in zip(self.odefuncs, am_params):
            integration_times = sample_unique(self.num_sample).to(z0) * self.time_length
            am_param_unpacked = odefunc.diffeq._unpack_params(am_param)
            odefunc.before_odeint(output_div=False, output_acc=False)
            states = odeint(
                odefunc,
                (z,) + tuple(am_param_unpacked),
                integration_times,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
            )
            z_t = states[0]
            z = z_t[-1]

            time_length = integration_times[-1] - integration_times[0]
            samples = None
            for t, z in zip(integration_times[1:-1], z_t[1:-1]):
                output = odefunc._forward(t, z, output_div = True, output_acc = need_acc, additional = am_param_unpacked)[1:]
                if not samples:
                    samples = [[] for i in range(len(output))]
                for i in range(len(output)):
                    samples[i].append(output[i])
            for i in range(len(samples)):
                samples[i] = torch.stack(samples[i]).mean(0)*time_length
            dlogps.append(samples[0])
            if need_acc:
                laccs.append(samples[1])
        return z, sum(dlogps), sum(laccs) if need_acc else None

    def solve(self, z0, am_params):
        need_acc = self.training and self.coef_acc is not None
        integration_times = torch.tensor([0.0, self.time_length]).to(z0)
        states = (z0, torch.zeros(z0.shape[0], 1).to(z0)) 
        num_vars = 2
        if need_acc:
            states += (torch.tensor(0.0).to(z0), )
            num_vars += 1

        for odefunc, am_param in zip(self.odefuncs, am_params):
            am_param_unpacked = odefunc.diffeq._unpack_params(am_param)
            odefunc.before_odeint(output_div=True, output_acc=need_acc)
            states = odeint(
                odefunc,
                states + tuple(am_param_unpacked),
                integration_times,
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
            )
            states = tuple(state[-1] for state in states[:num_vars])
        return states[:2] + (states[2] if need_acc else None,)


class AmortizedBiasCNFVAE(AmortizedCNFVAE):
    amortization_type = "bias"

    def _amortized_layers(self, args):
        hidden_dims = get_hidden_dims(args)
        bias_size = sum(hidden_dims)
        return nn.ModuleList([nn.Linear(self.h_size, bias_size) for _ in range(args.num_blocks)])


class AmortizedLowRankCNFVAE(AmortizedCNFVAE):
    amortization_type = "low_rank"

    def _amortized_layers(self, args):
        out_dims = get_hidden_dims(args)
        in_dims = (out_dims[-1],) + out_dims[:-1]
        params_size = (sum(in_dims) + sum(out_dims)) * args.rank + sum(out_dims)
        return nn.ModuleList([nn.Linear(self.h_size, params_size) for _ in range(args.num_blocks)])


class HypernetCNFVAE(AmortizedCNFVAE):
    amortization_type = "hyper"

    def _amortized_layers(self, args):
        hidden_dims = get_hidden_dims(args)
        input_dims = (args.z_size,) + hidden_dims[:-1]
        assert args.layer_type == "concat", "hypernets only support concat layers at the moment"
        weight_dims = [concat_layer_num_params(in_dim, out_dim) for in_dim, out_dim in zip(input_dims, hidden_dims)]
        weight_size = sum(weight_dims)
        return nn.ModuleList([nn.Linear(self.h_size, weight_size) for _ in range(args.num_blocks)])


class LypernetCNFVAE(AmortizedCNFVAE):
    amortization_type = "lyper"

    def _amortized_layers(self, args):
        dims = (args.z_size,) + get_hidden_dims(args)
        weight_size = concat_layer_num_params(dims[-2], dims[-1])
        return nn.ModuleList([nn.Linear(self.h_size, weight_size) for _ in range(args.num_blocks)])
