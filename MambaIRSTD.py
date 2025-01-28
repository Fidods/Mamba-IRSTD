import math

import torch

from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange, repeat
from models.modules import Down_wt, GSC
from torchvision.transforms.functional import rgb_to_grayscale
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm import Mamba2
# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

class EFFA(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x, edge):
        edge = self.sigmoid(edge * x + edge)
        x = edge + x
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x) + edge
def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel
def downsample(x):
    return x[:, :, ::2, ::2]
def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out
def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels))

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        filtered = conv_gauss(current, gauss_kernel(channels))
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class stem(nn.Module):
    def __init__(self, inch, outch):
        super(stem, self).__init__()
        self.inch = inch
        self.outch = outch
        self.conv_1 = nn.Conv2d(inch, outch, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(outch)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(outch, outch, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(outch)
        self.res = nn.Conv2d(inch, outch, 1, 1, 0)

    def forward(self, x):
        res = self.res(x)
        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)) + res)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class ResAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mamba=True, attn=True):
        super(ResAttnBlock, self).__init__()
        self.radio = 2
        self.mamba = mamba
        self.attn = attn
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(out_channels, out_channels * self.radio, 3, 1, 1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * self.radio)
        if self.mamba:
            self.mamba = Mamba_block(out_channels * self.radio, out_channels * self.radio)
            self.silu = nn.SiLU()
        if self.attn:
            self.att = EFFA()
        self.conv3 = nn.Conv2d(out_channels * self.radio, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, edge):
        B, C, H, W = x.shape
        residual = self.res(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.mamba:
            C = self.out_channels * self.radio
            x = x.reshape(B, C, int(H * W)).transpose(1, 2)
            x = self.mamba(x, H, W).reshape(B, C, H, W)
            x = self.silu(x)
        # print(x.shape)
        if self.attn:
            x = x * self.att(x, edge)
        x = self.bn3(self.conv3(x))
        out = F.relu(x + residual, True)
        return out


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):  # (b,h,w,c)->(b,h,w,2c)->(b,2h,2w,c/2) ->(b,c/2,2h,2w,)
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        # print(self.dim)
        # print(B, H, W, C)
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x).permute(0, 3, 1, 2)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x, H, W, relative_pos=None):
        B, N, C = x.shape
        # print('x input',x.shape)
        x = x.permute(0, 2, 1).reshape(B, H, W, C)

        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.reshape(B, N, C)
        # print('x output',out.shape)
        return out


class Mamba_block(nn.Module):
    def __init__(self, inch, outch):
        super(Mamba_block, self).__init__()
        self.mamba = SS2D(inch)
        self.conv = nn.Conv1d(outch, outch, 3, 1, 1)
        self.LN = nn.LayerNorm(outch)
        self.Linear = nn.Linear(inch, outch)
        self.res = nn.Linear(inch, outch)
        self.dropout = nn.Dropout(0.1)
        self.sig = nn.Sigmoid()

    def forward(self, x, H, W):
        B, tokens, C = x.shape
        res = self.dropout(self.res(x))
        res = self.sig(res)
        out = self.Linear(x)
        out = self.sig(self.conv(out.transpose(1, 2))).transpose(1, 2)
        out = self.LN(self.mamba(out, H, W) * res)
        return out


class DRMB(nn.Module):
    def __init__(self, inch, outch):
        super(DRMB, self).__init__()
        self.unfold_8 = nn.Unfold(kernel_size=8, padding=0, stride=8)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inch, inch * 2, 3, stride=1, padding=1, groups=inch),  # DSC
            nn.BatchNorm2d(inch * 2),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(int(inch), inch * 4, 1, stride=1, padding=0),
            nn.BatchNorm2d(inch * 4),
        )

        self.conv2 = Down_wt(inch * 2, inch * 4)
        self.conv3 = nn.Conv2d(inch * 4, inch * 4, 3, stride=1, padding=1, groups=8)
        self.bn2 = nn.BatchNorm2d(inch * 4)
        self.relu = nn.ReLU(inplace=True)

        self.mamba1 = Mamba_block(inch * 2, inch * 2)
        self.Linear1 = nn.Linear(64, 128)
        self.LN1 = nn.LayerNorm(128)
        self.Linear2 = nn.Linear(128, 64)
        self.LN2 = nn.LayerNorm(64)
        self.sig = nn.Sigmoid()

        self.skip_scale_2 = nn.Parameter(torch.ones(1))
        self.skip_scale_4 = nn.Parameter(torch.ones(1))

        self.BN = nn.BatchNorm2d(inch * 8)
        self.conv4 = nn.Conv2d(inch * 8, outch, 1, 1, 0)
        self.RELU = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.res(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.relu(self.bn2(x + res))

        B, C, H, W = x.shape
        C = int(C / 2)
        x1, x2 = torch.chunk(x, 2, dim=1)

        x_2 = x2.reshape(B, C, int(H * W)).transpose(1, 2)  # 8 * C/(H*W//16)
        x_mamba1 = self.mamba1(x_2, H, W) + x_2 * self.skip_scale_2
        x_mamba1 = x_mamba1.reshape(B, C, H, W)

        x_4 = self.unfold_8(x2).transpose(1, 2)  # c=32c
        x_4 = x_4.reshape(B, int(H * W * C / 64), 64)  # 8 * C/(H*W//16)
        # x_mamba2 = self.mamba2(x_4,H,W) + x_4 * self.skip_scale_4
        x_4 = self.sig(self.LN2(self.Linear2(self.LN1(self.Linear1(x_4)))) + x_4 * self.skip_scale_4)
        x_mamba2 = x_4.reshape(B, C, H, W)

        x_fusion = torch.cat([x_mamba1, x_mamba2, x], 1)
        x_mamba = self.BN(x_fusion)
        x_mamba = self.conv4(x_mamba)

        # x_mamba=x_fusion.reshape(B,)
        return x_mamba


class MSAFH(nn.Module):
    def __init__(self):
        super(MSAFH, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 8, 3, 1, 1)
        self.conv5 = nn.Conv2d(16, 1, 3, 1, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(8)

        self.relu = nn.ReLU()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, edge):
        if self.training:
            out = self.relu(self.bn1(self.conv1(self.up(x[0] + edge[4]))) + edge[3])

            out = self.relu(self.bn2(self.conv2(self.up((x[1] + out) + edge[3]))) + edge[2])
            out = self.relu(self.bn3(self.conv3(self.up((x[2] + out) + edge[2]))) + edge[1])  # 71.24 39
            out = self.relu(self.bn4(self.conv4(self.up((x[3] + out) + edge[1]))))  # +edge[0])
        else:
            out = self.relu(self.bn1(self.conv1(self.up(x[0] + edge[4]))))  # +edge[3])
            out = self.relu(self.bn2(self.conv2(self.up((x[1] + out) + edge[3]))))  # +edge[2])
            out = self.relu(self.bn3(self.conv3(self.up((x[2] + out) + edge[2]))))  # +edge[1])# 71.24 39
            out = self.relu(self.bn4(self.conv4(self.up((x[3] + out) + edge[1]))))  # +edge[0])

        out = self.conv5(torch.cat([out, x[4]], 1))
        return out


class MFAAB(nn.Module):
    def __init__(self, inch, outch):
        super(MFAAB, self).__init__()
        self.skip_scale_1 = nn.Parameter(torch.ones(1))
        self.skip_scale_2 = nn.Parameter(torch.ones(1))
        self.skip_scale_3 = nn.Parameter(torch.ones(1))
        self.conv1x1 = nn.Sequential(nn.Conv2d(inch, outch, 1, 1, 0), nn.BatchNorm2d(outch))
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(inch, outch, 3, 1, 1, groups=inch), nn.BatchNorm2d(outch),
                                       nn.ReLU(True))
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(outch, outch, 3, 1, 1, groups=outch), nn.BatchNorm2d(outch))
        self.conv5x5_1 = nn.Sequential(nn.Conv2d(inch, outch, 5, 1, 2, groups=inch), nn.BatchNorm2d(outch),
                                       nn.ReLU(True))
        self.conv5x5_2 = nn.Sequential(nn.Conv2d(outch, outch, 5, 1, 2, groups=outch), nn.BatchNorm2d(outch))
        self.fusion = nn.Sequential(nn.Conv2d(outch, outch, 3, 1, 1), nn.BatchNorm2d(outch), nn.ReLU(True))
        self.final = nn.Sequential(nn.Conv2d(outch, outch, 1, 1, 0), nn.BatchNorm2d(outch), nn.ReLU(True))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1x1(x)
        conv1 = conv1 * (conv1 * self.skip_scale_1)
        conv3 = self.conv3x3_2(self.conv3x3_1(x))
        conv3 = conv3 * (conv3 * self.skip_scale_2)
        conv5 = self.conv5x5_2(self.conv5x5_1(x))
        conv5 = conv5 * (conv5 * self.skip_scale_3)
        fusion = self.fusion(conv1 + conv3 + conv5)
        out = self.final(fusion * self.sig(x) + x)
        return out


class MambaIRSTD(nn.Module):
    def __init__(self, inch):
        super(MambaIRSTD, self).__init__()
        self.size = [256, 128, 64, 32, 16]
        channels = [8, 16, 32, 64, 128, 256]
        # encode
        self.stem = stem(inch, channels[0])

        self.TMambaM1 = DRMB(channels[0], channels[1])
        self.TMambaM2 = DRMB(channels[1], channels[2])
        self.TMambaM3 = DRMB(channels[2], channels[3])
        self.TMambaM4 = DRMB(channels[3], channels[4])

        self.skip_scale1 = nn.Parameter(torch.ones(1))
        self.skip_scale2 = nn.Parameter(torch.ones(1))
        self.skip_scale3 = nn.Parameter(torch.ones(1))
        self.skip_scale4 = nn.Parameter(torch.ones(1))

        # decode
        self.up4 = PatchExpand2D(channels[4])
        self.up3 = PatchExpand2D(channels[3])
        self.up2 = PatchExpand2D(channels[2])
        self.up1 = PatchExpand2D(channels[1])

        self.uplayer1 = ResAttnBlock(channels[1], channels[0], False, attn=True)
        self.uplayer2 = ResAttnBlock(channels[2], channels[1], False, attn=True)
        self.uplayer3 = ResAttnBlock(channels[3], channels[2], False, attn=True)
        self.uplayer4 = ResAttnBlock(channels[4], channels[3], False, attn=True)
        self.layer5 = ResAttnBlock(channels[4], channels[4], mamba=False, attn=True)
        self.layer5_2 = MFAAB(channels[4], channels[4])

        self.Head = MSAFH()

    def forward(self, x):
        gray_img = rgb_to_grayscale(x)
        edge = make_laplace_pyramid(gray_img, 5, 1)

        x = self.stem(x)

        skip_x0 = x * self.skip_scale4
        x1 = self.TMambaM1(x)

        skip_x1 = x1 * self.skip_scale1
        x2 = self.TMambaM2(x1)

        skip_x2 = x2 * self.skip_scale2
        x3 = self.TMambaM3(x2)

        skip_x3 = x3 * self.skip_scale3
        x4 = self.TMambaM4(x3)

        skip_x4 = x4 * self.skip_scale4
        out5 = self.layer5_2(self.layer5(x4, edge[4]), edge[4])  # b,128,16

        out4 = self.uplayer4((torch.cat([skip_x3, self.up4(out5)], 1)), edge[3])  # b,64,32
        out3 = self.uplayer3((torch.cat([skip_x2, self.up3(out4)], 1)), edge[2])  # b,32,64
        out2 = self.uplayer2((torch.cat([skip_x1, self.up2(out3)], 1)), edge[1])  # b,16,128
        out1 = self.uplayer1((torch.cat([skip_x0, self.up1(out2)], 1)), edge[0])  # b,8,256

        input = [out5, out4, out3, out2, out1]
        final_out = self.Head(input, edge)

        return final_out


if __name__ == "__main__":
    import time

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.__version__)
    x = torch.rand(4, 3, 256, 256)
    start_time = time.time()
    net = MyMambaIRSTD(3)

    net = net.cuda()
    x = x.cuda()
    output = net(x)
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    print(output.shape)
