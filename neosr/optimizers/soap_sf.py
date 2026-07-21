import gc
import string
from collections.abc import Callable

import torch
from torch import Tensor


def warmup(lr: float, step: int, warmup_steps: int):
    if step >= warmup_steps:  # if instead of min to guard against 0 div
        return lr
    return lr * step / warmup_steps


def is_compiling():
    try:
        return torch.compiler.is_compiling()
    except torch._dynamo.exc.TorchDynamoException:
        return True


def set_(dst: Tensor, src: Tensor):
    if not is_compiling() and src.data_ptr() == dst.data_ptr():
        return
    if src.shape != dst.shape:
        src = src.reshape_as(dst)
    if (
        not is_compiling()
        and src.is_contiguous()
        and dst.is_contiguous()
        and src.dtype == dst.dtype
    ):
        dst.set_(src)
    else:
        dst.copy_(src)


def promote(x):
    if isinstance(x, torch.dtype) and x in (torch.bfloat16, torch.float16):
        return torch.float32
    if isinstance(x, Tensor) and x.dtype in (torch.bfloat16, torch.float16):
        return x.float()
    return x


# @decorator_knowngood
def _compilable_copy_stochastic_(target: Tensor, source: Tensor):
    """Taken as-is from https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905"""
    # create a random 16 bit integer
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))


def copy_stochastic_(target: Tensor, source: Tensor):
    if not is_compiling() and target.data_ptr() == source.data_ptr():
        return
    if target.dtype == torch.bfloat16 and source.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
    ):
        _compilable_copy_stochastic_(target, source.float())
    set_(target, source)


def copy_stochastic_list_(target: list[Tensor], source: list[Tensor]):
    for t, s in zip(target, source, strict=False):
        copy_stochastic_(t, s)


# @decorator_knowngood
def _compilable_schedule_free_(
    p: list[Tensor],
    z: list[Tensor],
    ckp1: Tensor,
    grad: list[Tensor],
    lr: Tensor,
    beta1: Tensor,
):
    p32, z32, g32 = [list(map(promote, x)) for x in (p, z, grad)]
    for p_, z_, g_ in zip(p32, z32, g32, strict=False):
        p_.lerp_(z_, ckp1)
        p_.add_(g_, alpha=lr * (beta1 * (1 - ckp1) - 1))
        z_.add_(g_, alpha=-lr)
    copy_stochastic_list_(p, p32)
    copy_stochastic_list_(z, z32)


def scalar_guard(x, ref):
    if isinstance(x, float):
        return torch.empty((), dtype=torch.float32, device=ref.device).fill_(x)
    if isinstance(x, int):
        return torch.empty((), dtype=torch.int64, device=ref.device).fill_(x)
    return x


def schedule_free_(
    lr: float,
    weight_lr_power: float,
    weight_sum: float,
    beta1: float,
    parameters: list[Tensor],
    z: list[Tensor],
    grad: list[Tensor],
    r: float = 0.0,
    step: int = 0,
):
    weight = lr**weight_lr_power * max(step, 1) ** r
    weight_sum = weight_sum + weight

    try:
        ckp1 = weight / weight_sum
    except ZeroDivisionError:
        ckp1 = 0

    # These operations update y in-place,
    # without computing x explicitly.
    lr, ckp1 = scalar_guard(lr, parameters[0]), scalar_guard(ckp1, parameters[0])
    _compilable_schedule_free_(parameters, z, ckp1, grad, lr, beta1)
    return weight_sum


def append_or_extend(base, new):
    if isinstance(new, list):
        base.extend(new)
    else:
        base.append(new)


def dim_merger(grad, max_precond_dim, split: bool = False):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.

    we don't want to merge fan-in into fan-out,
    but we want to merge conv kernels into fan-in or at least merge the kernel
    so, [128, 64, 3, 3] should result in [128, 576] or [128, 64, 9] instead of [73728] or [8192, 3, 3] the baseline
    would've done
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1

    for sh in shape[1:][::-1]:
        temp_shape = curr_shape * sh
        if temp_shape > max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape
    new_shape = [*shape[:1], *new_shape[::-1]]

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)  # needs to be .reshape() due to channels_last
    if not split:
        return new_grad

    grads = [new_grad]
    for i, sh in reversed(list(enumerate(new_shape[:]))):
        if sh == 1:
            grads = [g.squeeze(dim=i) for g in grads]
            continue
        if sh <= max_precond_dim:
            continue
        grads = [a for g in grads for a in g.split(max_precond_dim, dim=i)]
    if len(grads) == 1:
        return new_grad
    new_grads = []
    for g in grads:
        append_or_extend(new_grads, dim_merger(g, max_precond_dim, split))
    return new_grads


def beta_debias(beta, step):
    return 1 - (1 - beta) / (1 - beta**step)


# @decorator_knowngood
def _compilable_exp_avg_sq_(
    state: list[Tensor],
    grad: list[Tensor],
    beta2: Tensor,
    eps: Tensor,
    out: list[Tensor | None],
):
    torch._foreach_mul_(state, beta2)
    [s.addcmul_(g, g, value=1 - beta2) for s, g in zip(state, grad, strict=False)]
    denom = torch._foreach_sqrt(state)
    [denom.clamp_(min=eps) for denom in denom]
    if out[0] is None:
        return denom

    copy_stochastic_list_(out, denom)
    return out


def list_guard(x):
    if isinstance(x, list | tuple):
        return x
    return [x]


def exp_avg_sq_(state, grad, beta2, eps, out=None):
    state, grad, out = list_guard(state), list_guard(grad), list_guard(out)
    beta2, eps = scalar_guard(beta2, state[0]), scalar_guard(eps, state[0])
    return _compilable_exp_avg_sq_(state, grad, beta2, eps, out)


def adaptive_gradient_clipping_(
    parameters: list[Tensor],
    gradients: list[Tensor],
    clip_val: float,
    minimum: float = 1e-3,
    eps: float = 1e-8,
):
    if clip_val <= 0:
        return
    p_norm = torch._foreach_norm(parameters)
    g_norm = torch._foreach_norm(gradients)
    torch._foreach_maximum_(p_norm, minimum)
    torch._foreach_maximum_(g_norm, eps)
    torch._foreach_div_(p_norm, g_norm)
    torch._foreach_mul_(p_norm, clip_val)
    torch._foreach_minimum_(p_norm, 1)
    torch._foreach_mul_(gradients, p_norm)


def clean():
    torch.cuda.empty_cache()
    gc.collect()


# @decorator
def zeropower_via_newtonschulz5(G, init, steps=2, eps=1e-7):
    r"""
    Modified from "modded-nanogpt" under the MIT license:
    Original: https://github.com/KellerJordan/modded-nanogpt/blob/a0dcbfdd9a0617d091d5123cfc354745428e40d3/train_gpt2.py

    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    init = init / (init.norm() + eps)  # ensure top singular value <= 1
    X = X / (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T  # preconditioner
        B = A @ init
        init = X = a * init + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


def ortho(x):
    zeroth_power_mode = "qr"
    if zeroth_power_mode == "qr":
        return torch.linalg.qr(x).Q
    if zeroth_power_mode == "svd":
        u, _s, v = torch.linalg.svd(x)
        return u @ v.T
    msg = f"Unknown zeroth_power_mode: {zeroth_power_mode}"
    raise NotImplementedError(msg)


def get_orthogonal_matrix_QR(GG, Q, exp_avg_sq):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q, strict=False):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))
        else:
            matrix.append(promote(m.data))
            orth_matrix.append(promote(o.data))

    indices = []

    for _ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q, strict=False)):
        if len(m) == 0:
            indices.append(None)
            continue

        tmp = m @ o
        est_eig = torch.einsum("ij,ij->j", o, tmp)
        sort_idx = torch.argsort(est_eig, descending=True)
        indices.append(sort_idx)
        zeroth_power_mode = "qr"
        if zeroth_power_mode == "eigh":
            set_(q, torch.linalg.eigh(m)[1])
        elif zeroth_power_mode.startswith("newtonschulz"):
            iterations = zeroth_power_mode[len("newtonschulz") :]
            iterations = 10 if not iterations else int(iterations)
            set_(q, zeropower_via_newtonschulz5(m, o[:, sort_idx], iterations))
        else:
            set_(q, ortho(tmp[:, sort_idx]))

    indices = tuple(
        slice(None)
        if ind is None
        else ind.view(*(1,) * i, -1, *(1,) * (exp_avg_sq.dim() - i - 1))  #
        for i, ind in enumerate(indices)
    )
    set_(exp_avg_sq, exp_avg_sq[indices])


def get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(promote(m.data))
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue

        device, dtype = m.device, m.dtype
        for modifier in (None, torch.double, "cpu"):
            if modifier is not None:
                m = m.to(modifier)  # noqa: PLW2901
            try:
                Q = torch.linalg.eigh(
                    m + 1e-30 * torch.eye(m.shape[0], device=m.device)
                )[1].to(device=device, dtype=dtype)
                break
            except torch.OutOfMemoryError:
                pass
            except RuntimeError:  # failed to compute eigenvalues
                continue
            clean()
        else:
            msg = "Failed to compute eigenvalues."
            raise RuntimeError(msg)

        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)

    return final


# @decorator
def compute_ggt(grad, GG, max_precond_dim, precondition_1d, beta):
    einsum_base = string.ascii_lowercase + string.ascii_uppercase
    if grad.dim() == 1 and (not precondition_1d or grad.shape[0] > max_precond_dim):
        return

    for idx, sh in enumerate(grad.shape):
        if sh > max_precond_dim:
            continue
        b = einsum_base[idx]
        g0 = einsum_base[: grad.dim()]
        g1 = g0.replace(b, b.upper())
        outer_product = torch.einsum(f"{g0},{g1}->{b + b.upper()}", grad, grad)
        GG[idx].lerp_(promote(outer_product), 1 - beta)


def update_preconditioner(
    grad, state, max_precond_dim, precondition_1d, beta, update_precond
):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    compute_ggt(grad, state["GG"], max_precond_dim, precondition_1d, beta)
    if state["Q"] is None:
        state["Q"] = get_orthogonal_matrix(state["GG"])
    if update_precond:
        get_orthogonal_matrix_QR(state["GG"], state["Q"], state["exp_avg_sq"])


def init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state["Q"] = None  # Will hold all the eigenbases of the preconditioner.
    state[
        "GG"
    ] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state["GG"].append([])
            return
        state["GG"].append(
            torch.zeros(
                grad.shape[0], grad.shape[0], device=grad.device, dtype=grad.dtype
            )
        )
        return

    for sh in grad.shape:
        if sh > max_precond_dim:
            state["GG"].append([])
        else:
            state["GG"].append(
                torch.zeros(sh, sh, device=grad.device, dtype=grad.dtype)
            )


# @decorator
def project(grad, Q, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    einsum_base = string.ascii_lowercase + string.ascii_uppercase
    param = einsum_base[: grad.dim()]
    preconditioners = ",".join([
        (g + g.upper())[:: -1 if back else 1]
        for m, g in zip(Q, param, strict=False)
        if len(m) > 0
    ])
    if preconditioners:
        out = "".join([c.upper() if c.upper() in preconditioners else c for c in param])
        grad = torch.einsum(
            f"{param},{preconditioners}->{out}", grad, *[q for q in Q if len(q) > 0]
        )
    return grad


# @decorator_knowngood
def _compilable_stochastic_lerp_(x: list[Tensor], y: list[Tensor], a: float | Tensor):
    for x_, y_ in zip(x, y, strict=False):
        x32 = promote(x_)
        y32 = promote(y_)
        x32.lerp_(y32, a)
        copy_stochastic_(x_, x32)


# @decorator_knowngood
def _compilable_mars_correction_(g: Tensor, old_g: Tensor, a: Tensor):
    g_copy = [g_.clone() for g_ in g]
    _compilable_stochastic_lerp_(g, old_g, a)
    copy_stochastic_list_(old_g, g_copy)


def mars_correction(g, old_g, beta1, gamma):
    a = -gamma * beta1 / (1 - beta1)
    g, old_g = list_guard(g), list_guard(old_g)
    a = scalar_guard(a, g[0])
    _compilable_mars_correction_(g, old_g, a)


def merge_group(group, *tensors):
    if not group.get("merge_dims", False):
        return tensors
    if isinstance(tensors[0], list):
        return [merge_group(group, *t) for t in tensors]

    out = []
    for t in tensors:
        append_or_extend(
            out,
            dim_merger(
                t,
                group["max_size_triangular"]
                if "max_size_triangular" in group
                else group["max_precond_dim"],
                group.get("split", False),
            ),
        )
    return out


class StatefulOptimizer(torch.optim.Optimizer):
    ema_decay: float = 0.001

    def __init__(self, params, defaults, foreach: bool = True, use_ema: bool = False):
        super().__init__(params, {**defaults, "foreach": foreach})
        self.fake_groups = {}
        self.use_ema = use_ema
        self.mapping = {}

    def get_groups(self, group):
        if group["foreach"]:
            return [group]

        for p in group["params"]:
            if p not in self.fake_groups:
                self.fake_groups[p] = {**group, "params": [p]}

        return [self.fake_groups[p] for p in group["params"]]

    def state_(self, arg: Tensor):
        return self.state[self.mapping.get(arg, arg)]

    def mars_correct_list(self, p_list, g_list, mars_gamma, beta):
        for p, g in zip(p_list, g_list, strict=False):
            state = self.state_(p)
            if "mars_old_grad" not in state:
                state["mars_old_grad"] = torch.zeros_like(g)
        old_gs = [self.state_(p)["mars_old_grad"] for p in p_list]
        mars_correction(g_list, old_gs, mars_gamma, beta)

    def split_p_and_g_in_group(
        self,
        group: dict,
        skip_none: bool = True,
        should_promote: bool = True,
        beta1: float = -1.0,
    ):
        for p in group["params"]:
            if skip_none and p.grad is None:
                continue

            if p.grad is None:
                grad = None
            else:
                grad = promote(p.grad) if should_promote else p.grad
                if beta1 >= 0 and group.get("mars"):
                    self.mars_correct_list([p], [grad], group["mars_gamma"], beta1)

                p.grad = None

            p_views = merge_group(group, p)
            if grad is not None:
                grad = merge_group(group, grad)
            for i, pv in enumerate(p_views):
                self.mapping[pv] = (p, i)
            if isinstance(p_views, Tensor):
                yield p_views, grad
                continue
            if grad is None:
                yield from zip(p_views, [None] * len(p_views), strict=False)
                continue
            yield from zip(p_views, grad, strict=False)

    def _step(self, group):
        raise NotImplementedError

    def ema_update(self):
        with torch.no_grad():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    active_p = list(group["params"])

                    if not active_p:
                        return

                    k = group["ema_step"] = group.get("ema_step", -1) + 1

                    for p in active_p:
                        if "param_ema" not in self.state_(p):
                            self.state_(p)["param_ema"] = torch.zeros_like(
                                p.data, memory_format=torch.preserve_format
                            )

                    y, param_ema = zip(
                        *[(p.data, self.state_(p)["param_ema"]) for p in active_p],
                        strict=False,
                    )
                    torch._foreach_lerp_(
                        param_ema, y, weight=beta_debias(1 - self.ema_decay, k + 1)
                    )

    def step(self, closure: Callable | None = None):
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()

        # we assume that parameters are constant and that there are no excessive recompiles
        with torch.no_grad(), torch._dynamo.utils.disable_cache_limit():
            for top_group in self.param_groups:
                for group in self.get_groups(top_group):
                    self._step(group)
                    self.mapping.clear()
                    if self.use_ema:
                        self.ema_update(group)

        return loss


class ScheduleFree(StatefulOptimizer):
    def eval(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1 = group["beta"] if "beta" in group else group["betas"][0]
            if beta1 > 0 and train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        # Set p.data to x
                        z = promote(state["z"])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - 1 / beta1)
                        copy_stochastic_(p.data, p32)
                group["train_mode"] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group["train_mode"]
            beta1 = group["beta"] if "beta" in group else group["betas"][0]
            if beta1 > 0 and not train_mode:
                for p in group["params"]:
                    state = self.state_(p)
                    if "z" in state:
                        z = promote(state["z"])
                        p32 = promote(p.data)
                        p32.lerp_(end=z, weight=1 - beta1)
                        copy_stochastic_(p.data, p32)
                group["train_mode"] = True

    def _step(self):
        raise NotImplementedError


def stochastic_lerp_(x: list[Tensor], y: list[Tensor], a: float | Tensor):
    x, y = list_guard(x), list_guard(y)
    a = scalar_guard(a, x[0])
    _compilable_stochastic_lerp_(x, y, a)


# @decorator_knowngood
def _compilable_cautioning_(g: Tensor, update: Tensor):
    mask = (g * update) > 0
    update.masked_fill_(~mask, 0)
    scale = mask.numel() / mask.sum().clamp(min=1)
    update.mul_(scale)


def caution(g, update):
    _compilable_cautioning_(g, update)


# @decorator_knowngood
def main_compilable_exp_avg_sq_(exp_avg_sq, grad_projected, old_debiased2, eps):
    eas32, gp32 = [list(map(promote, x)) for x in (exp_avg_sq, grad_projected)]
    denom = exp_avg_sq_(eas32, gp32, old_debiased2, eps)
    torch._foreach_div_(gp32, denom)

    copy_stochastic_list_(exp_avg_sq, eas32)
    copy_stochastic_list_(grad_projected, gp32)


class soap_sf(ScheduleFree):
    """

    Implementation adapted from HeavyBall: https://github.com/ClashLuke/HeavyBall
    Based on research:
    - SOAP: Improving and Stabilizing Shampoo using Adam
      https://arxiv.org/abs/2409.11321
    - ScheduleFree: The Road Less Scheduled
      https://arxiv.org/abs/2405.15682
    - PaLM: Scaling Language Modeling with Pathways
      https://arxiv.org/abs/2204.02311

    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        beta=0.9,
        beta2_scale: float = 0.8,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 2048,  #
        merge_dims: bool = True,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        correct_bias: bool = True,
        warmup_steps: int = 1600,
        r=0.0,
        weight_lr_power=2.0,
        gradient_clip_val: float = 0.1,
        betas=(None, None),
        split: bool = False,
        foreach: bool = True,
        mars: bool = False,
        caution: bool = False,
        mars_gamma: float = 0.0025,
        **kwargs,  # noqa: ARG002
    ):
        if betas[0] is not None:
            beta = betas[0]

        assert not caution, "Caution is not implemented in ScheduleFree optimizers"

        defaults = {
            "lr": lr,
            "beta": beta,
            "beta2_scale": beta2_scale,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
            "warmup_steps": warmup_steps,
            "r": r,
            "weight_lr_power": weight_lr_power,
            "train_mode": True,
            "step": -1,
            "gradient_clip_val": gradient_clip_val,
            "weight_sum": 0,
            "split": split,
            "mars": mars,
            "caution": caution,
            "mars_gamma": mars_gamma,
        }
        super().__init__(params, defaults, foreach)

    def _step(self, group):
        vals = []
        max_precond_dim = group["max_precond_dim"]
        precondition_1d = group["precondition_1d"]
        mars = group["mars"]

        step = group["step"] = group.get("step", 0) + 1

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.float()
            vals.append((p, grad))

        if not vals:
            return

        p_list, grad = zip(*vals, strict=False)

        adaptive_gradient_clipping_(
            p_list, grad, group["gradient_clip_val"], eps=group["eps"]
        )

        vals = []

        for p, g in self.split_p_and_g_in_group(group, beta1=group["beta"]):
            state = self.state_(p)

            if "z" not in state:
                state["z"] = torch.clone(p).float()
                state["exp_avg_sq"] = torch.zeros_like(
                    g, dtype=torch.float32, memory_format=torch.preserve_format
                )
                if mars:
                    state["mars_prev_grad"] = g.clone()
                init_preconditioner(g, state, max_precond_dim, precondition_1d)
                update_preconditioner(
                    g, state, max_precond_dim, precondition_1d, 0, update_precond=True
                )
                continue  # first step is skipped so that we never use the current gradients in the projection.

            # Projecting gradients to the eigenbases of Shampoo's preconditioner
            # i.e. projecting to the eigenbases of matrices in state['GG']
            grad_projected = project(g, state["Q"], back=False)
            z, exp_avg_sq = state["z"], state["exp_avg_sq"]
            vals.append((p, g, grad_projected, z, exp_avg_sq))

        if not vals:
            return

        p_list, grad, grad_projected, z, exp_avg_sq = zip(*vals, strict=False)

        beta2 = 1 - max(step, 1) ** -group["beta2_scale"]
        new_debiased2 = beta_debias(beta2, step)

        # Decay the first and second moment running average coefficient
        # In-place operations to update the averages at the same time
        old_debiased_tensor = torch.empty(
            (), dtype=torch.float32, device=p_list[0].device
        ).fill_(new_debiased2)
        main_compilable_exp_avg_sq_(
            exp_avg_sq, grad_projected, old_debiased_tensor, group["eps"]
        )

        update_precond = (
            group["step"] > 0 and group["step"] % group["precondition_frequency"] == 0
        )

        for p, g, gp in zip(p_list, grad, grad_projected, strict=False):
            state = self.state_(p)
            # Projecting back the preconditioned (by Adam) exponential moving average of gradients
            # to the original space
            # CANT DO /= HERE AS EXP_AVG MAY POINT TO THE BUFFER
            set_(gp, project(gp, state["Q"], back=True))

            update_preconditioner(
                g,
                state,
                max_precond_dim,
                precondition_1d,
                1 - new_debiased2,
                update_precond,
            )

        # Weight decay calculated at y
        if group["weight_decay"] > 0:
            torch._foreach_add_(grad, p_list, alpha=group["weight_decay"])

        lr = warmup(group["lr"], step, group["warmup_steps"])
        group["weight_sum"] = schedule_free_(
            lr,
            group["weight_lr_power"],
            group["weight_sum"],
            group["beta"],
            p_list,
            z,
            grad_projected,
            group["r"],
            step,
        )
