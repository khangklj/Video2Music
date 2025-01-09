import math
from typing import List, Optional

import torch
from torch import Tensor

from torch.optim.optimizer import (
    Optimizer,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _dispatch_sqrt,
    _foreach_doc,
    _get_value,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
)

__all__ = ["RAdamW", "radanw"]


class RAdanW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.92, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= betas[3] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 3: {betas[3]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]), dtype=torch.float32)

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, exp_diffs, exp_diff_sqs, neg_prev_grads, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            
            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("RAdanW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            # Lazy state initialization
            if len(state) == 0:
                state["step"] = torch.tensor(0.0, dtype=torch.float32)
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                
                state["exp_diff"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                
                state["exp_diff_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            if 'neg_prev_grad' not in state or group['step'] == 1:
                state['neg_prev_grad'] = p.grad.clone().mul_(-1.0)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            exp_diffs.append(state["exp_diff"])
            exp_diff_sqs.append(state["exp_diff_sq"])
            neg_prev_grads.append(state["neg_prev_grad"])
            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_diffs = []
            exp_diff_sqs = []
            neg_prev_grads = []
            state_steps = []
            beta1, beta2, beta3, beta4 = group["betas"]

            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, 
                                           exp_diffs, exp_diff_sqs, neg_prev_grads, state_steps)

            radanw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_diffs,
                exp_diff_sqs,
                neg_prev_grads,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                beta4=beta4,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                foreach=group["foreach"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        return loss

def radanw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_diffs: List[Tensor],
    exp_diff_sqs: List[Tensor],
    neg_prev_grads: List[Tensor],
    state_steps: List[Tensor],
    foreach: Optional[bool] = None,
    differentiable: bool = False,
    has_complex: bool = False,
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    lr: float,
    weight_decay: float,
    eps: float,
):
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_radanw
    else:
        func = _single_tensor_radanw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        exp_diffs,
        exp_diff_sqs,
        neg_prev_grads,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        differentiable=differentiable,
        has_complex=has_complex,
    )


def _single_tensor_radanw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],       # m_t
    exp_avg_sqs: List[Tensor],    # v_t
    exp_diffs: List[Tensor],      # d_t
    exp_diff_sqs: List[Tensor],   # n_t
    neg_prev_grads: List[Tensor], # -g_(t-1)
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    lr: float,
    weight_decay: float,
    eps: float,
    differentiable: bool,
    has_complex: bool,
):

    for i, param in enumerate(params):
        grad = grads[i]
        neg_prev_grad = neg_prev_grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_diff = exp_diffs[i]
        exp_diff_sq = exp_diff_sqs[i]
        step_t = state_steps[i]

        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # update step
        step_t += 1
        step = _get_value(step_t)

        # ============< AdamW >============ #
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)                                # m_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = exp_avg / bias_correction1           # m_hat_t

        # ============< RAdam >============ #
        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        if rho_t > 5.0:
            # Compute the variance rectification term and update parameters accordingly
            rect = math.sqrt(
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            )                                                         # r_t
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
            adaptive_lr = math.sqrt(bias_correction2) / exp_avg_sq_sqrt # l_t
        else:
            rect = 1.0                                                # r_t
            adaptive_lr = 1.0                                         # l_t

        param.add_(bias_corrected_exp_avg * lr * adaptive_lr * rect, alpha=-1.0)

        # ============< Adan >============ #
        neg_prev_grad.add_(grad)                                    # g_t - g_(t-1)
        exp_diff.mul_(beta3).add_(neg_prev_grad, alpha=1 - beta3)   # d_t

        exp_diff_sq.mul_(beta4).add_(grad.add_(neg_prev_grad, alpha=1 - beta3).pow_(2), alpha=1 - beta4)     # n_t
        
        exp_diff_sq_sqrt = exp_diff_sq.sqrt()
        if differentiable:
            exp_diff_sq_sqrt = exp_diff_sq_sqrt.add(eps)
        else:
            exp_diff_sq_sqrt = exp_diff_sq_sqrt.add_(eps)
        eta = lr / exp_diff_sq_sqrt                                 # eta_t

        param.add_(exp_diff.mul_(eta), alpha=-(1 - beta3))

        neg_prev_grad.zero_().add_(grad, alpha=-1.0)

def _multi_tensor_radanw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],       # m_t
    exp_avg_sqs: List[Tensor],    # v_t
    exp_diffs: List[Tensor],      # d_t
    exp_diff_sqs: List[Tensor],   # n_t
    neg_prev_grads: List[Tensor], # -g_(t-1)
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    lr: float,
    weight_decay: float,
    eps: float,
    differentiable: bool,
    has_complex: bool,
):

    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, 
                                                                    exp_diffs, exp_diff_sqs, neg_prev_grads,
                                                                    state_steps])
    for ((
        grouped_params,
        grouped_grads,
        grouped_exp_avgs,
        grouped_exp_avg_sqs,
        grouped_exp_diffs,
        grouped_exp_diff_sqs,
        grouped_neg_prev_grads,
        grouped_state_steps,
    ), _) in grouped_tensors.values():
        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_avg_sqs,
                          grouped_exp_diffs, grouped_exp_diff_sqs, grouped_neg_prev_grads)

        # ============< AdamW >============ #
        bias_correction1 = [1 - beta1 ** _get_value(step) for step in grouped_state_steps]
        
        torch._foreach_mul_(grouped_params, 1 - lr * weight_decay)
        
        # Decay the first and second moment running average coefficient
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        torch._foreach_mul_(grouped_exp_avg_sqs, beta2)
        torch._foreach_addcmul_(grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2)

        # correcting bias for the first moving moment
        bias_corrected_exp_avg = torch._foreach_div(grouped_exp_avgs, bias_correction1)

        # ============< RAdam >============ #
        # maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho_t_list = [rho_inf - 2 * _get_value(step) * (beta2 ** _get_value(step)) /
                      (1 - beta2 ** _get_value(step)) for step in grouped_state_steps]
        
        # Delete the local intermediate since it won't be used anymore to save on peak memory
        del grouped_grads

        rect = [
            _dispatch_sqrt(
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            )
            if rho_t > 5
            else 0
            for rho_t in rho_t_list
        ]
        unrectified = [0 if rect > 0 else 1.0 for rect in rect]

        unrect_step_size = _stack_if_compiling([(lr * rect / bc) * -1 for rect, bc in zip(unrectified, bias_correction1)])
        bias_correction2_sqrt_times_rect_step_size = [
            _dispatch_sqrt(1 - beta2 ** _get_value(step)) * (lr * rect / bc) * -1
            for step, rect, bc in zip(grouped_state_steps, rect, bias_correction1)
        ]

        buffer = torch._foreach_sqrt(grouped_exp_avg_sqs)
        torch._foreach_add_(buffer, eps)
        torch._foreach_div_(buffer, bias_correction2_sqrt_times_rect_step_size)
        torch._foreach_reciprocal_(buffer)
        torch._foreach_add_(buffer, unrect_step_size)

        # Here, buffer = sqrt(1 - beta2^t) * rect_step_size / (sqrt(v) + eps) + unrect_step_size
        torch._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)
        
        # ============< Adan >============ #
        torch._foreach_add_(grouped_neg_prev_grads, grouped_grads)
        
        torch._foreach_mul_(grouped_exp_diffs, beta3)
        buffer = torch._foreach_mul(grouped_neg_prev_grads, 1 - beta3)
        torch._foreach_add_(grouped_exp_diffs, buffer)      # d_t

        torch._foreach_mul_(grouped_exp_diff_sqs, beta4)
        buffer = torch._foreach_mul(grouped_neg_prev_grads, 1 - beta3)
        torch._foreach_add_(grouped_grads, buffer)
        torch._foreach_pow_(buffer, 2)
        torch._foreach_mul_(buffer, 1 - beta4)

        torch._foreach_add_(grouped_exp_diff_sqs, buffer)   # n_t

        grouped_exp_diff_sqs_sqrt = torch._foreach_sqrt(grouped_exp_diff_sqs)
        torch._foreach_add_(grouped_exp_diff_sqs_sqrt, eps)

        eta = lr / grouped_exp_diff_sqs_sqrt

        buffer = torch._foreach_mul(grouped_exp_diffs, eta, -(1 - beta3))
        torch._foreach_add_(grouped_params, grouped_exp_diffs)

        torch._foreach_zero_(grouped_neg_prev_grads)
        buffer = torch._foreach_mul(grouped_grads, -1.0)
        torch._foreach_add_(grouped_neg_prev_grads, buffer)
