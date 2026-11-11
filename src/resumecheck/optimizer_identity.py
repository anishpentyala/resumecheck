"""Opt-in optimizer observations bound to live model parameter names.

Not a loader or preflight. CPU, exact built-in SGD/Adam/AdamW only.
"""
from __future__ import annotations

from .recorder import fingerprint_tensor


def fingerprint_named_optimizer(model, optimizer):
    """Fingerprint state_dict plus each group's actual parameter ownership.

    Resolve names through live Parameter identity, not state_dict param_names,
    which loading can overwrite. Compare observations at the same point in two
    runs. This does not certify, load, repair or remap checkpoint state. Model
    parameters outside the optimizer are allowed. Aliases, wrappers, repeated
    parameters, state_dict hooks, and non-CPU/non-strided parameters fail.
    Torch is required only when called. Uses the existing v2 content hash.
    """
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        raise TypeError('only exact built-in SGD, Adam and AdamW optimizers are supported')
    hook_maps = [getattr(optimizer, name, None) for name in (
        '_optimizer_state_dict_pre_hooks', '_optimizer_state_dict_post_hooks')]
    if any(not isinstance(hooks, dict) for hooks in hook_maps):
        raise ValueError('optimizer hook registry layout is unsupported')
    if any(hook_maps):
        raise ValueError('optimizer state_dict hooks are unsupported')
    names_by_id = {}
    seen_names = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError('model parameter names must be unique nonempty strings')
        seen_names.add(name)
        names_by_id.setdefault(id(parameter), []).append(name)
    binding = []
    seen_parameters = set()
    for group in optimizer.param_groups:
        row = []
        for parameter in group['params']:
            ident = id(parameter)
            names = names_by_id.get(ident, [])
            if len(names) != 1:
                raise ValueError('each optimizer parameter needs exactly one live model name')
            if ident in seen_parameters:
                raise ValueError('repeated optimizer parameters are unsupported')
            seen_parameters.add(ident)
            if parameter.device.type != 'cpu' or parameter.layout != torch.strided:
                raise ValueError('only CPU strided optimizer parameters are supported')
            row.append({'name': names[0], 'shape': tuple(parameter.shape),
                        'dtype': str(parameter.dtype),
                        'requires_grad': bool(parameter.requires_grad)})
        if not row:
            raise ValueError('empty optimizer parameter groups are unsupported')
        binding.append(row)
    if not binding:
        raise ValueError('optimizer must contain parameters')
    state = optimizer.state_dict()
    groups = state.get('param_groups')
    if not isinstance(groups, list) or len(groups) != len(binding):
        raise ValueError('serialized/live optimizer group topology differs')
    ids = set()
    for group, row in zip(groups, binding):
        params = group.get('params')
        if not isinstance(params, list) or len(params) != len(row):
            raise ValueError('serialized/live optimizer parameter counts differ')
        for ident in params:
            if type(ident) is not int or ident < 0 or ident in ids:
                raise ValueError('serialized optimizer parameter IDs must be distinct integers')
            ids.add(ident)
    states = state.get('state')
    if not isinstance(states, dict) or any(type(k) is not int or k not in ids for k in states):
        raise ValueError('serialized optimizer state has unbound parameter IDs')
    return fingerprint_tensor({'kind': 'resumecheck.named-optimizer.v1',
                               'optimizer': type(optimizer).__name__,
                               'binding': binding, 'state_dict': state})
