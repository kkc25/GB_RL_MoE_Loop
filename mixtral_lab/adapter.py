"""Frozen, terminal-FFN Mixtral adapter for Transformers 5.15.1.

This is a research path, not a fused deployment kernel. Prefix capture raises a
private exception before the last FFN and always removes its temporary hook.
CPU/disk offload hooks are unsupported; GPU-to-GPU sharding is permitted.
The native model and all its weights remain unmodified after each call.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import torch
from torch.nn import functional as F

PAIR_TABLE = tuple(itertools.combinations(range(8), 2))


def action_to_pair(actions: torch.Tensor) -> torch.Tensor:
    if actions.dtype != torch.long or bool(((actions < 1) | (actions > 28)).any()):
        raise ValueError('pair actions must be int64 in 1..28; STOP is not an expert pair')
    return torch.tensor(PAIR_TABLE, dtype=torch.long, device=actions.device)[actions - 1]


def pair_to_action(pairs: torch.Tensor) -> torch.Tensor:
    if pairs.dtype != torch.long or pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError('pairs must be int64 [N,2]')
    a, b = pairs.sort(dim=-1).values.unbind(-1)
    if bool(((a < 0) | (b >= 8) | (a == b)).any()):
        raise ValueError('each pair must contain two distinct experts in 0..7')
    return a * 8 - a * (a + 1) // 2 + b - a  # 1-based IDs


def _has_offload_hook(hook):
    return bool(getattr(hook, 'offload', False)) or any(
        _has_offload_hook(child) for child in getattr(hook, 'hooks', ()))


def _device(module):
    parameter = next(module.parameters())
    if parameter.device.type == 'meta':
        raise ValueError('meta/offloaded module is unsupported')
    return parameter.device


@dataclass
class Observation:
    h: torch.Tensor
    h0: torch.Tensor
    n: torch.Tensor
    router_logits: torch.Tensor
    native_pairs: torch.Tensor
    rounds: torch.Tensor
    previous_pairs: torch.Tensor
    max_calls: int
    features: torch.Tensor


@dataclass
class AdapterOutput:
    logits: torch.Tensor
    final_hidden: torch.Tensor
    actions: torch.Tensor
    rounds: torch.Tensor


class _CapturedPrefix(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


class MixtralAdapter:
    """Only the final pretrained FFN is controlled; native residual scale is 1."""
    def __init__(self, model, *, head_chunk=128):
        if getattr(model.config, 'model_type', None) != 'mixtral':
            raise ValueError('expected MixtralForCausalLM')
        if model.config.num_local_experts != 8 or model.config.num_experts_per_tok != 2:
            raise ValueError('this experiment requires exactly 8 experts and Top-2 routing')
        if isinstance(head_chunk, bool) or not isinstance(head_chunk, int) or head_chunk < 1:
            raise ValueError('head_chunk must be a positive integer')
        if getattr(model, 'is_quantized', False) or getattr(model, 'hf_quantizer', None) is not None:
            raise ValueError('quantized models are not part of the BF16/FP32 reference experiment')
        mapping = getattr(model, 'hf_device_map', {})
        if any(str(value) in ('cpu', 'disk', 'meta') for value in mapping.values()):
            raise ValueError('CPU/disk/meta offload is unsupported; use resident GPU shards')
        for module in model.modules():
            hook = getattr(module, '_hf_hook', None)
            if hook is not None and _has_offload_hook(hook):
                raise ValueError('offload hooks are incompatible with early prefix capture')
        self.model = model.eval().requires_grad_(False)
        self.layer = model.model.layers[-1]
        self.moe = self.layer.mlp
        if not hasattr(self.moe.experts, 'gate_up_proj') or not hasattr(self.moe.experts, 'down_proj'):
            raise ValueError('requires the Transformers 5.15.1 packed Mixtral expert layout')
        self.head_chunk = head_chunk
        self.hidden_size = model.config.hidden_size
        self.feature_dim = 2 * self.hidden_size + 20
        self.device = _device(self.layer.post_attention_layernorm)
        if any(p.device != self.device for p in self.moe.parameters()):
            raise ValueError('terminal norm, gate and experts must share one device; do not split decoder layers')
        self._capturing = False

    @torch.no_grad()
    def capture(self, input_ids, attention_mask=None):
        if self._capturing:
            raise RuntimeError('adapter prefix capture is not reentrant')
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError('input_ids must be int64 [batch,sequence]')
        self._capturing = True
        def stop_before_ffn(module, args):
            raise _CapturedPrefix(args[0].detach())
        handle = self.layer.post_attention_layernorm.register_forward_pre_hook(stop_before_ffn)
        try:
            try:
                self.model.model(input_ids=input_ids, attention_mask=attention_mask,
                                 use_cache=False, return_dict=True)
            except _CapturedPrefix as captured:
                return captured.hidden
            raise RuntimeError('terminal FFN was not reached')
        finally:
            handle.remove()
            self._capturing = False

    def _flat(self, h):
        if h.ndim not in (2, 3) or h.shape[-1] != self.hidden_size:
            raise ValueError('hidden must be [N,D] or [B,S,D]')
        if h.device != self.device:
            raise ValueError('hidden must be resident on the terminal layer device')
        return h.reshape(-1, self.hidden_size)

    @torch.no_grad()
    def route(self, h):
        n = self.layer.post_attention_layernorm(self._flat(h))
        logits, weights, pairs = self.moe.gate(n)
        return logits, pairs, weights

    @torch.no_grad()
    def native_pair(self, h):
        return self.route(h)[1]

    @torch.no_grad()
    def observe(self, h, h0=None, rounds=None, previous_pairs=None, max_calls=1):
        if isinstance(max_calls, bool) or max_calls not in (1, 2):
            raise ValueError('diagnostic calls must be 1 or 2')
        h = self._flat(h)
        h0 = h if h0 is None else self._flat(h0)
        if h0.shape != h.shape:
            raise ValueError('h0 and h shapes differ')
        count = len(h)
        rounds = torch.zeros(count, dtype=torch.long, device=h.device) if rounds is None else rounds
        previous_pairs = torch.full((count, 2), -1, dtype=torch.long, device=h.device) if previous_pairs is None else previous_pairs
        if rounds.shape != (count,) or rounds.dtype != torch.long or bool(((rounds < 0) | (rounds >= max_calls)).any()):
            raise ValueError('invalid active observation rounds')
        if rounds.device != h.device or previous_pairs.device != h.device:
            raise ValueError('observation tensors must share the terminal device')
        if previous_pairs.shape != (count, 2) or previous_pairs.dtype != torch.long:
            raise ValueError('previous_pairs must be int64 [N,2]')
        fresh = rounds == 0
        if bool((previous_pairs[fresh] != -1).any()):
            raise ValueError('first observation must not contain a previous pair')
        pair_to_action(previous_pairs[~fresh])
        n = self.layer.post_attention_layernorm(h)
        n0 = self.layer.post_attention_layernorm(h0)
        logits, _, pairs = self.moe.gate(n)
        centered = logits.float() - logits.float().mean(-1, keepdim=True)
        previous = torch.zeros(count, 8, dtype=torch.float32, device=h.device)
        has_pair = (previous_pairs >= 0).all(-1)
        if bool(has_pair.any()):
            previous[has_pair] = previous[has_pair].scatter(1, previous_pairs[has_pair], 1.)
        drift = torch.log1p((h.float()-h0.float()).norm(dim=-1)/(h0.float().norm(dim=-1)+1e-8))
        scalars = torch.stack((rounds.float()/max_calls, (max_calls-rounds).float()/max_calls,
                               (rounds == 0).float(), drift), -1)
        features = torch.cat((n.float(), n0.float(), centered, previous, scalars), -1)
        return Observation(h, h0, n, logits, pairs, rounds, previous_pairs, max_calls, features)

    def features(self, h):
        return self.observe(h).features

    @torch.no_grad()
    def step_pairs(self, h, pairs):
        flat = self._flat(h)
        pair_to_action(pairs)  # structural/range validation
        if pairs.shape[0] != len(flat) or pairs.device != flat.device:
            raise ValueError('pair rows/device must match hidden')
        n = self.layer.post_attention_layernorm(flat)
        logits, native_weights, native_pairs = self.moe.gate(n)
        same = (pairs.sort(-1).values == native_pairs.sort(-1).values).all(-1)
        selected = torch.where(same[:, None], native_pairs, pairs)
        # Conditional softmax avoids 0/0 if both non-native probabilities
        # underflow in the full eight-expert distribution. Native rows retain
        # HF's exact full-softmax/topk/renormalization arithmetic below.
        weights = torch.softmax(logits.float().gather(1, selected), -1)
        weights = torch.where(same[:, None], native_weights, weights)
        update = self.moe.experts(n, selected, weights)
        return (flat + update).reshape(h.shape)

    @torch.no_grad()
    def step(self, h, actions):
        return self.step_pairs(h, action_to_pair(actions))

    @torch.no_grad()
    def readout(self, h):
        self._flat(h)
        normalized = self.model.model.norm(h.to(_device(self.model.model.norm)))
        return self.model.lm_head(normalized.to(_device(self.model.lm_head)))

    @torch.no_grad()
    def score_pairs(self, h, targets, pairs):
        hidden = self.step_pairs(h, pairs).reshape(-1, self.hidden_size)
        if targets.shape != (len(hidden),) or targets.dtype != torch.long:
            raise ValueError('targets must be int64 [N]')
        pieces = []
        for start in range(0, len(hidden), self.head_chunk):
            logits = self.readout(hidden[start:start+self.head_chunk])
            pieces.append(F.cross_entropy(logits.float(), targets[start:start+self.head_chunk].to(logits.device),
                                          reduction='none').to(hidden.device))
        return torch.cat(pieces) if pieces else hidden.new_empty((0,), dtype=torch.float32)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, *, policy=None, max_calls=1):
        if isinstance(max_calls, bool) or max_calls not in (1, 2):
            raise ValueError('max_calls must be 1 or 2')
        h0 = self.capture(input_ids, attention_mask)
        shape = h0.shape
        h = h0.reshape(-1, self.hidden_size).clone()
        initial = h.clone()
        count = len(h)
        rounds = torch.zeros(count, dtype=torch.long, device=h.device)
        done = torch.zeros(count, dtype=torch.bool, device=h.device)
        previous = torch.full((count, 2), -1, dtype=torch.long, device=h.device)
        trace = torch.full((count, max_calls), -1, dtype=torch.long, device=h.device)
        for slot in range(max_calls):
            indices = torch.nonzero(~done, as_tuple=False).flatten()
            if not len(indices):
                break
            obs = self.observe(h[indices], initial[indices], rounds[indices], previous[indices], max_calls)
            actions = pair_to_action(obs.native_pairs) if policy is None else policy(obs)
            if actions.dtype != torch.long or actions.shape != (len(indices),):
                raise ValueError('policy must return int64 [active_rows] actions')
            if actions.device != h.device or bool(((actions < 0) | (actions > 28)).any()):
                raise ValueError('policy actions must be on terminal device in 0..28')
            stop = actions == 0
            if bool((stop & (rounds[indices] == 0)).any()):
                raise ValueError('STOP is illegal before the first FFN call')
            trace[indices, slot] = actions
            done[indices[stop]] = True
            continuing = indices[~stop]
            if len(continuing):
                chosen = actions[~stop]
                h[continuing] = self.step(h[continuing], chosen)
                previous[continuing] = action_to_pair(chosen)
                rounds[continuing] += 1
                done[continuing] = rounds[continuing] >= max_calls
        final = h.reshape(shape)
        return AdapterOutput(self.readout(final), final, trace.reshape(*shape[:-1], max_calls), rounds.reshape(shape[:-1]))
