"""Optional supervised correction after *fixed native* R1, not recurrent MoE.

This adds parameters, not a second expert call or a learned STOP policy. Results
cannot establish a loop advantage without a parameter/compute-matched control.
FP32 low-rank arithmetic; correction is cast to native hidden dtype before the
residual add. Frozen final norm/head remain differentiable with respect to h.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import io
import json
import math
import os
import tempfile
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = 'mixtral-second-round-residual-v1'


@dataclass(frozen=True)
class SecondRoundConfig:
    rank: int = 8
    updates: int = 20
    batch_contexts: int = 8
    max_train_contexts: int = 256
    eval_interval: int = 5
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    residual_scale: float = 0.1
    seed: int = 17

    def validate(self):
        for name, upper in [('rank', 256), ('updates', 10000), ('batch_contexts', 1024),
                            ('max_train_contexts', 100000), ('eval_interval', 10000)]:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= upper:
                raise ValueError(f'{name} must be integer in [1,{upper}]')
        for name in ('learning_rate', 'gradient_clip', 'residual_scale'):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'{name} must be finite positive')
        if self.residual_scale > 1:
            raise ValueError('residual_scale must be <= 1')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be nonnegative int64')


class SecondRoundResidual(nn.Module):
    """Fixed-on low-rank residual; bypassing this module is the STOP control."""
    def __init__(self, hidden_size, config, device='cpu'):
        super().__init__()
        config.validate()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < config.rank:
            raise ValueError('hidden_size must be >= rank')
        self.config = config
        self.hidden_size = hidden_size
        # Local CPU generator: initialization does not consume global CPU/CUDA RNG.
        gen = torch.Generator(device='cpu').manual_seed(config.seed)
        a = torch.randn(config.rank, hidden_size, generator=gen) / math.sqrt(hidden_size)
        self.down = nn.Parameter(a.to(device=device, dtype=torch.float32))
        self.up = nn.Parameter(torch.zeros(hidden_size, config.rank, device=device, dtype=torch.float32))

    def correction(self, h1):
        if h1.shape[-1] != self.hidden_size or h1.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('expected FP32/BF16 native hidden with matching dimension')
        x = h1.detach().float()
        # Parameter-free RMS normalization, no checkpoint norm mutation.
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        return F.linear(F.linear(x, self.down), self.up) * self.config.residual_scale

    def forward(self, h1):
        # Detached native prefix is a contract, not full-model fine-tuning.
        return h1.detach() + self.correction(h1).to(h1.dtype)


def differentiable_readout(adapter, hidden):
    """Unlike adapter.readout, deliberately NOT under no_grad."""
    if any(p.requires_grad for p in adapter.model.parameters()):
        raise ValueError('backbone must be frozen')
    norm, head = adapter.model.model.norm, adapter.model.lm_head
    np, hp = next(norm.parameters()), next(head.parameters())
    normalized = norm(hidden.to(device=np.device, dtype=np.dtype))
    return head(normalized.to(device=hp.device, dtype=hp.dtype))


def _digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _binding(identities):
    return hashlib.sha256(json.dumps(identities, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic_write(path, writer):
    """Publish only a complete sibling file; preserve existing result on failure."""
    path = Path(path)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save(path, module, identities, update, nll):
    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    data = dict(format=FORMAT, config=asdict(module.config), hidden_size=module.hidden_size,
                    identities=identities, binding_sha256=_binding(identities), state=state,
                    state_sha256=_digest(state), contract_sha256=_binding(dict(identities=identities, config=asdict(module.config), hidden_size=module.hidden_size)), update=update, selection_nll=nll)
    _atomic_write(path, lambda temporary: torch.save(data, temporary))


def load_second_round_checkpoint(path, identities, device='cpu', expected_config=None, expected_hidden_size=None):
    """Identity-bound inference load only; training never resumes this checkpoint."""
    if not isinstance(identities, dict) or not identities:
        raise ValueError('nonempty identity mapping is required')
    payload = Path(path).read_bytes()
    data = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if data.get('format') != FORMAT or data.get('binding_sha256') != _binding(identities) or data.get('identities') != identities:
        raise ValueError('second-round checkpoint identity/format mismatch')
    contract = dict(identities=identities, config=data['config'], hidden_size=data['hidden_size'])
    if data.get('contract_sha256') != _binding(contract):
        raise ValueError('second-round configuration binding mismatch')
    if expected_config is not None and data['config'] != asdict(expected_config):
        raise ValueError('second-round expected configuration mismatch')
    if expected_hidden_size is not None and data['hidden_size'] != expected_hidden_size:
        raise ValueError('second-round expected hidden size mismatch')
    if _digest(data['state']) != data['state_sha256']:
        raise ValueError('second-round state digest mismatch')
    module = SecondRoundResidual(data['hidden_size'], SecondRoundConfig(**data['config']), device)
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in data['state'].values()):
        raise ValueError('invalid second-round weights')
    module.load_state_dict(data['state'], strict=True)
    return module.eval()


def train_second_round(adapter, train_h0, train_targets, selection_evaluator,
                       config, identities, output_dir):
    """Return selection-best residual and report; callback(module) -> {'nll':...}.

    Callback must evaluate fixed native R1 plus module on selection only, without
    fitting. Update zero participates and ties retain it. This is not held-out
    evidence or a per-token learned STOP gate. Train inputs are already shifted.
    """
    config.validate()
    if any(p.requires_grad for p in adapter.model.parameters()):
        raise ValueError('backbone must be frozen')
    if train_h0.ndim != 2 or train_h0.shape[1] != adapter.hidden_size or not 1 <= len(train_h0) <= config.max_train_contexts:
        raise ValueError('invalid/beyond-budget training hidden contexts')
    if train_targets.shape != (len(train_h0),) or train_targets.dtype != torch.long:
        raise ValueError('training targets must be int64 [contexts], already shifted')
    if not torch.isfinite(train_h0).all():
        raise ValueError('nonfinite training contexts')
    if bool(((train_targets < 0) | (train_targets >= adapter.model.config.vocab_size)).any()):
        raise ValueError('targets outside vocabulary')
    if not isinstance(identities, dict) or not identities:
        raise ValueError('nonempty identity mapping is required')
    _binding(identities)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)  # no overwrite or resume
    start = time.perf_counter()
    h0 = train_h0.detach().to(adapter.device)
    # Chunk native prefix completion to bound expert workspace.
    with torch.no_grad():
        h1 = torch.cat([adapter.step_pairs(c, adapter.native_pair(c))
                        for c in h0.split(config.batch_contexts)]).detach()
    labels = train_targets.detach().to(adapter.device)
    module = SecondRoundResidual(adapter.hidden_size, config, adapter.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=config.learning_rate, weight_decay=0.)
    generator = torch.Generator(device='cpu').manual_seed(config.seed)
    def evaluate():
        module.eval()
        with torch.no_grad():
            result = selection_evaluator(module)
        nll = float(result['nll'])
        if not math.isfinite(nll):
            raise ValueError('nonfinite selection NLL')
        return nll
    initial = best_nll = evaluate()
    best_state = copy.deepcopy(module.state_dict())
    best_update, successful, rows = 0, 0, []
    _save(output/'checkpoint_initial.pt', module, identities, 0, initial)
    _save(output/'checkpoint_best.pt', module, identities, 0, initial)
    status = 'completed'
    for update in range(1, config.updates + 1):
        tick = time.perf_counter()
        module.train()
        idx = torch.randperm(len(h1), generator=generator)[:config.batch_contexts].to(h1.device)
        optimizer.zero_grad(set_to_none=True)
        before = copy.deepcopy(module.state_dict())
        corrected = module(h1[idx])
        logits = differentiable_readout(adapter, corrected)
        loss = F.cross_entropy(logits.float(), labels[idx].to(logits.device))
        if not torch.isfinite(loss):
            status = 'stopped_nonfinite_loss'
            break
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip)
        if not torch.isfinite(grad):
            status = 'stopped_nonfinite_gradient'
            break
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in module.parameters()):
            module.load_state_dict(before)
            status = 'stopped_nonfinite_parameters'
            break
        successful += 1
        with torch.no_grad():
            correction = module.correction(h1[idx])
            row = dict(update=update, train_nll=float(loss.detach()), gradient_norm=float(grad),
                       batch_contexts=len(idx), correction_rms=float(correction.square().mean().sqrt()),
                       correction_to_hidden_norm=float(correction.norm()/h1[idx].float().norm().clamp_min(1e-12)),
                       elapsed_seconds=time.perf_counter()-tick)
        if update % config.eval_interval == 0 or update == config.updates:
            nll = evaluate()
            row['selection_nll'] = nll
            row['passes_r1_selection_gate'] = nll <= initial
            if nll < best_nll:
                best_nll, best_update = nll, update
                best_state = copy.deepcopy(module.state_dict())
                _save(output/'checkpoint_best.pt', module, identities, update, nll)
        rows.append(row)
    # Assess last accepted weights even if a later step hit a numerical guard.
    latest_nll = evaluate()
    if latest_nll < best_nll:
        best_nll, best_update = latest_nll, successful
        best_state = copy.deepcopy(module.state_dict())
        _save(output/'checkpoint_best.pt', module, identities, best_update, best_nll)
    _save(output/'checkpoint_latest.pt', module, identities, successful, latest_nll)
    module.load_state_dict(best_state)
    module.zero_grad(set_to_none=True)
    module.eval()
    summary = dict(status=status, experiment='fixed_native_r1_plus_trainable_residual',
                   config=asdict(config), initial_selection_nll=initial, best_selection_nll=best_nll,
                   best_update=best_update, latest_selection_nll=latest_nll, selection_improved=best_nll < initial,
                   practical_gain_established=False, heldout_gain_established=False, successful_updates=successful, train_contexts=len(h1), effective_batch_contexts=min(len(h1), config.batch_contexts),
                   trainable_parameters=sum(p.numel() for p in module.parameters()),
                   selection_gate_passed=best_nll <= initial, elapsed_seconds=time.perf_counter()-start,
                   arithmetic='FP32 adapter; cast correction to native FP32/BF16 before residual; frozen differentiable norm/head',
                   inference_control='fixed-on correction or explicit bypass/STOP; no learned STOP',
                   limits=['selection is not held-out test', 'added capacity is not evidence of loop benefit',
                           'parameter-matched one-pass adapter control not implemented', 'no second expert call'],
                   binding_sha256=_binding(identities))
    _atomic_write(output/'metrics.json', lambda p: p.write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n'))
    _atomic_write(output/'summary.json', lambda p: p.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n'))
    return module, summary
