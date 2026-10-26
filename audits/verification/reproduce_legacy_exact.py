"""Run the actual legacy train_exact update on a two-leaf analytic fixture.

The repository is unchanged. Its enumerator is replaced only in this process
by two analytic leaves, to isolate the production loss line. Separately the
unaltered neural enumerator is exercised with a tiny random model at two
budgets to test total probability mass. No trained checkpoint is needed.
"""
from pathlib import Path
import importlib
import json
import math
import sys
import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1] / 'GB_RL_MoE_Loop'
sys.path.insert(0, str(ROOT))
from config import ModelConfig, ControllerConfig, ExperimentConfig
from model import LoopMoE
from policy import Controller
from eval.oracle import exact_path_batch
from train.common import CostLedger


class Recorder:
    def __init__(self):
        self.rows = []

    def write(self, **kw):
        self.rows.append(kw)


def main():
    trainer = importlib.import_module('train.exact')
    config = ExperimentConfig()
    config.model.experts_per_layer = 2
    config.model.max_calls = 1
    config.data.min_depth = config.data.max_depth = 1
    config.train.updates = 1
    config.train.batch_size = 2
    config.train.controller_learning_rate = 0.01
    config.train.validation_interval = 10
    config.train.dtype = 'float32'
    model = torch.nn.Linear(1, 1)
    controller = torch.nn.Module()
    controller.register_parameter('z', torch.nn.Parameter(torch.tensor(math.log(4.0))))
    before = float(1 + torch.sigmoid(controller.z.detach()))

    def analytic_paths(model, controller, input_ids, labels, *args):
        n = input_ids.shape[0]
        yield (0,), torch.ones(n), torch.nn.functional.logsigmoid(-controller.z).expand(n)
        yield (1,), torch.full((n,), 2.0), torch.nn.functional.logsigmoid(controller.z).expand(n)

    original_paths, original_evaluate = trainer.exact_path_batch, trainer.evaluate
    trainer.exact_path_batch = analytic_paths
    trainer.evaluate = lambda *a, **k: {'fixture': True}
    rec = Recorder()
    try:
        trainer.train_exact(model, controller, config, torch.device('cpu'), CostLedger(), rec)
    finally:
        trainer.exact_path_batch, trainer.evaluate = original_paths, original_evaluate
    gradient = float(controller.z.grad)
    after = float(1 + torch.sigmoid(controller.z.detach()))
    assert gradient < 0 and after > before

    torch.manual_seed(17)
    mc = ModelConfig(layers=2, hidden_size=16, attention_heads=2, head_dim=8,
                     intermediate_size=16, experts_per_layer=2,
                     adaptive_layers=(1,), max_calls=2, target_only=True)
    tiny_model = LoopMoE(mc).eval()
    tiny_controller = Controller(mc, ControllerConfig(projected_hidden=8, gru_hidden_size=8)).eval()
    inputs = torch.tensor([[1, 5, 3]])
    labels = torch.tensor([[-100, -100, 6]])
    masses = {}
    for budget in (1, 2):
        mass = 0.0
        count = 0
        for path, objective, logp in exact_path_batch(tiny_model, tiny_controller, inputs, labels,
                                                   budget, 0.0, torch.device('cpu'), 'float32'):
            mass += float(logp.detach().exp().sum())
            count += 1
        masses[str(budget)] = {'enumerated_paths': count, 'total_probability': mass}
    assert abs(masses['1']['total_probability']-3.0) < 1e-5
    assert abs(masses['2']['total_probability']-1.0) < 1e-5
    result = {
        'scope': 'Actual legacy train_exact on injected analytic leaves, plus unmodified tiny neural exact_path_batch; no checkpoint or LM claim',
        'torch': torch.__version__, 'implemented_gradient': gradient, 'true_gradient': 0.16,
        'expected_cost_before': before, 'expected_cost_after_actual_adamw_update': after,
        'trainer_records': rec.rows, 'budget_probability_mass': masses,
    }
    Path(__file__).with_name('legacy_exact_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
