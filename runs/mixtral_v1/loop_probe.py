from pathlib import Path
import json
import torch

from experiments.mixtral_experiment import (load_local_model, precision_scope, input_device,
    validate_native, code_identity)
from mixtral_lab.protocol import load_config, prepare_data, batch_row, token_ce, fresh_output
from mixtral_lab.adapter import pair_to_action
from mixtral_lab.training import load_actor_checkpoint_with_metadata


@torch.no_grad()
def main():
    config = load_config('local_configs/mixtral_train_grouped_r1_u64.json')
    output = fresh_output('runs/mixtral_v1/loop_r2_probe')
    with precision_scope(config['allow_tf32']):
        model, tokenizer, adapter, model_info = load_local_model(config)
        data, _ = prepare_data(config, tokenizer)
        parity = validate_native(adapter, data['calibration'], config)
        if not parity['eligible']:
            raise RuntimeError('native equivalence failed')
        identity = json.loads(Path('runs/mixtral_v1/train_grouped_r1_u64/report.json').read_text())['identity']
        actor, actor_meta = load_actor_checkpoint_with_metadata(
            'runs/mixtral_v1/train_grouped_r1_u64/router/checkpoint_best.pt', identity, adapter.device)

        def native_then_stop(obs):
            if bool((obs.rounds == 0).all()):
                return pair_to_action(obs.native_pairs)
            return torch.zeros_like(obs.rounds)

        variants = {
            'controlled_original_r1': (None, 1),
            'controlled_original_two_loops': (None, 2),
            'best_actor_r1': (actor, 1),
            'best_actor_two_loops_no_stop_training': (actor, 2),
            'native_first_then_stop': (native_then_stop, 2),
        }
        results = {}
        for name, (policy, calls) in variants.items():
            total, tokens, rounds = 0., 0, []
            for row in data['selection']:
                batch = batch_row(row, input_device(model))
                out = adapter.forward(**batch, policy=policy, max_calls=calls)
                ce = token_ce(out.logits, batch['input_ids'], batch['attention_mask'])
                total += float(ce.double().sum()); tokens += len(ce)
                rounds.append(out.rounds.detach().cpu())
                if not bool(((out.actions >= 0) & (out.actions <= 28)).all()):
                    raise RuntimeError('invalid loop action')
            rounds = torch.cat(rounds)
            results[name] = {'nll': total / tokens, 'tokens': tokens,
                             'mean_rounds': float(rounds.float().mean()),
                             'min_rounds': int(rounds.min()), 'max_rounds': int(rounds.max())}
        report = {'status': 'complete', 'scope': 'bounded R1 actor reused in terminal R2 loop; not R2 training',
                  'newtest_used': False, 'native_equivalence_eligible': True,
                  'actor_checkpoint': actor_meta, 'variants': results,
                  'code': code_identity(), 'device_map': model_info['device_map']}
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report['variants'], indent=2))


if __name__ == '__main__':
    main()
