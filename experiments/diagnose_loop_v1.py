"""Recompute the recorded v1 experiment's retrospective oracle, on CPU only.

This uses observed token losses, not a fitted or deployable gate. No model,
corpus download, torch import, GPU, or access to a new test set is needed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

DEFAULT_FIT = Path('runs/mixtral_trainable_loop_v1/') / (
    'trainable_loop_v1_seed17_exportfix_20261030T080210Z/fit')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label, tolerance=1e-10):
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f'{label}: recomputed {a!r} != recorded {b!r}')


def quantile(values, probability):
    ordered = sorted(values)
    if not ordered or not 0 <= probability <= 1:
        raise ValueError('nonempty values and quantile in [0,1] required')
    at = (len(ordered) - 1) * probability
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (at - low) * (ordered[high] - ordered[low])


def validate_record(record):
    """Reject malformed evidence and audit every available scalar/count."""
    count = record.get('tokens')
    if type(count) is not int or count < 1:
        raise ValueError('positive integer token count required')
    for name in ('native_ce', 'candidate_ce', 'valid_position_indices'):
        if len(record.get(name, [])) != count:
            raise ValueError(f'{name} length mismatch')
    positions = record['valid_position_indices']
    if any(type(x) is not int or x < 0 for x in positions) or positions != sorted(set(positions)):
        raise ValueError('valid positions must be nonnegative, unique, increasing integers')
    for name in ('native_ce', 'candidate_ce'):
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in record[name]):
            raise ValueError('finite nonnegative token CE required')
    close(math.fsum(record['native_ce']) / count, record['native_nll'], 'native NLL')
    candidate_nll = record.get('candidate_nll', record.get('nll'))
    close(math.fsum(record['candidate_ce']) / count, candidate_nll, 'candidate NLL')
    if record.get('continue_mask') is not None:
        mask = record['continue_mask']
        if len(mask) != count or any(type(x) is not bool for x in mask):
            raise ValueError('boolean continuation mask length mismatch')
        if 'continued_supervised_tokens' in record and sum(mask) != record['continued_supervised_tokens']:
            raise ValueError('supervised continuation count mismatch')
    return record


def bootstrap_document_ratio(documents, *, seed=17, repetitions=4000):
    """Paired document cluster bootstrap of pooled token-weighted gains.

    Each (token_count, total_gain) pair represents an entire source document.
    This describes these observed documents; it excludes training-seed variance,
    gate fitting, threshold/checkpoint search, and corpus distribution uncertainty.
    """
    if not documents or repetitions < 2 or any(n <= 0 for n, _ in documents):
        raise ValueError('nonempty positive document counts and >=2 repeats required')
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        draw = [documents[rng.randrange(len(documents))] for _ in documents]
        samples.append(math.fsum(gain for _, gain in draw) / sum(n for n, _ in draw))
    return [quantile(samples, .025), quantile(samples, .975)]


def summarize(records, *, seed=17, repetitions=4000):
    if not records:
        raise ValueError('empty evidence group')
    records = [validate_record(record) for record in records]
    ids = [record['document_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate document within canonical evidence group')
    gains = [n-c for r in records for n, c in zip(r['native_ce'], r['candidate_ce'])]
    count = len(gains)
    native = math.fsum(x for r in records for x in r['native_ce']) / count
    full = math.fsum(x for r in records for x in r['candidate_ce']) / count
    positive = [x for x in gains if x > 0]
    negative = [x for x in gains if x < 0]
    oracle_gain = math.fsum(positive) / count
    document_forced = []
    document_oracle = []
    per_document = []
    for r in records:
        local = [n-c for n, c in zip(r['native_ce'], r['candidate_ce'])]
        forced = math.fsum(local)
        oracle = math.fsum(max(0., g) for g in local)
        document_forced.append((len(local), forced))
        document_oracle.append((len(local), oracle))
        per_document.append(dict(document_id=r['document_id'], tokens=len(local),
            forced_gain_nats=forced/len(local), oracle_gain_nats=oracle/len(local),
            oracle_continue_fraction=sum(g > 0 for g in local)/len(local)))
    ranked = sorted(positive, reverse=True)
    budgets = {}
    for fraction in (.1, .25, .5, 1.):
        at_most = int(count*fraction)
        selected = min(at_most, len(ranked))
        gain = math.fsum(ranked[:at_most])/count
        budgets[str(fraction)] = dict(maximum_calls=at_most, oracle_calls=selected,
            oracle_gain_nats=gain, oracle_nll=native-gain)
    return dict(documents=len(records), tokens=count, native_nll=native,
        forced_nll=full, forced_gain_nats=native-full,
        oracle_nll=native-oracle_gain, oracle_gain_nats=oracle_gain,
        oracle_continue_tokens=len(positive), oracle_continue_fraction=len(positive)/count,
        beneficial_fraction=len(positive)/count, harmful_fraction=len(negative)/count,
        gain_over_0_01_fraction=sum(g > .01 for g in gains)/count,
        beneficial_mean_gain=math.fsum(positive)/len(positive) if positive else 0.,
        harmful_mean_loss=-math.fsum(negative)/len(negative) if negative else 0.,
        positive_contribution_nats=math.fsum(positive)/count,
        negative_contribution_nats=math.fsum(negative)/count,
        gain_quantiles={str(q):quantile(gains,q) for q in (0.,.01,.1,.25,.5,.75,.9,.99,1.)},
        oracle_at_most_call_fraction=budgets,
        forced_gain_document_bootstrap_95=bootstrap_document_ratio(
            document_forced,seed=seed,repetitions=repetitions),
        oracle_gain_document_bootstrap_95=bootstrap_document_ratio(
            document_oracle,seed=seed,repetitions=repetitions),
        per_document=per_document)


def verify_same_records(left, right, label):
    """Ignore pointer duplication, never count the same checkpoint twice."""
    if len(left) != len(right):
        raise ValueError(f'{label}: document count mismatch')
    left = {r['document_id']:r for r in left}
    right = {r['document_id']:r for r in right}
    if left.keys() != right.keys():
        raise ValueError(f'{label}: document identity mismatch')
    for name in left:
        for field in ('tokens', 'native_ce', 'candidate_ce', 'valid_position_indices'):
            if left[name][field] != right[name][field]:
                raise ValueError(f'{label}: {field} evidence mismatch')


def diagnose(fit_dir=DEFAULT_FIT, *, seed=17, repetitions=4000):
    fit_dir = Path(fit_dir)
    read = lambda name: json.loads((fit_dir/name).read_text())
    report = read('report.json')
    evidence_path = fit_dir/'token_evidence.jsonl'
    evidence = defaultdict(list)
    with evidence_path.open() as stream:
        for line in stream:
            item = json.loads(line)
            evidence[item['source']].append(validate_record(item['record']))
    descriptor = report['evidence_artifacts']['tokens']
    if digest(evidence_path) != descriptor['sha256']:
        raise ValueError('raw evidence hash differs from recorded report identity')
    if evidence_path.stat().st_size != descriptor['bytes']:
        raise ValueError('raw evidence byte length mismatch')
    if sum(map(len,evidence.values())) != descriptor['records']:
        raise ValueError('raw evidence record count mismatch')
    result = dict(schema='mixtral-loop-v1-retrospective-oracle-v1',
        status='cpu_recomputed_prior_results_only', seed=seed, bootstrap_repetitions=repetitions,
        input_files={}, raw_evidence_records=descriptor['records'],
        evidence_source_counts={k:len(v) for k,v in sorted(evidence.items())},
        branches={}, gate_candidates=[],
        limitations=[
            'Retrospective label-informed oracle; not a fitted, causal, or deployable gate.',
            'Arithmetic STOP/full-R2 mixtures use observed full-shape losses; they do not establish actual compact-execution parity or speed.',
            'Document bootstrap covers observed document variability only, not training seeds, checkpoint/threshold search, or population generalization.',
            'The same development documents were inspected repeatedly; oracle results are exploratory and cannot serve as a new final test.',
            'Online optimizer losses use changing parameters and are not fixed-checkpoint train NLL.',
            'No v2 model training, GPU execution, or new experimental outcome is reported.'])
    expected = {'report/r2_fit/selected_train_probe', 'report/control_fit/selected_train_probe',
                'report/choice/development'}
    for branch, report_key, prefix in (
        ('r2','r2_fit','r2_history'),
        ('parameter_control','control_fit','parameter_control_history')):
        history = read(f'{branch}/history.json')
        optimization = read(f'{branch}/optimization_history.json')
        if [h['epoch'] for h in history] != list(range(9)):
            raise ValueError('expected original fixed checkpoints 0..8')
        summary = report[report_key]
        chosen = summary['best_epoch']
        branch_result = dict(selected_epoch=chosen, updates=len(optimization),
            first_online_batch_ce=optimization[0]['ce'],
            gradients_over_clip_fraction=sum(x['gradient_norm']>1. for x in optimization)/len(optimization),
            fixed_checkpoints=[], online_epoch_means=[])
        for entry in history:
            epoch = entry['epoch']
            row = dict(epoch=epoch, update=entry['update'])
            for role in ('train','development'):
                source = f'{prefix}/{epoch}/{role}'
                expected.add(source)
                verify_same_records(evidence[source], entry[role]['per_document'], source)
                row[role] = summarize(evidence[source],seed=seed,repetitions=repetitions)
                close(row[role]['forced_nll'], entry[role]['nll'], source+' pooled NLL')
                if 'native_nll' in entry[role]:
                    close(row[role]['native_nll'], entry[role]['native_nll'], source+' native NLL')
            branch_result['fixed_checkpoints'].append(row)
        selected_source = f'report/{report_key}/selected_train_probe'
        verify_same_records(evidence[selected_source], evidence[f'{prefix}/{chosen}/train'], selected_source)
        if min(history, key=lambda h:h['development']['nll'])['epoch'] != chosen:
            raise ValueError('selected checkpoint is not recorded development winner')
        for epoch in range(1,9):
            visits = [x for x in optimization if x['epoch']==epoch]
            tokens = sum(len(x['positions']) for x in visits)
            branch_result['online_epoch_means'].append(dict(epoch=epoch, updates=len(visits),
                token_exposures=tokens,
                changing_parameter_ce=math.fsum(x['ce']*len(x['positions']) for x in visits)/tokens))
        result['branches'][branch] = branch_result
    for index, item in enumerate(report['threshold_candidates']):
        source = f'report/threshold_candidates/{index}/development'
        expected.add(source)
        verify_same_records(evidence[source], item['development']['per_document'], source)
        summary = summarize(evidence[source],seed=seed,repetitions=repetitions)
        result['gate_candidates'].append(dict(threshold=item['threshold'],
            nll=summary['forced_nll'], gain_nats=summary['forced_gain_nats'],
            continued_supervised_tokens=item['development']['continued_supervised_tokens'],
            paired_document_bootstrap_95=summary['forced_gain_document_bootstrap_95']))
    if set(evidence) != expected:
        raise ValueError('unknown or missing evidence groups: '+str(set(evidence)^expected))
    verify_same_records(evidence['report/choice/development'],
                        report['choice']['development']['per_document'],'selected gate choice')
    result['gate_choice'] = dict(mode=report['choice']['mode'],threshold=report['choice']['threshold'],
        development_nll=report['choice']['development']['nll'])
    if result['gate_choice']['mode'] == 'stop':
        selected = summarize(evidence['report/choice/development'],seed=seed,repetitions=repetitions)
        close(selected['forced_nll'],selected['native_nll'],'STOP native equality')
    result['canonical_checkpoint_record_count'] = sum(
        len(evidence[f'{prefix}/{epoch}/{role}'])
        for prefix in ('r2_history','parameter_control_history')
        for epoch in range(9) for role in ('train','development'))
    for relative in ('report.json','token_evidence.jsonl','r2/history.json',
                     'r2/optimization_history.json','parameter_control/history.json',
                     'parameter_control/optimization_history.json'):
        file = fit_dir/relative
        result['input_files'][relative] = dict(path=file.as_posix(),bytes=file.stat().st_size,sha256=digest(file))
    return result


def markdown(result):
    r2 = result['branches']['r2']
    chosen = r2['fixed_checkpoints'][r2['selected_epoch']]
    control = result['branches']['parameter_control']
    lines = ['# 上轮真实 R2：逐 token 回算与理想门诊断', '',
        '**这是历史结果的 CPU 回算，不是新实验成绩。** 原始证据 680 条；固定 checkpoint 的独立记录 576 条。其余是选中 checkpoint / 门阈值的重复引用，分别审计、不合并计数。', '',
        '理想门在每个位置读取真实标签后选择 STOP 或 R2，报告 `min(native CE, R2 CE)`。这是已观测 full-shape 损失的回顾性上界；不证明可学习门、压缩执行数值一致或速度收益。', '',
        '| epoch | R2 固定 train NLL | R2 固定 dev NLL | dev 理想门 NLL | dev 理想开启率 |',
        '|---:|---:|---:|---:|---:|']
    for row in r2['fixed_checkpoints']:
        d=row['development']
        lines.append(f"| {row['epoch']} | {row['train']['forced_nll']:.6f} | {d['forced_nll']:.6f} | {d['oracle_nll']:.6f} | {d['oracle_continue_fraction']:.1%} |")
    lines += ['', f"native train/dev NLL 为 {chosen['train']['native_nll']:.6f} / {chosen['development']['native_nll']:.6f}；R2 强制全开开发集最佳是 epoch {r2['selected_epoch']}。一遍适配器最佳是 epoch {control['selected_epoch']}，dev NLL {control['fixed_checkpoints'][control['selected_epoch']]['development']['forced_nll']:.6f}。", '',
        '选中 R2 的收益/损害拆解：', '',
        '| 集合 | 文档/有效位置 | gain>0.01 比例 | 正收益贡献 nat/token | 负收益贡献 nat/token | 理想门收益 95% 文档 bootstrap |',
        '|---|---:|---:|---:|---:|---|']
    for role in ('train','development'):
        row=chosen[role]; lo,hi=row['oracle_gain_document_bootstrap_95']
        lines.append(f"| {role} | {row['documents']} / {row['tokens']} | {row['gain_over_0_01_fraction']:.1%} | {row['positive_contribution_nats']:.6f} | {row['negative_contribution_nats']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', '门阈值候选（相同 epoch 1 R2）：', '',
        '| threshold | dev NLL | 有效位置开启数 | 相对 native 收益 |',
        '|---:|---:|---:|---:|']
    for row in result['gate_candidates']:
        lines.append(f"| {row['threshold']} | {row['nll']:.6f} | {row['continued_supervised_tokens']} | {row['gain_nats']:.6f} |")
    lines += ['', '所有已测阈值都损害开发集，最终选择全 STOP。理想门显示“有局部可用收益”与“门能从前 R2 状态预测收益”是不同问题；旧门使用适配器训练集收益作为标签，无法据此保证来源外泛化。新方案应冻结少量 checkpoint，为 forced-all 和 gated outcome 分别选模，并使用与适配器来源隔离的 gate-fit 文档。', '',
        f"首批在线 R2 CE 为 {r2['first_online_batch_ce']:.6f}；全部更新的梯度 norm 超过 clip=1。epoch 0 固定开发集 NLL {r2['fixed_checkpoints'][0]['development']['forced_nll']:.6f} 明确显示 zero-up 适配器没有给 R2 identity 起点。在线训练 CE 随参数变化，不等同于表中的固定 checkpoint NLL。", '',
        '区间按文档成簇重采样（seed 17，4000 次），使用 token 加权差值；只涵盖这些 24/8 文档的不确定性，不包含训练随机种子、重复开发选择、数据分布或 oracle 不可预测性。8 篇 dev 不足以支撑广泛泛化结论。', '',
        '完整逐 epoch 分布、文档差值、原始路径和 SHA256 见同目录 `loop_v1_oracle.json`。未读取新最终测试，未执行 GPU 训练。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir',type=Path,default=DEFAULT_FIT)
    parser.add_argument('--output',type=Path,default=Path('diagnostics/loop_v1_oracle.json'))
    args=parser.parse_args(argv)
    result=diagnose(args.fit_dir)
    if args.output.exists() or args.output.with_suffix('.md').exists():
        raise FileExistsError('diagnostic output exists; choose a new output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    args.output.with_suffix('.md').write_text(markdown(result))
    print(json.dumps(dict(output=str(args.output),sha256=digest(args.output),status=result['status'])))
    return result


if __name__=='__main__':
    main()
