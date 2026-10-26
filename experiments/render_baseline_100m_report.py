from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.recovery_common import sha256_file


DECISIONS = {
    "controlled_replication": "进入受控复现；仅在新的独立数据和种子确认后才讨论更强声明",
    "efficiency_only_redesign": "仅保留效率方向并重新设计；不自动扩大，不作质量或 SOTA 声明",
    "stop_or_fundamental_redesign": "停止当前架构的直接扩张，回到机制或训练目标 redesign",
}


def render(validation_path, report_path, status_path):
    validation_path = Path(validation_path)
    data = json.loads(validation_path.read_text())
    arms = data["arms"]
    profiles = data["profiles"]
    decision = data["next_stage_decision"]
    delta = data["loop_minus_dense_nll"]
    speedup = data["loop_speedup_vs_dense"]
    report = f"""# 100M MoE 固定 R1 与参数匹配 dense 对照报告

**日期：2026-10-26**

## 结论

- MoE−dense 的 R1 NLL 差：**{delta:+.9f} nat**（负值有利于 MoE）。
- MoE 相对 dense 的完整 R1 前向速度差：**{speedup:+.2%}**（正值有利于 MoE）。
- 冻结门槛判定：**{decision}**——{DECISIONS[decision]}。

本结果是同数据、同训练预算的内部两组对照，不是外部 SOTA 认证。

## 质量与训练账本

| 指标 | MoE 固定 R1 | 参数匹配 dense |
|---|---:|---:|
| 参数量 | {arms['loop_r1']['parameters']:,} | {arms['dense_parameter']['parameters']:,} |
| 输入 token | {arms['loop_r1']['input_tokens']:,} | {arms['dense_parameter']['input_tokens']:,} |
| 监督 token | {arms['loop_r1']['supervised_tokens']:,} | {arms['dense_parameter']['supervised_tokens']:,} |
| R1 NLL | {arms['loop_r1']['r1_nll']:.9f} | {arms['dense_parameter']['r1_nll']:.9f} |
| 训练秒数 | {arms['loop_r1']['elapsed_seconds']:.2f} | {arms['dense_parameter']['elapsed_seconds']:.2f} |
| GPU 小时 | {arms['loop_r1']['allocated_gpu_hours']:.5f} | {arms['dense_parameter']['allocated_gpu_hours']:.5f} |
| 峰值显存 bytes | {arms['loop_r1']['peak_memory_bytes']:,} | {arms['dense_parameter']['peak_memory_bytes']:,} |

两组均完成 12,207 updates；R1 验证各使用 130,834 个监督 token。packed 实际字节、配置、checkpoint、stream metadata 和 GPU 账本均通过自动核验。

## 完整前向效率

| 指标 | MoE 固定 R1 | 参数匹配 dense |
|---|---:|---:|
| R1 平均 CUDA event ms | {profiles['loop_r1']['mean_event_ms_r1']:.3f} | {profiles['dense_parameter']['mean_event_ms_r1']:.3f} |
| R1 有效 tokens/s | {profiles['loop_r1']['effective_tokens_per_second_r1']:.0f} | {profiles['dense_parameter']['effective_tokens_per_second_r1']:.0f} |
| 相对自身 R2 加速 | {profiles['loop_r1']['speedup_vs_r2']['mean']:.2%} | {profiles['dense_parameter']['speedup_vs_r2']['mean']:.2%} |

两组 profile 均为每深度 1,000 次正式调用，独占 A6000，config/checkpoint/packed 身份与训练组匹配。

## 下一阶段

冻结协议：`baseline_100m/decision_protocol.json`（SHA256 `{data['decision_protocol_sha256']}`）。

- 质量与速度同时占优才允许受控复现。
- 只有效率优势时不自动扩大，也不宣称质量/SOTA 优势。
- 外部 SOTA 仍需要另外的同规模强基线、独立数据和统一协议。

机器核验：`runs/next_2026_10_26/baseline_100m_validation.json`。
"""
    report_path = Path(report_path)
    report_path.write_text(report)
    status_path = Path(status_path)
    status = status_path.read_text()
    marker = "## 100M 强基线对照"
    status = status.split(marker)[0].rstrip() + "\n\n"
    status += marker + "\n\n" + report.split("## 结论\n", 1)[1].split("## 质量与训练账本", 1)[0].strip() + "\n"
    status_path.write_text(status)
    return {"report": str(report_path), "status": str(status_path), "decision": decision, "validation_sha256": sha256_file(validation_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="runs/next_2026_10_26/baseline_100m_validation.json")
    parser.add_argument("--report", default="Loop_MoE_Baseline_100M_Report_2026-10-26.md")
    parser.add_argument("--status", default="Loop_MoE_Current_Status_2026-10-26.md")
    args = parser.parse_args()
    print(json.dumps(render(args.validation, args.report, args.status), indent=2))


if __name__ == "__main__":
    main()
