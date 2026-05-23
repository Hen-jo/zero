"""
TensorBoard → W&B sync script
별도 서버에서 실행: python tb_to_wandb.py --log_dir ./tensorboard --project dr-zero
"""

import time
import argparse
from pathlib import Path
from collections import defaultdict

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import wandb


def load_events(log_dir: str):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea


def sync_once(log_dir: str, last_step: dict, run: wandb.sdk.wandb_run.Run) -> int:
    ea = load_events(log_dir)
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return 0

    # step → {tag: value} 수집
    step_data: dict = defaultdict(dict)
    for tag in tags:
        for event in ea.Scalars(tag):
            if event.step > last_step.get(tag, -1):
                step_data[event.step][tag] = event.value

    if not step_data:
        return 0

    for step in sorted(step_data):
        run.log(step_data[step], step=step)
        for tag in step_data[step]:
            last_step[tag] = max(last_step.get(tag, -1), step)

    return len(step_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="TensorBoard 로그 디렉토리 경로")
    parser.add_argument("--project", default="dr-zero")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--interval", type=int, default=3600, help="동기화 주기 (초)")
    args = parser.parse_args()

    log_dir = str(Path(args.log_dir).resolve())
    print(f"[tb_to_wandb] log_dir: {log_dir}")
    print(f"[tb_to_wandb] project: {args.project}, interval: {args.interval}s")

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        resume="allow",
    )

    last_step: dict = {}
    try:
        while True:
            n = sync_once(log_dir, last_step, run)
            print(f"[{time.strftime('%H:%M:%S')}] synced {n} new steps → wandb")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[tb_to_wandb] 종료")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
