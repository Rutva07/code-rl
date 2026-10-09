"""Reproduce the v2 system comparison through a real localhost REST server."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from coderl.client import ToolClient
from coderl.data import load_tasks
from coderl.edits import EditPlanner
from coderl.evaluate import evaluate, compare_reports
from coderl.train import supervised_edit_policy, train_edit_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=12501)
    p.add_argument('--trials', type=int, default=11)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--data', default='data/v2')
    p.add_argument('--output', default='outputs/v2')
    p.add_argument('--legacy-checkpoint', default='outputs/verified/grpo/policy.pt')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--confirm', action='store_true', help='fresh repeat of both systems; no retraining or tuning')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = dict(os.environ, CODERL_WORKER_DIR=str((out/'.worker').resolve()))
    log = (out/'api.log').open('a')
    server = subprocess.Popen([sys.executable, '-m', 'coderl.cli', 'serve', '--port', str(port)], env=env, stdout=log, stderr=log)
    tools = ToolClient(url)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('API exited; inspect api.log')
            try:
                health = httpx.get(url+'/health', trust_env=False, timeout=2).json()
                break
            except httpx.HTTPError:
                time.sleep(0.1)
        else:
            raise RuntimeError('API readiness timeout')
        (out/'environment.json').write_text(json.dumps(health, indent=2))
        train = load_tasks(Path(args.data)/'train.jsonl')
        test = load_tasks(Path(args.data)/'test.jsonl')
        if not args.evaluate_only and not args.confirm:
            print('edit SFT', json.dumps(supervised_edit_policy(train, out/'edit_sft')), flush=True)
            print('edit GRPO', json.dumps(train_edit_policy(train, tools, out/'grpo', steps=args.steps,
                                        checkpoint=out/'edit_sft/policy.pt', trials=5)), flush=True)
        suffix = '_confirm' if args.confirm else ''
        def run(name, checkpoint):
            report = evaluate(test, tools, EditPlanner(checkpoint), out/(name+suffix),
                              attempts=args.attempts, trials=args.trials)
            print(name+suffix, json.dumps(report), flush=True)
        run('before', args.legacy_checkpoint)
        if not args.confirm:
            run('sft_only', out/'edit_sft/policy.pt')
        run('after', out/'grpo/policy.pt')
        comparison = compare_reports(out/('before'+suffix), out/('after'+suffix), out/('comparison'+suffix+'.json'))
        print('comparison'+suffix, json.dumps(comparison), flush=True)
        if not args.confirm:
            print('GRPO ablation', json.dumps(compare_reports(out/'sft_only', out/'after', out/'grpo_ablation.json')), flush=True)
    finally:
        tools.close()
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


if __name__ == '__main__':
    main()
