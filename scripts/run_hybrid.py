"""Evaluate the bounded GRPO + pretrained SFT fallback on the complete v2 split."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import torch
from coderl.client import ToolClient
from coderl.data import load_tasks
from coderl.edits import EditPlanner
from coderl.llm import HFLM, LMPlanner, HybridPlanner
from coderl.evaluate import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='outputs/v2/qwen_sft/model')
    p.add_argument('--checkpoint', default='outputs/v2/grpo/policy.pt')
    p.add_argument('--tasks', default='data/v2/test.jsonl')
    p.add_argument('--output', default='outputs/v2/hybrid')
    p.add_argument('--max-source-chars', type=int, default=300)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--trials', type=int, default=11)
    p.add_argument('--attempts', type=int, default=3)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = HFLM(args.model, dtype=torch.bfloat16).eval()
    planner = HybridPlanner(EditPlanner(args.checkpoint),LMPlanner(model,max_new_tokens=args.max_new_tokens,temperature=0),args.max_source_chars)
    (out/'protocol.json').write_text(json.dumps({'policy':'GRPO bounded edits + pretrained SFT fallback',
        'max_source_chars':args.max_source_chars, 'max_new_tokens':args.max_new_tokens, 'temperature':0,
        'attempts':args.attempts, 'trials':args.trials, 'all_v2_test_programs':True,
        'fallback_rule':'one generation after a bounded proposal or instead of preservation, for source <=300 chars',
        'SFT_checkpoint_selected_after_three_program_pilot':True,
        'confirmation':'fresh timing on frozen outputs; this is development evaluation, not an untouched final benchmark'},indent=2))
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port = s.getsockname()[1]
    api = f'http://127.0.0.1:{port}'
    env = dict(os.environ, CODERL_WORKER_DIR=str((Path('outputs/v2')/'.worker').resolve()))
    log = (out/'api.log').open('w')
    server = subprocess.Popen([sys.executable,'-m','coderl.cli','serve','--port',str(port)],env=env,stdout=log,stderr=log)
    tools = ToolClient(api)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('API failed')
            try:
                health = httpx.get(api+'/health',trust_env=False,timeout=2).json()
                break
            except httpx.HTTPError:
                time.sleep(.1)
        else:
            raise RuntimeError('API startup timeout')
        (out/'environment.json').write_text(json.dumps(health,indent=2))
        print(json.dumps(evaluate(load_tasks(args.tasks),tools,planner,out/'after',attempts=args.attempts,trials=args.trials)),flush=True)
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
