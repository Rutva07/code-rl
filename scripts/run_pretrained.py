"""Small real pretrained-model GRPO run and matched held-out generation checks."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import gc

import httpx
import torch
from coderl.client import ToolClient
from coderl.data import load_tasks
from coderl.llm import HFLM, LMPlanner
from coderl.train_lm import train_lm_grpo
from coderl.evaluate import evaluate, compare_reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--sft', default='outputs/v2/qwen_sft/model')
    p.add_argument('--output', default='outputs/v2/pretrained')
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--max-new-tokens', type=int, default=256)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    train = [t for t in load_tasks('data/v2/train.jsonl') if len(t.source) <= 300][:2]
    test = [t for t in load_tasks('data/v2/test.jsonl') if len(t.source) <= 300][:3]
    (out/'protocol.json').write_text(json.dumps({'train_ids':[t.task_id for t in train],
        'test_ids':[t.task_id for t in test], 'selection':'first <=300 character sources in frozen v2 subsets',
        'evaluation_temperature':0, 'attempts':1, 'max_new_tokens':args.max_new_tokens, 'steps':args.steps,
        'base_repository':'Qwen/Qwen2.5-Coder-0.5B-Instruct',
        'base_revision':'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'}, indent=2))
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port = s.getsockname()[1]
    api = f'http://127.0.0.1:{port}'
    log = (out/'api.log').open('w')
    env = dict(os.environ, CODERL_WORKER_DIR=str((out/'.worker').resolve()))
    server = subprocess.Popen([sys.executable,'-m','coderl.cli','serve','--port',str(port)], env=env, stdout=log, stderr=log)
    tools = ToolClient(api)
    try:
        for _ in range(300):
            if server.poll() is not None:
                raise RuntimeError('worker failed')
            try:
                httpx.get(api+'/health',trust_env=False,timeout=2).raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(.1)
        else:
            raise RuntimeError('API startup timeout')
        model = HFLM(args.sft, dtype=torch.bfloat16)
        print('GRPO',json.dumps(train_lm_grpo(model,train,tools,out/'grpo',steps=args.steps,group_size=2,
                        max_new_tokens=args.max_new_tokens,threads=args.threads)),flush=True)
        del model
        gc.collect()
        for name, checkpoint in [('base',args.base),('sft',args.sft),('grpo',str(out/'grpo/model'))]:
            torch.manual_seed(42)
            model = HFLM(checkpoint, dtype=torch.bfloat16).eval()
            print(name,json.dumps(evaluate(test,tools,LMPlanner(model,max_new_tokens=args.max_new_tokens,temperature=0),
                                          out/name,attempts=1,trials=11)),flush=True)
            del model
            gc.collect()
        print('base vs SFT',json.dumps(compare_reports(out/'base',out/'sft',out/'base_sft_comparison.json')),flush=True)
        print('SFT vs GRPO',json.dumps(compare_reports(out/'sft',out/'grpo',out/'sft_grpo_comparison.json')),flush=True)
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
