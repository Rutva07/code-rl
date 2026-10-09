"""One command starts a real HTTP worker, trains and evaluates, then shuts it down."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import socket
import torch

import httpx
from coderl.client import ToolClient
from coderl.data import load_tasks
from coderl.edits import EditPlanner
from coderl.train import train_edit_policy
from coderl.evaluate import evaluate,compare_reports


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=12501)
    p.add_argument('--data',default='data')
    p.add_argument('--output',default='outputs/verified')
    p.add_argument('--trials',type=int,default=11)
    p.add_argument('--include-lm-smoke',action='store_true')
    p.add_argument('--evaluate-only',action='store_true')
    args=p.parse_args()
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port=s.getsockname()[1]
    api=f'http://127.0.0.1:{port}'
    log=(output/'api.log').open('w')
    server=subprocess.Popen([sys.executable,'-m','coderl.cli','serve','--port',str(port)],stdout=log,stderr=log)
    tools=ToolClient(api)
    try:
        for _ in range(200):
            if server.poll() is not None: raise RuntimeError('worker failed to start; inspect api.log')
            try:
                health=httpx.get(api+'/health',timeout=2,trust_env=False).json()
                break
            except httpx.HTTPError: time.sleep(0.1)
        else: raise RuntimeError('worker readiness timeout')
        (output/'environment.json').write_text(json.dumps(health,indent=2))
        train=load_tasks(Path(args.data)/'train.jsonl')
        test=load_tasks(Path(args.data)/'test.jsonl')
        if not args.evaluate_only:
            report=train_edit_policy(train,tools,output/'grpo',steps=args.steps)
            print('training',json.dumps(report),flush=True)
        print('before',json.dumps(evaluate(test,tools,EditPlanner(output/'grpo/initial.pt'),output/'before',trials=args.trials)),flush=True)
        print('after',json.dumps(evaluate(test,tools,EditPlanner(output/'grpo/policy.pt'),output/'after',trials=args.trials)),flush=True)
        print('comparison',json.dumps(compare_reports(output/'before',output/'after',output/'comparison.json')),flush=True)
        if args.include_lm_smoke:
            from coderl.llm import ByteLM
            from coderl.train_lm import sft,train_lm_grpo
            torch.manual_seed(42)
            model=ByteLM()
            print('SFT',json.dumps(sft(model,train,output/'lm_sft',steps=4)),flush=True)
            print('LM GRPO',json.dumps(train_lm_grpo(model,train,tools,output/'lm_grpo',steps=2,max_new_tokens=32)),flush=True)
    finally:
        tools.close()
        server.terminate()
        try: server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        log.close()


if __name__=='__main__': main()
