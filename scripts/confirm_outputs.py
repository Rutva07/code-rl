"""Retest frozen before/after programs, without another search or model tuning."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
from coderl.client import ToolClient
from coderl.data import load_tasks
from coderl.evaluate import summarize, compare_reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='outputs/v2')
    p.add_argument('--before', help='Optional separate directory containing the old system evaluation')
    p.add_argument('--output', default='outputs/v2/confirmation')
    p.add_argument('--tasks', default='data/v2/test.jsonl')
    p.add_argument('--trials', type=int, default=21)
    args = p.parse_args()
    source, out = Path(args.input), Path(args.output)
    selected = {'before':Path(args.before) if args.before else source/'before', 'after':source/'after'}
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.tasks)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0))
        port = s.getsockname()[1]
    api = f'http://127.0.0.1:{port}'
    env = dict(os.environ, CODERL_WORKER_DIR=str((source/'.worker').resolve()))
    log = (out/'api.log').open('w')
    server = subprocess.Popen([sys.executable,'-m','coderl.cli','serve','--port',str(port)], env=env, stdout=log, stderr=log)
    tools = ToolClient(api)
    results = {'before':[], 'after':[]}
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
        initial = {phase:{r['task_id']:r for r in map(json.loads,(selected[phase]/'results.jsonl').read_text().splitlines())} for phase in results}
        for task in tasks:
            baseline = tools.compile(task.source)
            for phase in results:
                directory = out/phase
                directory.mkdir(exist_ok=True)
                filename = f"{task.problem_id}_{task.provenance['src_id']}.cpp"
                previous = initial[phase][task.task_id]
                record = {'task_id':task.task_id,'problem_id':task.problem_id,
                          'baseline_valid':previous['baseline_valid'],'selection_frozen':True}
                if record['baseline_valid']:
                    code = (selected[phase]/filename).read_text()
                    compiled = tools.compile(code)
                    final = tools.compare(baseline['artifact_id'],compiled['artifact_id'],task.tests,args.trials)
                    changed = baseline['artifact_id'] != compiled['artifact_id']
                    if changed and not final.get('accepted',False):
                        record['frozen_candidate_check'] = final
                        record['reverted_after_confirmation'] = True
                        code = task.source
                        changed = False
                        final = tools.compare(baseline['artifact_id'],baseline['artifact_id'],task.tests,args.trials)
                    record.update(final=final,optimized=changed and final['accepted'],selected_changed=changed,
                                  selected_source_sha256=hashlib.sha256(code.encode()).hexdigest())
                    (directory/filename).write_text(code)
                results[phase].append(record)
        for phase, rows in results.items():
            directory = out/phase
            (directory/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            attempts = json.loads((selected[phase]/'summary.json').read_text())['attempt_budget']
            report = summarize(rows,len(tasks),attempts,args.trials,.05)
            report['selection'] = 'frozen outputs of the original three-proposal search; fresh timings, no retuning'
            (directory/'summary.json').write_text(json.dumps(report,indent=2))
            print(phase,json.dumps(report),flush=True)
        print('confirmed comparison',json.dumps(compare_reports(out/'before',out/'after',out/'comparison.json')),flush=True)
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
