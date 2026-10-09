"""Deterministic real-data subsets. Run before inspecting optimization results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

import pyarrow.parquet as pq
from coderl.data import PAIR_REPO, PAIR_REV, load_tasks
from coderl.schema import Task, TestCase
from coderl.edits import PAUSE

MERGED_URL = 'https://drive.google.com/uc?id=1evBDJapwRvCQK6VUCTV8ZE9WG2k3QJQr'
COLS = ['problem_id', 'src_id', 'tgt_id', 'src_code', 'tgt_code', 'usable_case_ids']


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(archive, root='data'):
    root = Path(root)
    out = root / 'v2'
    out.mkdir(parents=True, exist_ok=True)
    oldtrain = {t.problem_id for t in load_tasks(root/'train.jsonl')}
    validation = {t.problem_id for t in load_tasks(root/'validation.jsonl')}
    testrows, seen = [], set()
    for r in pq.read_table(root/'raw/test.parquet', columns=COLS).to_pylist():
        p = r['problem_id']
        if p not in seen | oldtrain | validation and 20 <= len(r['src_code']) <= 1800 and json.loads(r['usable_case_ids']):
            testrows.append(r)
            seen.add(p)
            if len(testrows) == 24:
                break
    heldout = seen | validation
    rows = pq.read_table(root/'raw/train.parquet', columns=COLS).to_pylist()
    rows = [r for r in rows if r['problem_id'] not in heldout]
    rows.sort(key=lambda r: digest('42:'+r['problem_id']+':'+r['src_id']+':'+r['tgt_id']))
    # Two training strata expose the new transformation on TRAIN problems only.
    # Within each stratum ordering uses a hash, never measured candidate speedup.
    trainrows, seen_train = [], set()
    for pause in [True, False]:
        count = 0
        for r in rows:
            if r['problem_id'] in seen_train or not 20 <= len(r['src_code']) <= 1800:
                continue
            if bool(PAUSE.search(r['src_code'])) != pause or not json.loads(r['usable_case_ids']):
                continue
            trainrows.append(r)
            seen_train.add(r['problem_id'])
            count += 1
            if count == 32:
                break
    # Compact full-program examples for CPU LoRA, deduplicated and capped by problem.
    sftrows, sources, per_problem = [], set(), {}
    for r in rows:
        p, h = r['problem_id'], digest(r['src_code'])
        if h in sources or per_problem.get(p, 0) >= 4:
            continue
        if not (20 <= len(r['src_code']) <= 600 and 20 <= len(r['tgt_code']) <= 600):
            continue
        sftrows.append(r)
        sources.add(h)
        per_problem[p] = per_problem.get(p, 0) + 1
        if len(sftrows) == 512:
            break
    wanted = {r['problem_id'] for r in trainrows + testrows}
    casesroot = root/'merged_tests'
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            parts = Path(member.name).parts
            if not member.isfile() or len(parts) != 3 or parts[1] not in wanted:
                continue
            if not re.fullmatch(r'(input|output)\.\d+\.txt', parts[2]):
                continue
            destination = casesroot/parts[1]/parts[2]
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(tar.extractfile(member).read())
    counts = {}
    for split, selected in [('train', trainrows), ('test', testrows)]:
        tasks = []
        for r in selected:
            ids = sorted(map(int, json.loads(r['usable_case_ids'])))
            if split == 'train':
                # A fixed eight-case sample for training throughput; test evaluation
                # checks EVERY usable merged case, including generated dataset cases.
                ids = sorted(sorted(ids, key=lambda i: digest(f'42:{r["problem_id"]}:{i}'))[:8])
            cases = [TestCase(name=f'input.{i}', stdin=(casesroot/r['problem_id']/f'input.{i}.txt').read_text(),
                              stdout=(casesroot/r['problem_id']/f'output.{i}.txt').read_text()) for i in ids]
            tasks.append(Task(task_id=f'{r["problem_id"]}:{r["src_id"]}', problem_id=r['problem_id'],
                              source=r['src_code'], target=r['tgt_code'], split=split, tests=cases,
                              provenance={'pair_repository': PAIR_REPO, 'pair_revision': PAIR_REV,
                                  'src_id': r['src_id'], 'tgt_id': r['tgt_id'], 'source_sha256': digest(r['src_code']),
                                  'test_kind': 'PIE authors merged CodeNet public + AlphaCode-generated dataset tests',
                                  'case_ids': ids, 'merged_archive_url': MERGED_URL}))
        (out/f'{split}.jsonl').write_text(''.join(t.model_dump_json()+'\n' for t in tasks))
        counts[split] = {'tasks': len(tasks), 'cases': sum(len(t.tests) for t in tasks)}
    (out/'sft.jsonl').write_text(''.join(json.dumps({'task_id':f'{r["problem_id"]}:{r["src_id"]}:{r["tgt_id"]}',
            'problem_id':r['problem_id'], 'split':'train', 'source':r['src_code'], 'target':r['tgt_code'],
            'provenance':{'pair_repository':PAIR_REPO, 'pair_revision':PAIR_REV,
                          'source_sha256':digest(r['src_code']), 'target_sha256':digest(r['tgt_code'])}})+'\n' for r in sftrows))
    protocol = {'seed':42, 'heldout_problem_ids':sorted(seen), 'validation_problem_ids':sorted(validation),
        'prepared':counts, 'sft_pairs':len(sftrows), 'sft_problems':len(per_problem),
        'test_selection':'first 24 unique test problems, source <=1800 chars, excluding legacy train and validation',
        'train_selection':'32 pause-present and 32 pause-absent unique TRAIN problems; source <=1800 chars; hash order',
        'sft_selection':'512 unique TRAIN sources <=600 chars with target <=600 chars; at most four per problem; hash order',
        'correctness':'all metadata usable_case_ids on held-out problems, token-exact expected outputs',
        'evaluation_attempts':3, 'evaluation_trials':11, 'acceptance_threshold':0.05,
        'merged_archive_url':MERGED_URL, 'merged_archive_sha256':hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        'test_cases_include_dataset_generated_inputs':True, 'new_synthetic_cases_added':False,
        'attribution':'end-to-end system revision; improvement is not automatically attributable to GRPO'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True, help='PIE merged archive, tar.gz despite Google Drive filename')
    parser.add_argument('--data', default='data')
    args = parser.parse_args()
    prepare(args.archive, args.data)
