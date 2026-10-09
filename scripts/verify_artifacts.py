"""Verify dataset hashes, split isolation, and sampled rollout accounting."""
import hashlib
import json
from pathlib import Path

from coderl.data import load_tasks


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'data/manifest.json').read_text())
    failures=[]
    for name,expected in manifest['files'].items():
        path=root/'data'/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failures.append(name)
    sets=[{t.problem_id for t in load_tasks(root/'data'/f'{s}.jsonl')} for s in ['train','validation','test']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'split overlap'
    summary=json.loads((root/'outputs/verified/grpo/summary.json').read_text())
    count=0
    independent=0
    with (root/'outputs/verified/grpo/trajectories.jsonl').open() as f:
        for line in f:
            row=json.loads(line)
            assert row['trajectory_id']==count and row['split']=='train'
            independent+=not row['measurement_reused']
            count+=1
    assert count==summary['sampled_trajectories']==summary['steps']*summary['group_size']
    assert independent==summary['distinct_measured_candidates']
    assert not failures,failures
    v2 = root/'data/v2/manifest.json'
    extra = 0
    if v2.exists():
        for name, expected in json.loads(v2.read_text())['files'].items():
            path = root/'data'/name
            assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
            extra += 1
        from coderl.data import load_sft_pairs
        train = load_tasks(root/'data/v2/train.jsonl')
        test = load_tasks(root/'data/v2/test.jsonl')
        sft = load_sft_pairs(root/'data/v2/sft.jsonl')
        assert not ({t.problem_id for t in train+sft} & {t.problem_id for t in test})
        summary2 = json.loads((root/'outputs/v2/grpo/summary.json').read_text())
        count2 = 0
        measured2 = 0
        with (root/'outputs/v2/grpo/trajectories.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                assert row['trajectory_id'] == count2 and row['split'] == 'train'
                count2 += 1
                measured2 += not row['measurement_reused']
        assert count2 == summary2['sampled_trajectories'] == summary2['steps']*summary2['group_size']
        assert measured2 == summary2['distinct_measured_candidates']
    print(json.dumps({'dataset_hashes_verified':len(manifest['files']),
                      'v2_dataset_hashes_verified':extra,
                      'v2_sampled_trajectories':count2 if v2.exists() else None,
                      'v2_independent_candidate_evaluations':measured2 if v2.exists() else None,
                      'split_problem_ids_disjoint':True,'sampled_trajectories':count,
                      'independent_candidate_evaluations':independent},indent=2))


if __name__=='__main__': main()
