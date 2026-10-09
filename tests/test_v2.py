"""Regression checks for the benchmark and checkpoint changes."""
import json
from pathlib import Path

import torch
from coderl.edits import EditPolicy, ACTIONS, apply_edit, features
from coderl.data import load_tasks, load_sft_pairs
from coderl.worker import Worker


def test_legacy_checkpoint_preserves_original_logits(tmp_path):
    torch.manual_seed(42)
    legacy = torch.nn.Sequential(torch.nn.Linear(10,64), torch.nn.Tanh(), torch.nn.Linear(64,4))
    torch.save({'actions':ACTIONS[:4], 'state_dict':{'net.'+k:v for k,v in legacy.state_dict().items()}}, tmp_path/'old.pt')
    policy = EditPolicy.load(tmp_path/'old.pt')
    x = features('int main(){system("pause");}')
    assert torch.allclose(legacy(x[:10]), policy(x)[:4])
    assert policy(x)[4:].max() < policy(x)[:4].min()


def test_pause_cleanup_preserves_if_else_structure_and_native_timer(tmp_path):
    source = '#include<cstdlib>\n#include<cstdio>\nint main(){if(true) system("pause"); else puts("wrong"); puts("ok");}'
    worker = Worker(tmp_path/'worker')
    compiled = worker.compile(apply_edit(source, 'remove_pause'))
    assert compiled['ok']
    result = worker.run(compiled['artifact_id'], '')
    assert result['stdout'].split() == ['ok'] and result['returncode'] == 0
    assert result['native_timing'] and 0 < result['wall_ms'] <= result['host_wall_ms']
    assert result['user_cpu_ms'] >= 0 and result['system_cpu_ms'] >= 0
    assert 'system("ls")' in apply_edit('system("ls");', 'remove_pause')


def test_v2_problems_are_disjoint_and_real_cases_have_provenance():
    root = Path(__file__).resolve().parents[1]/'data/v2'
    train, test = load_tasks(root/'train.jsonl'), load_tasks(root/'test.jsonl')
    sft = load_sft_pairs(root/'sft.jsonl')
    assert not ({t.problem_id for t in train+sft} & {t.problem_id for t in test})
    assert len(test) == 24 and sum(len(t.tests) for t in test) == 2379
    assert all(t.provenance['case_ids'] and t.provenance['pair_revision'] for t in train+test)
    protocol = json.loads((root/'protocol.json').read_text())
    assert not protocol['new_synthetic_cases_added']


def test_hybrid_fallback_uses_only_source_budget_and_still_stops():
    from coderl.llm import HybridPlanner
    class Preserve:
        def propose(self, source, history, attempt):
            return {'action':'identity', 'candidate':source, 'plan':'preserve'}
    class Generate:
        def propose(self, source, history, attempt):
            return {'action':'llm_generate', 'candidate':source+'\n', 'plan':'generate'}
    planner = HybridPlanner(Preserve(), Generate(), max_source_chars=30)
    assert planner.propose('int main(){}', [], 0)['action'] == 'llm_generate'
    assert planner.propose('int main(){}', [{'action':'llm_generate'}], 1)['action'] == 'identity'
    assert planner.propose('x'*31, [], 0)['action'] == 'identity'


def test_identical_selected_programs_cannot_show_model_runtime_improvement(tmp_path):
    from coderl.evaluate import compare_reports
    for phase, speed in [('before',1.8),('after',2.1)]:
        directory = tmp_path/phase
        directory.mkdir()
        (directory/'summary.json').write_text(json.dumps({'attempt_budget':1,'trials':11}))
        (directory/'results.jsonl').write_text(json.dumps({'task_id':'p:s','problem_id':'p',
            'baseline_valid':True,'optimized':True,'final':{'speedup':speed}})+'\n')
        (directory/'p_s.cpp').write_text('int main(){}')
    result = compare_reports(tmp_path/'before',tmp_path/'after',tmp_path/'comparison.json')
    assert result['runtime_reduction_vs_initial_percent'] == 0
    assert result['identical_selected_programs'] == 1
