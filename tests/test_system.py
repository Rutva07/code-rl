import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from coderl.agent import build_agent
from coderl.api import create_app
from coderl.client import ToolClient
from coderl.edits import EditPlanner, apply_edit, features
from coderl.grpo import grpo_loss,group_advantages
from coderl.llm import ByteLM,extract_code
from coderl.schema import Task,TestCase as Case
from coderl.train import train_edit_policy
from coderl.train_lm import completion_log_probs,train_lm_grpo
from coderl.worker import Worker,output_equal


SOURCE='#include <iostream>\nint main(){int x;std::cin>>x;std::cout<<x*2<<std::endl;}\n'


@pytest.fixture(scope='module')
def worker(tmp_path_factory):
    return Worker(tmp_path_factory.mktemp('worker'))


@pytest.fixture
def tools(worker):
    transport=TestClient(create_app(worker))
    yield ToolClient(transport=transport)
    transport.close()


def test_rest_compiler_correctness_and_failure(tools):
    code=tools.compile(SOURCE)
    assert code['ok']
    assert tools.compile(SOURCE)['cached']
    assert tools.run(code['artifact_id'],'7\n')['stdout'].split()==['14']
    broken=tools.compile('int main(){ invalid }')
    assert not broken['ok'] and broken['stderr']
    wrong=tools.compile(SOURCE.replace('x*2','x*3'))
    result=tools.compare(code['artifact_id'],wrong['artifact_id'],[Case(name='one',stdin='7\n',stdout='14\n')],3)
    assert not result['correct'] and not result['accepted'] and result['reward']==-1


def test_timeouts_and_output_limit(tools):
    infinite=tools.compile('int main(){while(true){}}')
    result=tools.run(infinite['artifact_id'],'',timeout=0.05)
    assert result['timed_out'] or result['returncode']!=0
    spam=tools.compile('#include<cstdio>\nint main(){while(true) puts("1234567890");}')
    result=tools.run(spam['artifact_id'],'',timeout=1)
    assert result['returncode']!=0 and len(result['stdout'])<=4*1024**2


def test_rest_rejects_path_and_unknown_artifact(tools):
    assert tools.http.post('/run',json={'artifact_id':'../../etc/passwd'}).status_code==422
    assert tools.http.post('/run',json={'artifact_id':'a'*64}).status_code==404


def test_edits_compile_and_preserve_outputs(tools):
    edited=apply_edit(SOURCE,'fast_io_avoid_endl')
    assert "'\\n'" in edited
    compiled=tools.compile(edited)
    assert compiled['ok'],compiled
    assert tools.run(compiled['artifact_id'],'0\n')['stdout'].split()==['0']
    mixed=SOURCE.replace('int main()', 'int main()').replace('std::cin>>x;', 'scanf("%d", &x);')
    assert apply_edit(mixed,'fast_io')==mixed


def test_grpo_rewards_increase_good_action_probability():
    logits=torch.nn.Parameter(torch.zeros(2))
    optimizer=torch.optim.SGD([logits],lr=0.1)
    old=torch.log_softmax(logits.detach(),-1)
    initial=logits.softmax(-1)[1].item()
    for _ in range(5):
        loss,_=grpo_loss(logits.log_softmax(-1),old,old,torch.tensor([-1.,1.]),beta=0)
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert logits.softmax(-1)[1].item()>initial
    assert torch.equal(group_advantages(torch.ones(4)),torch.zeros(4))


def test_token_mask_excludes_padding():
    logp=torch.tensor([[-0.7,-0.8,-900.],[-0.6,-0.5,900.]],requires_grad=True)
    old=logp.detach().clone()
    mask=torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    # Zero padding before the exponent, as training does; avoids inf*0.
    selected=torch.where(mask.bool(),logp,torch.zeros_like(logp))
    old=torch.where(mask.bool(),old,torch.zeros_like(old))
    loss,_=grpo_loss(selected,old,old,torch.tensor([0.,1.]),mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.equal(logp.grad[:,2],torch.zeros(2))


class SequencePlanner:
    def __init__(self,candidates): self.candidates=candidates
    def propose(self,source,history,attempt):
        return {'action':'llm_generate','plan':'Use compiler feedback','candidate':self.candidates[attempt]}


def test_langgraph_repairs_compile_failure_and_rejects_wrong_answer(tools):
    task=Task(task_id='fixture',problem_id='fixture',source=SOURCE,
              tests=[Case(name='test',stdin='3\n',stdout='6\n')],split='test')
    candidates=['int main(){syntax_error}',SOURCE.replace('x*2','x*3'),SOURCE]
    state=build_agent(tools,SequencePlanner(candidates),3,3).invoke({'task':task.model_dump()})
    assert len(state['history'])==3
    assert state['history'][0]['feedback']['compile_failed']
    assert state['history'][1]['feedback']['test_failed']
    assert state['best']==SOURCE


def test_real_subset_has_disjoint_problems_and_provenance():
    root=Path(__file__).resolve().parents[1]/'data'
    if not (root/'train.jsonl').exists(): pytest.skip('downloaded data not present')
    from coderl.data import load_tasks
    sets=[]
    for split in ['train','validation','test']:
        tasks=load_tasks(root/(split+'.jsonl'))
        assert all(t.tests and t.target and t.provenance['test_kind']=='real CodeNet public input/output' for t in tasks)
        sets.append({t.problem_id for t in tasks})
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


def test_training_rejects_test_split(tools,tmp_path):
    task=Task(task_id='test',problem_id='test',source=SOURCE,tests=[Case(name='x',stdin='1',stdout='2')],split='test')
    with pytest.raises(ValueError): train_edit_policy([task],tools,tmp_path,steps=1)


def test_action_training_records_real_rollouts_and_invalid_baselines(tools,tmp_path):
    good=Task(task_id='good',problem_id='fixture',source=SOURCE,
              tests=[Case(name='x',stdin='1',stdout='2')],split='train')
    bad=good.model_copy(update={'task_id':'bad','tests':[Case(name='x',stdin='1',stdout='999')]})
    report=train_edit_policy([good,bad],tools,tmp_path,steps=2,group_size=4)
    assert report['sampled_trajectories']==8
    assert report['excluded_invalid_baselines']==1 and report['unique_training_tasks']==1
    rows=[json.loads(s) for s in (tmp_path/'trajectories.jsonl').read_text().splitlines()]
    assert all(r['task_id']=='good' and r['split']=='train' for r in rows)


def test_learned_preserve_action_stops_graph(tools,tmp_path):
    from coderl.edits import EditPolicy
    policy=EditPolicy()
    for p in policy.parameters(): p.data.zero_()
    policy.net[-1].bias.data[0]=10
    policy.save(tmp_path/'policy.pt')
    task=Task(task_id='fixture',problem_id='fixture',source=SOURCE,
              tests=[Case(name='x',stdin='1',stdout='2')],split='test')
    result=build_agent(tools,EditPlanner(tmp_path/'policy.pt'),3,3).invoke({'task':task.model_dump()})
    assert result['attempt']==1 and result['proposal']['action']=='identity'
    assert result['best']==SOURCE


def test_byte_lm_token_scoring_alignment():
    torch.manual_seed(0)
    model=ByteLM(16)
    prompt=[1,2,3];response=[4,5,256]
    scored=completion_log_probs(model,prompt,response)
    ids=torch.tensor([prompt+response])
    logits=model(ids[:,:-1])[0].log_softmax(-1)
    expected=torch.stack([logits[2+i,t] for i,t in enumerate(response)])
    assert torch.allclose(scored,expected)
    assert extract_code('Plan: edit\n```cpp\nint main(){}\n```')=='int main(){}'


def test_evaluation_reverts_candidate_that_loses_gain_on_recheck(tmp_path):
    from coderl.evaluate import evaluate
    import hashlib
    class NoisyTools:
        def __init__(self): self.calls=0
        def compile(self,source): return {'ok':True,'artifact_id':hashlib.sha256(source.encode()).hexdigest()}
        def compare(self,a,b,tests,trials,threshold):
            if a==b: return {'correct':True,'accepted':False,'reward':0.,'speedup':1.}
            self.calls+=1
            return {'correct':True,'accepted':self.calls==1,'reward':0.2 if self.calls==1 else -0.1,
                    'speedup':1.25 if self.calls==1 else 0.9}
    task=Task(task_id='fixture',problem_id='fixture',source=SOURCE,
              tests=[Case(name='x',stdin='1',stdout='2')],split='test')
    report=evaluate([task],NoisyTools(),SequencePlanner([SOURCE+'\n// candidate']),tmp_path,attempts=1,trials=3)
    assert report['correct_optimizations']==0 and report['geometric_mean_speedup_all_valid_programs']==1
    record=json.loads((tmp_path/'results.jsonl').read_text())
    assert record['reverted_after_recheck'] and not record['selected_changed']
    assert next(tmp_path.glob('*.cpp')).read_text()==SOURCE


def test_token_grpo_compiler_reward_update(tools,tmp_path):
    class ControlledByteLM(ByteLM):
        def __init__(self):
            super().__init__(16)
            self.counter=0
        def generate(self,prompt,max_new_tokens=128,temperature=1):
            self.counter+=1
            text=SOURCE if self.counter%2 else 'int main(){invalid}'
            return self.encode(text)+[self.eos_id]
    task=Task(task_id='fixture',problem_id='fixture',source=SOURCE,
              tests=[Case(name='x',stdin='1',stdout='2')],split='train')
    result=train_lm_grpo(ControlledByteLM(),[task],tools,tmp_path,steps=1,group_size=2)
    assert result['sampled_trajectories']==2 and result['informative_groups']==1
    assert (tmp_path/'model.pt').exists()
