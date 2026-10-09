import copy
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from .grpo import grpo_loss
from .llm import ByteLM, HFLM, prompt_for, extract_code


def completion_log_probs(model, prompt, completion):
    """Gather only response token log-probs: first response uses last prompt logit."""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + completion], device=device)
    if isinstance(model, HFLM):
        logits = model(ids[:, :-1], response_tokens=len(completion))[0]
        start = 0
    else:
        logits = model(ids[:, :-1])[0]
        start = len(prompt)-1
    labels = ids[0, len(prompt):]
    return logits[start:start+len(completion)].float().log_softmax(-1).gather(-1, labels[:,None]).squeeze(-1)


def sft(model, tasks, output, steps=20, max_tokens=8192, seed=42, lr=3e-4, threads=1):
    if not tasks or any(t.split != "train" or not t.target for t in tasks):
        raise ValueError("SFT requires real target programs from the train split")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    eligible = []
    for task in tasks:
        prompt = model.encode_prompt(prompt_for(task.source))
        response = model.encode("Plan: preserve outputs while reducing work.\n```cpp\n"+task.target+"\n```")+[model.eos_id]
        if len(prompt)+len(response) <= max_tokens:
            eligible.append((prompt,response,task.task_id))
    if not eligible:
        raise ValueError("no full source/target examples fit context; increase max_tokens")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    visited = set()
    with (output/'sft.jsonl').open('w') as f:
        for step in range(steps):
            prompt, response, task_id = eligible[step % len(eligible)]
            visited.add(task_id)
            model.train()
            loss = -completion_log_probs(model,prompt,response).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            f.write(json.dumps({'step':step,'task_id':task_id,'loss':float(loss.detach())})+'\n')
            f.flush()
    model.save(output/'model.pt' if isinstance(model,ByteLM) else output/'model')
    report = {'mode':'supervised_warm_start','steps':steps,'eligible_real_train_pairs':len(eligible),
              'model':model.__class__.__name__,'seed':seed, 'unique_pairs_trained':len(visited),
              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
              'dtype':str(next(model.parameters()).dtype), 'threads':threads}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    return report


def train_lm_grpo(model, tasks, tools, output, steps=2, group_size=4, max_new_tokens=128,
                  max_prompt_tokens=8192, seed=42, lr=1e-5, beta=0.01, trials=3, threads=1):
    if not tasks or any(t.split!='train' for t in tasks) or group_size<2:
        raise ValueError("train split and group_size>=2 required")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # A frozen full copy also works with LoRA; ensure device has room for both.
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    model.eval()  # No dropout mismatch between sampling and log-prob evaluation.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=0)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rollouts = 0
    informative = 0
    began = time.time()
    with (output/'trajectories.jsonl').open('w') as f, (output/'training.jsonl').open('w') as log:
        for step in range(steps):
            task = tasks[step % len(tasks)]
            prompt = model.encode_prompt(prompt_for(task.source))
            if len(prompt)>max_prompt_tokens:
                raise ValueError(f"{task.task_id} exceeds prompt budget; no source truncation")
            baseline = tools.compile(task.source)
            if not baseline['ok']:
                raise ValueError("baseline does not compile")
            completions = [model.generate(prompt,max_new_tokens) for _ in range(group_size)]
            if any(not c for c in completions):
                raise ValueError("empty model completion")
            with torch.no_grad():
                old = [completion_log_probs(model,prompt,c).detach() for c in completions]
                ref = [completion_log_probs(reference,prompt,c).detach() for c in completions]
            rewards=[]
            for completion in completions:
                text=model.decode(completion)
                candidate=extract_code(text)
                compiled=tools.compile(candidate) if candidate else {'ok':False,'stderr':'empty source'}
                result=tools.compare(baseline['artifact_id'],compiled['artifact_id'],task.tests,trials) if compiled['ok'] else {'reward':-1.0,'compile_failed':True,'compiler_diagnostics':compiled['stderr']}
                rewards.append(result['reward'])
                f.write(json.dumps({'trajectory_id':rollouts,'step':step,'task_id':task.task_id,'split':'train',
                        'response':text,'response_ids':completion,'feedback':result})+'\n')
                rollouts+=1
            device=next(model.parameters()).device
            r=torch.tensor(rewards,device=device)
            informative+=int(r.std(unbiased=False)>1e-6)
            optimizer.zero_grad()
            # Variable response lengths: pad only loss vectors and explicitly mask.
            length=max(map(len,completions))
            pad=lambda v:torch.nn.functional.pad(v,(0,length-len(v)))
            current=torch.stack([pad(completion_log_probs(model,prompt,c)) for c in completions])
            mask=torch.stack([pad(torch.ones(len(c),device=device)) for c in completions])
            loss,stats=grpo_loss(current,torch.stack([pad(v) for v in old]),torch.stack([pad(v) for v in ref]),r,mask,beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            log.write(json.dumps({'step':step,'loss':float(loss.detach()),'rewards':rewards,**stats})+'\n')
            f.flush()
            log.flush()
    model.save(output/'model.pt' if isinstance(model,ByteLM) else output/'model')
    report={'mode':'token_level_grpo','model':model.__class__.__name__,'steps':steps,
            'sampled_trajectories':rollouts,'informative_groups':informative,'seed':seed,'seconds':time.time()-began,
            'target_code_visible_to_actor':False,'max_new_tokens':max_new_tokens}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    return report
