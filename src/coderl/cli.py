import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='CodeRL C++ optimizer')
    commands=parser.add_subparsers(dest='command',required=True)
    serve=commands.add_parser('serve')
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8000)
    prep=commands.add_parser('prepare')
    prep.add_argument('--data',default='data')
    for split,n in [('train',24),('validation',6),('test',12)]: prep.add_argument('--'+split,type=int,default=n)
    train=commands.add_parser('train')
    train.add_argument('--steps',type=int,default=12501)
    train.add_argument('--group-size',type=int,default=4)
    train.add_argument('--checkpoint')
    train.add_argument('--trials',type=int,default=3)
    train.add_argument('--seed',type=int,default=42)
    train.add_argument('--output',default='outputs/grpo')
    evaluate=commands.add_parser('evaluate')
    evaluate.add_argument('--checkpoint')
    evaluate.add_argument('--attempts',type=int,default=1)
    evaluate.add_argument('--trials',type=int,default=11)
    evaluate.add_argument('--threshold',type=float,default=0.05)
    evaluate.add_argument('--output',default='outputs/evaluation')
    compare=commands.add_parser('compare-reports')
    compare.add_argument('--before',required=True)
    compare.add_argument('--after',required=True)
    compare.add_argument('--output',default='outputs/comparison.json')
    optimize=commands.add_parser('optimize')
    optimize.add_argument('--task-id',required=True)
    optimize.add_argument('--checkpoint')
    optimize.add_argument('--attempts',type=int,default=3)
    optimize.add_argument('--trials',type=int,default=11)
    optimize.add_argument('--output',default='outputs/optimized.json')
    optimize.add_argument('--model')
    optimize.add_argument('--hybrid',action='store_true',help='Use the code model as a short-source fallback to the edit policy')
    for name in ['sft','train-lm']:
        p=commands.add_parser(name)
        p.add_argument('--model',default='byte')
        p.add_argument('--checkpoint')
        p.add_argument('--device',default='cpu')
        p.add_argument('--lora',action='store_true')
        p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32')
        p.add_argument('--threads',type=int,default=1)
        p.add_argument('--steps',type=int,default=20 if name=='sft' else 2)
        p.add_argument('--group-size',type=int,default=4)
        p.add_argument('--max-new-tokens',type=int,default=128)
        p.add_argument('--max-prompt-tokens',type=int,default=8192)
        p.add_argument('--output',default='outputs/'+name)
    for name in ['train','evaluate','optimize','sft','train-lm']:
        p=commands.choices[name]
        p.add_argument('--tasks',default='data/test.jsonl' if name in ['evaluate','optimize'] else 'data/train.jsonl')
        p.add_argument('--api',default='http://127.0.0.1:8000')
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        uvicorn.run('coderl.api:create_app',factory=True,host=args.host,port=args.port)
        return
    if args.command=='prepare':
        from .data import prepare
        result=prepare(args.data,args.train,args.validation,args.test)
    elif args.command=='compare-reports':
        from .evaluate import compare_reports
        result=compare_reports(args.before,args.after,args.output)
    else:
        from .data import load_tasks, load_sft_pairs
        from .client import ToolClient
        tasks=load_sft_pairs(args.tasks) if args.command=='sft' else load_tasks(args.tasks)
        tools=ToolClient(args.api)
        try:
            if args.command=='train':
                from .train import train_edit_policy
                result=train_edit_policy(tasks,tools,args.output,args.steps,args.group_size,args.seed,trials=args.trials,checkpoint=args.checkpoint)
            elif args.command=='evaluate':
                from .edits import EditPlanner
                from .evaluate import evaluate
                result=evaluate(tasks,tools,EditPlanner(args.checkpoint),args.output,args.attempts,args.trials,args.threshold)
            elif args.command=='optimize':
                from .agent import build_agent
                from .edits import EditPlanner
                planner=EditPlanner(args.checkpoint)
                if args.model:
                    from .llm import HFLM, LMPlanner, HybridPlanner
                    import torch
                    torch.set_num_threads(4)
                    lm=LMPlanner(HFLM(args.model,dtype=torch.bfloat16),max_new_tokens=256 if args.hybrid else 1024,temperature=0 if args.hybrid else 1)
                    planner=HybridPlanner(planner,lm) if args.hybrid else lm
                elif args.hybrid:
                    raise ValueError('--hybrid requires --model')
                task=next(t for t in tasks if t.task_id==args.task_id)
                result=build_agent(tools,planner,args.attempts,args.trials).invoke({'task':task.model_dump(exclude={'target'})},{'recursion_limit':6*args.attempts+5})
                Path(args.output).parent.mkdir(parents=True,exist_ok=True)
                Path(args.output).write_text(json.dumps(result,indent=2))
            else:
                from .llm import ByteLM,HFLM
                from .train_lm import sft,train_lm_grpo
                import torch
                torch.manual_seed(42)
                torch.set_num_threads(args.threads)
                model=ByteLM.load(args.checkpoint) if args.model=='byte' and args.checkpoint else ByteLM() if args.model=='byte' else HFLM(args.checkpoint or args.model,args.device,args.lora,dtype=getattr(torch,args.dtype))
                if args.command=='sft':
                    result=sft(model,tasks,args.output,args.steps,max_tokens=args.max_prompt_tokens,threads=args.threads)
                else:
                    result=train_lm_grpo(model,tasks,tools,args.output,args.steps,args.group_size,args.max_new_tokens,args.max_prompt_tokens,threads=args.threads)
        finally:
            tools.close()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
