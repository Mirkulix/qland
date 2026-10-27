#!/usr/bin/env python3
"""Single-shot QAL task evaluation for an MLX adapter (proxy for Phase 8).

For every task in bench/tasks, the model gets the task statement plus the QAL
reference and must return a complete program; the hidden test cases are run with
bench/check.py. This is a single-shot code-generation benchmark, NOT the agent
benchmark (the model does not use tools or iterate).

  python3 bench/lora/eval_tasks_mlx.py --adapter <dir> --limit 23 --compare-base
"""
import argparse, json, os, re, subprocess, sys, tempfile
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO,'bench'))
import check

def strip_code(text):
    text=text.strip()
    m=re.search(r'```(?:qal)?\s*(.*?)```', text, re.S)
    if m: text=m.group(1)
    # drop a leading prose line before the first definition
    lines=text.splitlines()
    for i,l in enumerate(lines):
        if re.match(r'^\s*(fn|type|test)\s+[A-Za-z_]', l):
            return '\n'.join(lines[i:]).strip()+'\n'
    return text.strip()+'\n'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    ap.add_argument('--adapter',required=True)
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--max-tokens',type=int,default=768)
    ap.add_argument('--compare-base',action='store_true')
    ap.add_argument('--check-only',action='store_true',help='only qal check; skip hidden cases (fast)')
    args=ap.parse_args()
    from mlx_lm import load, generate
    ref=open(os.path.join(REPO,'qal','crates','qal-cli','src','reference.md'),encoding='utf-8').read()
    tasks=sorted(check.task_ids())
    if args.limit: tasks=tasks[:args.limit]
    base,tok=load(args.model)
    adapter,_=load(args.model,adapter_path=args.adapter)

    def run(model):
        rows=[]
        for task in tasks:
            statement=open(os.path.join(check.TASKS,task,'task.md'),encoding='utf-8').read()
            prompt=(f"Write one complete QAL program for the task below. It must define main(args: List[Str]) -> Int !io "
                    f"and use only the language in the reference. Return only QAL source.\n\nTASK:\n{statement}\n\nREFERENCE:\n{ref}")
            msgs=[{'role':'system','content':'You write QAL programs.'},{'role':'user','content':prompt}]
            p=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            text=strip_code(generate(model,tok,prompt=p,max_tokens=args.max_tokens,verbose=False))
            with tempfile.NamedTemporaryFile('w',suffix='.qal',delete=False,encoding='utf-8') as f:
                f.write(text); path=f.name
            try:
                if args.check_only:
                    import subprocess as sp
                    c=sp.run([os.path.join(REPO,'qal','target','release','qal'),'check',path],capture_output=True,text=True,timeout=20)
                    res={'task':task,'passed':1 if c.returncode==0 else 0,'total':1,'error':None if c.returncode==0 else 'check'}
                else:
                    res=check.check_qal(task, source=path)
            except Exception as e:
                res={'task':task,'passed':0,'total':0,'error':str(e)}
            finally:
                os.unlink(path)
            rows.append({'task':task,'passed':res.get('passed',0),'total':res.get('total',0),'error':res.get('error'),'generated':text[:500]})
        solved=sum(1 for r in rows if r['total'] and r['passed']==r['total'])
        return {'tasks':len(rows),'solved':solved,'passed_cases':sum(r['passed'] for r in rows),'total_cases':sum(r['total'] for r in rows)},rows
    out={}
    if args.compare_base: out['base'],out['base_rows']=run(base)
    out['adapter'],out['adapter_rows']=run(adapter)
    print(json.dumps({k:out[k] for k in ('base','adapter') if k in out},indent=2))
    with open(os.path.join(args.adapter,'eval_tasks_proxy.json'),'w') as f: json.dump(out,f,indent=2)
    sys.exit(0)

if __name__=='__main__': main()
