#!/usr/bin/env python3
"""Evaluate an MLX LoRA adapter on held-out SFT prompts (proxy, not Phase 8 acceptance)."""
import argparse, json, os, random, subprocess, tempfile, sys
REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QAL=os.path.join(REPO,'qal','target','release','qal')

def check(text):
    with tempfile.TemporaryDirectory(prefix='qal-mlx-eval-') as d:
        p=os.path.join(d,'out.qal'); open(p,'w',encoding='utf-8').write(text)
        c=subprocess.run([QAL,'check',p],capture_output=True,text=True,timeout=20)
        if c.returncode!=0: return False,'check'
        t=subprocess.run([QAL,'test',p],capture_output=True,text=True,timeout=30)
        return (t.returncode==0), ('ok' if t.returncode==0 else 'test')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    ap.add_argument('--adapter',required=True)
    ap.add_argument('--data',default=os.path.join(REPO,'bench','corpus','sft.jsonl'))
    ap.add_argument('--limit',type=int,default=10)
    ap.add_argument('--max-tokens',type=int,default=256)
    ap.add_argument('--compare-base',action='store_true')
    args=ap.parse_args()
    from mlx_lm import load, generate
    rows=[json.loads(l) for l in open(args.data,encoding='utf-8') if l.strip()]
    random.Random(1).shuffle(rows); rows=rows[:args.limit]
    base,tok=load(args.model)
    adapter,_=load(args.model,adapter_path=args.adapter)
    def run(model):
        passed=0; out=[]
        for r in rows:
            msgs=r['messages'][:2]
            prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            text=generate(model,tok,prompt=prompt,max_tokens=args.max_tokens,verbose=False)
            ok,why=check(text); passed+=ok
            out.append({'ok':ok,'why':why,'generated':text[:1000]})
        return {'examples':len(rows),'passed':passed,'rate':passed/len(rows) if rows else 0},out
    res={}
    if args.compare_base: res['base'],res['base_results']=run(base)
    res['adapter'],res['adapter_results']=run(adapter)
    print(json.dumps({k:res[k] for k in res if not k.endswith('_results')},indent=2))
    with open(os.path.join(args.adapter,'eval_proxy.json'),'w') as f: json.dump(res,f,indent=2)
    sys.exit(0 if res['adapter']['passed']==len(rows) else 1)

if __name__=='__main__': main()
