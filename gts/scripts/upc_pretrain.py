"""Synthetic effect pretraining; no game data or game-specific weight updates.
--resume warm-starts weights, not optimizer/sampler state. --warmup is quantization warmup.
"""
import argparse,json,time,math
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mamba_ssm.models.gts_upc import UPCConfig as Config,GTS3UPC,OPS

def batch(n,slots,symbols,generator,device='cpu'):
    names=torch.rand(n,symbols,generator=generator).argsort(-1)[:,:slots]
    targets=torch.randint(slots,(n,4),generator=generator)
    opcode=torch.randint(len(OPS),(n,),generator=generator)
    ins=torch.cat([opcode[:,None],names.gather(1,targets)],1)
    numeric=torch.randn(n,slots,4,generator=generator).clamp(-3,3)
    return [x.to(device) for x in (ins,names,numeric,opcode,targets)]
@torch.no_grad()
def evaluate(model,n=256,slots=8,seed=9001,device='cpu',batch_size=16):
    g=torch.Generator().manual_seed(seed);tot=op=ptr=exact=0;loss=0.;model.eval()
    for offset in range(0,n,batch_size):
        m=min(batch_size,n-offset);ins,names,num,y,p=batch(m,slots,model.config.symbols,g,device)
        z=model(ins,names,num);a=z['opcode'].argmax(-1)==y;b=z['pointers'].argmax(-1)==p
        op+=a.sum().item();ptr+=b.sum().item();exact+=(a&b.all(-1)).sum().item();tot+=m
        loss+=m*(F.cross_entropy(z['opcode'],y)+F.cross_entropy(z['pointers'].flatten(0,1),p.flatten())).item()
    return dict(n=tot,slots=slots,opcode_accuracy=op/tot,pointer_accuracy=ptr/(4*tot),exact_effect_accuracy=exact/tot,loss=loss/tot)
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1000);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--slots',type=int,default=8);p.add_argument('--threads',type=int,default=2);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=1701);p.add_argument('--out',default='artifacts');p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--device',default='cpu')
    p.add_argument('--mixed-slots',action='store_true');p.add_argument('--resume');p.add_argument('--tiny',action='store_true');p.add_argument('--warmup',type=int,default=0)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(a.seed);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    c=Config()
    if a.tiny:c=Config(d_model=64,n_layer=2,deep_depth=3,bank_trees=8,bank_heads=2)
    model=GTS3UPC.load(a.resume) if a.resume else GTS3UPC(c)
    model.to(a.device);print(json.dumps({'parameters':model.parameters_count(),'config':asdict(model.config),'device':a.device,'threads':a.threads}),flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,betas=(.9,.98),eps=1e-6,weight_decay=.01)
    g=torch.Generator().manual_seed(a.seed+1);curve=[];start=time.perf_counter();examples=0
    baseline=evaluate(model,64,a.slots,device=a.device,batch_size=a.batch);print('INITIAL',json.dumps(baseline),flush=True)
    for step in range(1,a.steps+1):
        model.unfreeze()
        if a.warmup:
            for layer in model.layers:
                for mix in (layer.mixer.bank,layer.mixer.deep):mix.quant_lambda=min(1,step/a.warmup)
        slots=([4,8,16,32][(step-1)%4] if a.mixed_slots else a.slots)
        ins,names,num,y,ptr=batch(a.batch,slots,model.config.symbols,g,a.device)
        opt.zero_grad(set_to_none=True);z=model(ins,names,num)
        loss=F.cross_entropy(z['opcode'],y)+F.cross_entropy(z['pointers'].flatten(0,1),ptr.flatten())
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();examples+=a.batch
        for group in opt.param_groups:group['lr']=a.lr*(.15+.85*.5*(1+math.cos(math.pi*step/a.steps)))
        if step==1 or step%a.eval_every==0 or step==a.steps:
            for layer in model.layers:
                for mix in (layer.mixer.bank,layer.mixer.deep):mix.quant_lambda=1.
            model.freeze();ev=evaluate(model,128,a.slots,device=a.device,batch_size=a.batch)
            row=dict(step=step,train_loss=loss.item(),elapsed=time.perf_counter()-start,examples=examples,**ev);curve.append(row)
            print(json.dumps(row),flush=True);model.save(out/'checkpoint.pt',step=step,seed=a.seed,examples=examples)
            (out/'training.json').write_text(json.dumps({'config':asdict(model.config),'parameters':model.parameters_count(),'baseline':baseline,'curve':curve,'arguments':vars(a)},indent=2))
    model.freeze();results={str(n):evaluate(model,256,n,seed=9100+n,device=a.device,batch_size=a.batch) for n in sorted(set([a.slots,4,8,16,32,64]))}
    (out/'binding_results.json').write_text(json.dumps(results,indent=2));print('FINAL',json.dumps(results),flush=True)
if __name__=='__main__':main()
