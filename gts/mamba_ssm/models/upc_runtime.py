"""A neural typed-effect interpreter around exact array/register primitives.

Syntax, live-register allocation, constants, slice metadata, budgets and physics
are exact. A frozen GTS3 model predicts each operation and its register bindings.
No teacher correction or exact-decoder fallback is used in neural mode.
There is no game-ID input. The first training curriculum is instruction decoding,
not planning or autonomous goal interpretation. Model failures must stay visible.
"""
from dataclasses import dataclass
import math
from typing import Any
import numpy as np
import torch
from .gts_upc import OPS
from .upc_frontend import Program,ProgramError,BudgetExceeded,compile_policy
ALIASES={'move_exact':'move','div_exact':'div','mod_exact':'mod','branch_false':'branch'}
ARITY={op:2 for op in ('add','sub','mul','min','max','lt','gt','div','mod','and','or','advance','le','ge','eq','ne')}
ARITY.update({op:1 for op in ('copy','abs','neg','sum','amax','amin','cumsum','slice','move','emit','branch','not')})
ARITY.update(load=0,const=0,jump=0,where=3)
@dataclass
class Effect:
    op:str
    dst:int
    args:tuple[int,...]

def features(value):
    if value is None:return np.zeros(4,np.float32)
    x=np.asarray(value)
    if not x.size:return np.array([0,0,0,float(x.dtype.kind=='b')],np.float32)
    return np.array([np.tanh(float(x.flat[0])),np.tanh(float(x.mean())),math.log1p(x.size)/10,float(x.dtype.kind=='b')],np.float32)

class Decoder:
    def __init__(self,model=None,native=None):self.model=model;self.native=native;self.calls=0;self.last_expected=None
    @torch.no_grad()
    def decode(self,op,dst,args,names,values):
        self.calls+=1
        if self.model is None:return Effect(op,dst,tuple(args))
        fields=[dst]+list(args)+[dst]*(3-len(args))
        ins=torch.tensor([[OPS.index(op),*fields]],dtype=torch.long)
        sy=torch.tensor([names],dtype=torch.long)
        num=torch.from_numpy(np.stack([features(v) for v in values])[None])
        z=(self.native or self.model)(ins,sy,num)
        decoded=OPS[int(z['opcode'].argmax(-1)[0])]
        ids=[names[int(j)] for j in z['pointers'].argmax(-1)[0]]
        return Effect(decoded,ids[0],tuple(ids[1:1+ARITY[decoded]]))

class Runtime:
    def __init__(self,decoder,max_instructions=20000,max_elements=1_000_000,seed=42):
        self.decoder=decoder;self.max_instructions=max_instructions;self.max_elements=max_elements
        self.seed=seed;self.last_stats={};self.last_effects=[]
    def scores(self,program:Program,state:dict[str,Any]):
        if program.registers+4>256:raise ProgramError('Initial GTS3 symbol vocabulary is bounded at 256')
        missing=set(program.inputs)-set(state)
        if missing:raise ProgramError(f'Missing state fields: {sorted(missing)}')
        for key,value in state.items():
            arr=np.asarray(value)
            if arr.dtype.kind not in 'biuf' or arr.size>self.max_elements or not np.isfinite(arr).all():
                raise BudgetExceeded(f'Invalid or oversized input: {key}')
        rng=np.random.default_rng(self.seed)
        # Arbitrary symbolic names, not fixed physical register IDs.
        symbols=rng.permutation(256);to_symbol={r:int(symbols[r]) for r in range(program.registers)}
        from_symbol={s:r for r,s in to_symbol.items()}
        dummy=int(symbols[program.registers]);spares=[int(x) for x in symbols[program.registers+1:]]
        regs={};last={r:-1 for r in range(program.registers)}
        for t,i in enumerate(program.code):
            for r in i.args:last[r]=t
        control=any(i.op in ('jump','branch_false') for i in program.code)
        steps=0;pc=0;peak=0;calls=self.decoder.calls;self.last_effects=[]
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            while pc<len(program.code):
                if steps>=self.max_instructions:raise BudgetExceeded('Instruction budget exhausted')
                if pc<0:raise ProgramError('Invalid program counter')
                oldpc=pc;i=program.code[pc];pc+=1;steps+=1
                op=ALIASES.get(i.op,i.op)
                if op not in OPS:raise ProgramError(f'Unsupported instruction {op}')
                dst=to_symbol[i.dst] if i.dst is not None else dummy
                args=[to_symbol[r] for r in i.args]
                # Addressable working set: every live register, plus writable dst.
                names=[to_symbol[r] for r in regs]
                if dst not in names:names.append(dst)
                for s in spares:
                    if len(names)>=4:break
                    if s not in names:names.append(s)
                rng.shuffle(names);peak=max(peak,len(names))
                vals=[regs.get(from_symbol.get(s,-1)) for s in names]
                e=self.decoder.decode(op,dst,args,names,vals)
                self.last_effects.append(e.op)
                av=[]
                for s in e.args:
                    r=from_symbol.get(s)
                    if r not in regs:raise ProgramError('Neural effect read an uninitialized register')
                    av.append(regs[r])
                op=e.op
                if op in ('add','sub','mul','min','max','lt','gt','le','ge','eq','ne','div','mod','and','or','where','advance'):
                    shape=np.broadcast_shapes(*(np.shape(v) for v in av))
                    if math.prod(shape)>self.max_elements:raise BudgetExceeded('Broadcast exceeds tensor budget')
                if op=='emit':
                    a=np.asarray(av[0],np.float32)
                    if a.ndim!=1 or not a.size or not np.isfinite(a).all():raise ProgramError('emit requires finite nonempty score vector')
                    self.last_stats=dict(instructions=steps,neural_calls=self.decoder.calls-calls,peak_live_slots=peak)
                    return a
                if op=='jump':
                    if not isinstance(i.meta,int) or not 0<=i.meta<=len(program.code):raise ProgramError('Invalid jump')
                    pc=i.meta;continue
                if op=='branch':
                    if np.size(av[0])!=1 or not isinstance(i.meta,int):raise ProgramError('Invalid branch')
                    if not 0<=i.meta<=len(program.code):raise ProgramError('Invalid branch target')
                    if not bool(np.asarray(av[0]).item()):pc=i.meta
                    continue
                if op=='load':
                    if not isinstance(i.meta,str) or i.meta not in state:raise ProgramError('Invalid load key')
                    v=np.asarray(state[i.meta])
                elif op=='const':v=np.asarray(i.meta,dtype=np.float32)
                elif op in ('copy','move'):v=av[0]
                elif op=='add' or op=='advance':v=np.add(*av)
                elif op=='sub':v=np.subtract(*av)
                elif op=='mul':v=np.multiply(*av)
                elif op=='min':v=np.minimum(*av)
                elif op=='max':v=np.maximum(*av)
                elif op=='abs':v=np.abs(av[0])
                elif op=='neg':v=np.negative(av[0])
                elif op=='slice':v=av[0][i.meta]
                elif op=='sum':v=np.sum(av[0],axis=i.meta)
                elif op=='amax':v=np.max(av[0],axis=i.meta)
                elif op=='amin':v=np.min(av[0],axis=i.meta)
                elif op=='cumsum':v=np.cumsum(av[0],axis=i.meta)
                elif op=='where':v=np.where(*av)
                elif op in ('div','mod'):
                    if np.any(np.asarray(av[1])==0):raise ProgramError('Division/modulus by zero')
                    v=(np.divide if op=='div' else np.remainder)(*av)
                elif op=='and':v=np.logical_and(*av)
                elif op=='or':v=np.logical_or(*av)
                elif op=='not':v=np.logical_not(av[0])
                elif op in ('lt','le','gt','ge','eq','ne'):v={'lt':np.less,'le':np.less_equal,'gt':np.greater,'ge':np.greater_equal,'eq':np.equal,'ne':np.not_equal}[op](*av)
                else:raise ProgramError('Unknown effect')
                arr=np.asarray(v)
                if arr.dtype.kind not in 'biuf' or arr.size>self.max_elements:raise BudgetExceeded('Invalid tensor type/budget')
                if not np.isfinite(arr).all():raise ProgramError('Nonfinite register')
                r=from_symbol.get(e.dst)
                if r is None:raise ProgramError('Neural effect wrote an unallocated destination')
                regs[r]=v
                if not control:
                    for r in list(regs):
                        if last[r]<=oldpc:del regs[r]
        raise ProgramError('Terminated without emit')
    def act(self,program,state,temperature=0.,legal=None,rng=None):
        if not math.isfinite(temperature) or temperature<0:raise ValueError('Invalid temperature')
        s=self.scores(program,state)
        legal=np.ones(s.shape,bool) if legal is None else np.asarray(legal,dtype=bool)
        if legal.shape!=s.shape or not legal.any():raise ProgramError('No legal actions')
        s=np.where(legal,s,-np.inf)
        if temperature==0:return int(s.argmax())
        p=np.exp((s.astype(np.float64)-s.max())/temperature);p/=p.sum()
        return int((rng or np.random.default_rng()).choice(len(s),p=p))
