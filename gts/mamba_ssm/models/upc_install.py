"""Neural installation of explicitly specified programs, then exact execution.
This is neural compilation, not a fresh neural decision on every game tick.
Only deterministic ISA effects are supported; no claim of dynamic planning.
"""
import json
import numpy as np
from .upc_frontend import Program,Instruction,ProgramError
from .upc_runtime import ALIASES,ARITY

def install(program,decoder,seed=177):
    if program.registers+4>256:raise ProgramError('Symbol vocabulary exhausted')
    rng=np.random.default_rng(seed);sy=rng.permutation(256)
    dummy=int(sy[program.registers]);reverse={int(sy[r]):r for r in range(program.registers)}
    spares=[int(x) for x in sy[program.registers+1:]];live=set();last={r:-1 for r in range(program.registers)}
    for t,i in enumerate(program.code):
        for r in i.args:last[r]=t
    control=any(i.op in ('jump','branch_false') for i in program.code)
    if control:live=set(range(program.registers))
    code=[];calls=decoder.calls
    for pc,i in enumerate(program.code):
        op=ALIASES.get(i.op,i.op);dst=int(sy[i.dst]) if i.dst is not None else dummy
        args=[int(sy[r]) for r in i.args];names=[int(sy[r]) for r in sorted(live)]
        if dst not in names:names.append(dst)
        for s in spares:
            if len(names)>=4:break
            if s not in names:names.append(s)
        rng.shuffle(names)
        e=decoder.decode(op,dst,args,names,[None]*len(names))
        if any(s not in reverse for s in e.args):raise ProgramError('Installation decoded invalid read')
        reads=tuple(reverse[s] for s in e.args)
        if not control and any(r not in live for r in reads):raise ProgramError('Installation decoded uninitialized read')
        canon={'move':'move_exact','div':'div_exact','mod':'mod_exact','branch':'branch_false'}.get(e.op,e.op)
        if e.op in ('emit','branch','jump'):write=None
        else:
            if e.dst not in reverse:raise ProgramError('Installation decoded invalid write')
            write=reverse[e.dst];live.add(write)
        code.append(Instruction(canon,write,reads,i.meta))
        if not control:live={r for r in live if last[r]>pc}
    # Cache identity must include model-decoded IR, never only the original text.
    signature=json.dumps([(i.op,i.dst,i.args,repr(i.meta)) for i in code])
    out=Program(code,program.registers,signature,program.inputs)
    return out,dict(neural_calls=decoder.calls-calls,instructions=len(code),seed=seed)
