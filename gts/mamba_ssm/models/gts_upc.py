"""GTS3-style typed-effect executor. No English tokenizer or language-model head."""
from dataclasses import dataclass,asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
OPS=('copy','add','sub','mul','min','max','abs','neg','lt','gt','div','mod','and','or',
     'where','sum','amax','amin','cumsum','slice','load','const','move','emit','branch','jump','not','advance','le','ge','eq','ne')
@dataclass
class UPCConfig:
    d_model:int=256
    n_layer:int=9
    bank_trees:int=32
    bank_heads:int=8
    bank_state:int=16
    deep_trees:int=4
    deep_depth:int=8
    d_conv:int=3
    symbols:int=256
    pointer_dim:int=64
    ternary:bool=True
    act_bits:int=8
    ternary_group:int=128
class RMSNorm(nn.Module):
    def __init__(self,d):super().__init__();self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*self.weight
class Block(nn.Module):
    def __init__(self,c,mixed,i):
        super().__init__();self.norm=RMSNorm(c.d_model)
        self.mixer=mixed(c.d_model,bank_trees=c.bank_trees,bank_heads=c.bank_heads,bank_state=c.bank_state,
            deep_trees=c.deep_trees,deep_depth=c.deep_depth,d_conv=c.d_conv,causal=False,
            ternary=c.ternary,ternary_group=c.ternary_group,act_bits=c.act_bits,route_ste=True,layer_idx=i)
    def forward(self,x,mask):return x+self.mixer(self.norm(x),attention_mask=mask)
class GTS3UPC(nn.Module):
    """Instruction [op,dst-symbol,a-symbol,b-symbol,c-symbol] plus a variable-size slot table.
    Predict opcode and four slot pointers. Runtime applies exact ISA semantics.
    Initial curriculum teaches binding/decoding, not autonomous planning.
    """
    def __init__(self,config=None):
        super().__init__();self.config=config or UPCConfig();c=self.config
        from mamba_ssm.modules.gts import GTSMixed
        self.symbol=nn.Embedding(c.symbols,c.d_model);self.opcode=nn.Embedding(len(OPS),c.d_model)
        self.role=nn.Embedding(7,c.d_model);self.numeric=nn.Linear(4,c.d_model,bias=False)
        self.layers=nn.ModuleList([Block(c,GTSMixed,i) for i in range(c.n_layer)])
        self.norm_f=RMSNorm(c.d_model);self.op_head=nn.Linear(c.d_model,len(OPS))
        self.query=nn.ModuleList([nn.Linear(c.d_model,c.pointer_dim,bias=False) for _ in range(4)])
        self.key=nn.Linear(c.d_model,c.pointer_dim,bias=False);self.effect=nn.Linear(c.d_model,c.d_model,bias=False)
        for e in (self.symbol,self.opcode,self.role):nn.init.normal_(e.weight,std=.02)
        nn.init.zeros_(self.numeric.weight)
    def encode_inputs(self,ins,symbols,numeric=None):
        op=self.opcode(ins[:,0])+self.role.weight[0]
        q=self.symbol(ins[:,1:5])+self.role.weight[1:5]
        slots=self.symbol(symbols)+self.role.weight[5]
        if numeric is not None:slots=slots+self.numeric(numeric)
        return torch.cat([op[:,None],q,slots],1)
    def heads(self,h,slot_mask=None):
        h=self.norm_f(h);op=self.op_head(F.silu(self.effect(h[:,0])));keys=self.key(h[:,5:])
        ptr=torch.stack([(keys*self.query[j](h[:,j+1])[:,None]).sum(-1)/math.sqrt(self.config.pointer_dim) for j in range(4)],1)
        if slot_mask is not None:ptr=ptr.masked_fill(~slot_mask[:,None,:],-torch.inf)
        return {'opcode':op,'pointers':ptr}
    def forward(self,ins,symbols,numeric=None,slot_mask=None):
        x=self.encode_inputs(ins,symbols,numeric)
        mask=None if slot_mask is None else torch.cat([torch.ones(ins.shape[0],5,dtype=torch.bool,device=ins.device),slot_mask],1)
        for layer in self.layers:x=layer(x,mask)
        return self.heads(x,slot_mask)
    def freeze(self,sparse=False):
        self.eval()
        for layer in self.layers:
            for m in (layer.mixer.bank,layer.mixer.deep):
                if not getattr(self,"_packed_only",False):m.freeze_quantized()
                if hasattr(m,'sparse_inference'):m.sparse_inference=sparse
        return self
    def unfreeze(self):
        if getattr(self,"_packed_only",False):raise RuntimeError("Packed checkpoints are inference-only")
        self.train()
        for layer in self.layers:
            for m in (layer.mixer.bank,layer.mixer.deep):
                m.unfreeze_quantized()
                if hasattr(m,'sparse_inference'):m.sparse_inference=False
        return self
    def parameters_count(self):return sum(p.numel() for p in self.parameters())
    def save(self,path,**extra):torch.save({'config':asdict(self.config),'model':self.state_dict(),**extra},path)
    @classmethod
    def load(cls,path,backend=None):
        d=torch.load(path,map_location='cpu',weights_only=True);c=d['config']
        c.pop('backend',None)  # standalone experiment metadata; shape/weights are compatible
        m=cls(UPCConfig(**c));m.load_state_dict(d['model']);return m
