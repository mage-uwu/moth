"""Restricted AST parser from the earlier UPC prototype. Never eval/exec input."""
from __future__ import annotations
import ast
from dataclasses import dataclass
import math
from typing import Any
import numpy as np


class ProgramError(ValueError):pass
class BudgetExceeded(RuntimeError):pass

@dataclass
class Instruction:
    op:str
    dst:int | None
    args:tuple=()
    meta:Any=None

@dataclass
class Program:
    code:list[Instruction]
    registers:int
    source:str
    inputs:tuple[str,...]

class Compiler:
    def __init__(self):self.code=[];self.names={};self.nreg=0;self.inputs=set()
    def alloc(self):
        self.nreg+=1
        if self.nreg>512:raise ProgramError('More than 512 register slots')
        return self.nreg-1
    def add(self,op,args=(),meta=None,dst=None):
        if dst is None:dst=self.alloc()
        self.code.append(Instruction(op,dst,tuple(args),meta));return dst
    @staticmethod
    def literal(node):
        if isinstance(node,ast.Constant) and isinstance(node.value,(int,float,str,bool)):
            if isinstance(node.value,(int,float)) and (not math.isfinite(node.value) or abs(node.value)>1e12):raise ProgramError('Literal out of range')
            return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -Compiler.literal(node.operand)
        if isinstance(node,(ast.Tuple,ast.List)):return tuple(Compiler.literal(x) for x in node.elts)
        raise ProgramError('Expected a literal')
    def idx(self,node):
        if isinstance(node,ast.Slice):return slice(None if node.lower is None else self.literal(node.lower),None if node.upper is None else self.literal(node.upper),None if node.step is None else self.literal(node.step))
        if isinstance(node,ast.Tuple):return tuple(self.idx(x) for x in node.elts)
        if isinstance(node,ast.Constant) and node.value is None:return None
        return self.literal(node)
    def expr(self,node):
        if isinstance(node,ast.Constant):
            x=self.literal(node)
            if isinstance(x,str):raise ProgramError('String is valid only as state key')
            return self.add('const',meta=x)
        if isinstance(node,ast.Name):
            if node.id not in self.names:raise ProgramError('Unknown register: '+node.id)
            return self.names[node.id]
        if isinstance(node,ast.BinOp):
            op={ast.Add:'add',ast.Sub:'sub',ast.Mult:'mul',ast.Div:'div_exact',ast.Mod:'mod_exact',ast.BitAnd:'and',ast.BitOr:'or'}.get(type(node.op))
            if op is None:raise ProgramError('Unsupported binary operation')
            return self.add(op,(self.expr(node.left),self.expr(node.right)))
        if isinstance(node,ast.UnaryOp):
            op={ast.USub:'neg',ast.Not:'not',ast.Invert:'not'}.get(type(node.op))
            if op is None:raise ProgramError('Unsupported unary operation')
            return self.add(op,(self.expr(node.operand),))
        if isinstance(node,ast.Compare) and len(node.ops)==1:
            op={ast.Lt:'lt',ast.LtE:'le',ast.Gt:'gt',ast.GtE:'ge',ast.Eq:'eq',ast.NotEq:'ne'}.get(type(node.ops[0]))
            if op is None:raise ProgramError('Unsupported comparison')
            return self.add(op,(self.expr(node.left),self.expr(node.comparators[0])))
        if isinstance(node,ast.Subscript):return self.add('slice',(self.expr(node.value),),self.idx(node.slice))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            name=node.func.id
            if name=='state' and len(node.args)==1:
                key=self.literal(node.args[0])
                if not isinstance(key,str):raise ProgramError('State key must be literal string')
                self.inputs.add(key);return self.add('load',meta=key)
            if name in ('sum','amax','amin','cumsum') and len(node.args)==2:
                return self.add(name,(self.expr(node.args[0]),),self.literal(node.args[1]))
            allowed={'copy':1,'abs':1,'min':2,'max':2,'where':3,'advance':2}
            if name in allowed and len(node.args)==allowed[name]:return self.add(name,tuple(self.expr(x) for x in node.args))
        raise ProgramError('Unsupported expression: '+ast.dump(node)[:140])
    def statements(self,nodes):
        for node in nodes:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                value=self.expr(node.value);name=node.targets[0].id
                if name not in self.names:self.names[name]=self.alloc()
                self.code.append(Instruction('move_exact',self.names[name],(value,)))
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='emit' and len(node.value.args)==1:
                score=self.expr(node.value.args[0]);self.code.append(Instruction('emit',None,(score,)))
            elif isinstance(node,ast.If):
                cond=self.expr(node.test);i=len(self.code);self.code.append(Instruction('branch_false',None,(cond,),None))
                self.statements(node.body);j=len(self.code);self.code.append(Instruction('jump',None,(),None))
                self.code[i].meta=len(self.code);self.statements(node.orelse);self.code[j].meta=len(self.code)
            elif isinstance(node,ast.While) and not node.orelse:
                start=len(self.code);cond=self.expr(node.test);i=len(self.code);self.code.append(Instruction('branch_false',None,(cond,),None))
                self.statements(node.body);self.code.append(Instruction('jump',None,(),start));self.code[i].meta=len(self.code)
            else:raise ProgramError('Unsupported statement: '+type(node).__name__)
    def compile(self,source):
        if len(source)>20000:raise ProgramError('Policy too long')
        try:tree=ast.parse(source)
        except SyntaxError as e:raise ProgramError(str(e)) from e
        self.statements(tree.body)
        if not any(i.op=='emit' for i in self.code):raise ProgramError('Policy must emit a score vector')
        return Program(self.code,self.nreg,source,tuple(sorted(self.inputs)))

def compile_policy(source:str)->Program:return Compiler().compile(source)
