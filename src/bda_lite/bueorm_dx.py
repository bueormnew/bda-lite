"""Complete BUEORM-DX reference implementation from the project specifications.

This module favors semantic fidelity and streaming correctness. `compile_bueorm`
offers PyTorch graph compilation where available, but a true chunkwise/Triton WY
kernel remains a separate optimization task.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(x, dim=-1, eps=1e-6): return x / (x.norm(dim=dim, keepdim=True) + eps)
def _chrono(n, lo, hi):
    half = torch.exp(torch.linspace(math.log(lo), math.log(hi), n)); decay = torch.pow(torch.tensor(.5), 1 / half).clamp(1e-6, 1 - 1e-6)
    return torch.log(decay / (1 - decay))


@dataclass
class BUEORMCoreConfig:
    d_model: int; n_heads: int = 8; d_k: Optional[int] = None; d_v: Optional[int] = None
    n_anchors: int = 16; beta_max: float = .15; lam: float = .99; spectral_budget: float = 4.
    thermostat_every: int = 256; conv_kernel: int = 4; anchor_gamma: float = 8.; anchor_score_decay: float = .999
    half_life_min: float = 8.; half_life_max: float = 4096.
    def resolved(self):
        c = copy.deepcopy(self); c.d_k = c.d_k or c.d_model // c.n_heads; c.d_v = c.d_v or c.d_model // c.n_heads
        if c.d_model % c.n_heads or c.d_k < 1 or c.d_v < 1 or c.half_life_min <= 0 or c.half_life_max < c.half_life_min: raise ValueError("invalid BUEORM core configuration")
        return c

@dataclass
class LocalAttentionConfig:
    d_model: int; n_heads: int = 8; window: int = 2048; rope_frac: float = .125; rope_base: float = 10000.
    def __post_init__(self):
        if self.d_model % self.n_heads or self.window < 1: raise ValueError("invalid local attention configuration")

@dataclass
class GlobalRelayConfig:
    d_model: int; chunk_size: int = 2048; latent: int = 256; n_queries: int = 4; summary_layers: int = 2; summary_heads: int = 4; recency_window: int = 4; gate_init_bias: float = -3.
    def __post_init__(self):
        if self.latent % self.summary_heads or self.chunk_size < 1 or self.recency_window < 1: raise ValueError("invalid relay configuration")

BlockSpec = Union[str, tuple[str, object]]
@dataclass
class BUEORMDXConfig:
    vocab_size: int; d_model: int = 256; n_macrocycles: int = 1
    block_pattern: tuple[BlockSpec, ...] = ("bueorm", "bueorm", "local", "relay")
    core: Optional[BUEORMCoreConfig] = None; local: Optional[LocalAttentionConfig] = None; relay: Optional[GlobalRelayConfig] = None
    def __post_init__(self):
        self.core = self.core or BUEORMCoreConfig(self.d_model); self.local = self.local or LocalAttentionConfig(self.d_model); self.relay = self.relay or GlobalRelayConfig(self.d_model)
    def resolve_blocks(self):
        defaults = {"bueorm": self.core, "local": self.local, "relay": self.relay}; result=[]
        for macro in range(self.n_macrocycles):
            count={}
            for spec in self.block_pattern:
                kind, cfg = (spec, defaults[spec]) if isinstance(spec, str) else spec
                if kind not in defaults or cfg.d_model != self.d_model: raise ValueError("invalid block pattern")
                idx=count.get(kind,0); count[kind]=idx+1; result.append((f"mc{macro}_{kind}{idx}",kind,cfg))
        return result


class BUEORMCore(nn.Module):
    """Spec-faithful recurrent associative memory, anchors and spectral thermostat."""
    def __init__(self, cfg: BUEORMCoreConfig):
        super().__init__(); self.cfg=cfg.resolved(); c=self.cfg; self.h,self.dk,self.dv,self.s=c.n_heads,c.d_k,c.d_v,c.n_anchors
        hd,vd=self.h*self.dk,self.h*self.dv
        self.q,self.k,self.v=nn.Linear(c.d_model,hd,False),nn.Linear(c.d_model,hd,False),nn.Linear(c.d_model,vd,False)
        self.cq,self.ck,self.cv=(nn.Conv1d(n,n,c.conv_kernel,groups=n,padding=c.conv_kernel-1) for n in (hd,hd,vd))
        self.decay,self.write,self.utility,self.out_gate=nn.Linear(c.d_model,hd),nn.Linear(c.d_model,vd),nn.Linear(c.d_model,self.h),nn.Linear(c.d_model,vd)
        self.tau,self.log_ts=nn.Parameter(torch.zeros(self.h)),nn.Parameter(torch.zeros(self.h)); self.out=nn.Linear(vd,c.d_model,False)
        for layer in (self.write,self.utility,self.out_gate): nn.init.normal_(layer.weight,std=.02); nn.init.zeros_(layer.bias)
        nn.init.normal_(self.decay.weight,std=.02); self.decay.bias.data.copy_(_chrono(hd,c.half_life_min,c.half_life_max))
    def init_state(self,b,device,dtype):
        tail=self.cfg.conv_kernel-1
        return {"M":torch.zeros(b,self.h,self.dk,self.dv,device=device,dtype=dtype),"energy":torch.zeros(b,self.h,self.dk,device=device,dtype=dtype),"A_k":torch.zeros(b,self.h,self.s,self.dk,device=device,dtype=dtype),"A_v":torch.zeros(b,self.h,self.s,self.dv,device=device,dtype=dtype),"usage":torch.zeros(b,self.h,self.s,device=device,dtype=dtype),"pi":_norm(torch.randn(b,self.h,self.dv,device=device,dtype=dtype)),"conv_q":torch.empty(b,0,self.h*self.dk,device=device,dtype=dtype),"conv_k":torch.empty(b,0,self.h*self.dk,device=device,dtype=dtype),"conv_v":torch.empty(b,0,self.h*self.dv,device=device,dtype=dtype),"step":0}
    def _conv(self,m,x,tail):
        # Exact streaming causal convolution: cache raw projected channels, then
        # use only left padding. This is equivalent to a single unsegmented call.
        joined=torch.cat((tail,x),1); weight,bias=m.weight,m.bias; y=F.conv1d(F.pad(joined.transpose(1,2),(m.kernel_size[0]-1,0)),weight,bias,groups=m.groups).transpose(1,2)
        return y[:,-x.shape[1]:],joined[:,-(m.kernel_size[0]-1):]
    def forward(self,x,state=None):
        b,t,_=x.shape; state=self.init_state(b,x.device,x.dtype) if state is None else state; h,dk,dv=self.h,self.dk,self.dv
        q_raw,k_raw,v_raw=self.q(x),self.k(x),self.v(x); cq,tail_q=self._conv(self.cq,q_raw,state["conv_q"]); ck,tail_k=self._conv(self.ck,k_raw,state["conv_k"]); cv,tail_v=self._conv(self.cv,v_raw,state["conv_v"])
        q=_norm(F.silu(cq).view(b,t,h,dk)); k=F.silu(ck).view(b,t,h,dk); v=_norm(F.silu(cv).view(b,t,h,dv))*math.sqrt(dv)
        decay=torch.sigmoid(self.decay(x)).view(b,t,h,dk); write=torch.sigmoid(self.write(x)).view(b,t,h,dv); utility=self.utility(x).view(b,t,h); gate=F.silu(self.out_gate(x)).view(b,t,h,dv)
        M,energy,ak,av,usage,pi=state["M"],state["energy"],state["A_k"],state["A_v"],state["usage"],state["pi"]; outs=[]; ts=F.softplus(self.log_ts)+1e-3
        for i in range(t):
            energy=self.cfg.lam*energy+(1-self.cfg.lam)*k[:,i].square(); kh=_norm(k[:,i]/torch.sqrt(energy+1e-6)); err=v[:,i]-torch.einsum("bhkv,bhk->bhv",M,kh)
            surprise=err.square().sum(-1)/(v[:,i].square().sum(-1)+1e-6); u=torch.sigmoid(utility[:,i])*torch.sigmoid((surprise-self.tau)/ts)
            M=decay[:,i].unsqueeze(-1)*M+self.cfg.beta_max*u[:,:,None,None]*torch.einsum("bhk,bhv->bhkv",kh,write[:,i]*err)
            addr=torch.softmax(-self.cfg.anchor_gamma*usage,-1); ww=addr*u.unsqueeze(-1); ak=ak+ww.unsqueeze(-1)*(kh.unsqueeze(2)-ak); av=av+ww.unsqueeze(-1)*(v[:,i].unsqueeze(2)-av); usage=(1-ww)*usage*self.cfg.anchor_score_decay+ww*u.unsqueeze(-1)
            step=state["step"]+i+1
            if step % self.cfg.thermostat_every == 0:
                with torch.no_grad(): uu=_norm(torch.einsum("bhkv,bhv->bhk",M,pi)); vv=torch.einsum("bhkv,bhk->bhv",M,uu); sigma=vv.norm(dim=-1); pi=_norm(vv); scale=(self.cfg.spectral_budget/(sigma+1e-6)).clamp(max=1)
                M=M*scale[:,:,None,None]
            anchor=torch.softmax(torch.einsum("bhk,bhsk->bhs",q[:,i],ak)/math.sqrt(dk),-1); outs.append((torch.einsum("bhkv,bhk->bhv",M,q[:,i])+torch.einsum("bhs,bhsv->bhv",anchor,av))*gate[:,i])
        return self.out(torch.stack(outs,1).reshape(b,t,h*dv)),{"M":M,"energy":energy,"A_k":ak,"A_v":av,"usage":usage,"pi":pi,"conv_q":tail_q,"conv_k":tail_k,"conv_v":tail_v,"step":state["step"]+t}


def _rope(x, frac, base, positions=None):
    b,t,h,d=x.shape; r=max(2,int(d*frac)//2*2); freq=1/(base**(torch.arange(0,r,2,device=x.device,dtype=x.dtype)/r)); positions=torch.arange(t,device=x.device,dtype=x.dtype) if positions is None else positions.to(dtype=x.dtype,device=x.device); a=positions[:,None]*freq
    co,si=torch.cos(a).repeat_interleave(2,-1)[None,:,None],torch.sin(a).repeat_interleave(2,-1)[None,:,None]; z=x[...,:r]; rot=torch.stack((-z[...,1::2],z[...,::2]),-1).flatten(-2)
    return torch.cat((z*co+rot*si,x[...,r:]),-1)


class LocalWindowAttention(nn.Module):
    def __init__(self,cfg: LocalAttentionConfig):
        super().__init__(); self.cfg=cfg; self.h=cfg.n_heads; self.d=cfg.d_model//cfg.n_heads; self.q,self.k,self.v,self.o=(nn.Linear(cfg.d_model,cfg.d_model,False) for _ in range(4))
    def forward(self,x,state=None):
        b,t,d=x.shape; state=state or {"k":x.new_empty(b,self.h,0,self.d),"v":x.new_empty(b,self.h,0,self.d),"position":0}; pos=state["position"]; positions=torch.arange(pos,pos+t,device=x.device)
        q=_rope(self.q(x).view(b,t,self.h,self.d),self.cfg.rope_frac,self.cfg.rope_base,positions).transpose(1,2); fresh_k=_rope(self.k(x).view(b,t,self.h,self.d),self.cfg.rope_frac,self.cfg.rope_base,positions).transpose(1,2); fresh_v=self.v(x).view(b,t,self.h,self.d).transpose(1,2)
        k,v=torch.cat((state["k"],fresh_k),2),torch.cat((state["v"],fresh_v),2); key_pos=torch.arange(pos-state["k"].shape[2],pos+t,device=x.device); mask=(key_pos[None]<=positions[:,None])&(key_pos[None]>positions[:,None]-self.cfg.window)
        y=self.o(F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(b,t,d)); return y,{"k":k[:,:,-self.cfg.window:],"v":v[:,:,-self.cfg.window:],"position":pos+t}


class _SummaryEncoder(nn.Module):
    def __init__(self,d,latent,queries):
        super().__init__(); self.q=nn.Parameter(torch.randn(queries,d)*.02); self.k,self.v=nn.Linear(d,d,False),nn.Linear(d,d,False); self.mlp=nn.Sequential(nn.Linear(d*(queries+1),2*latent),nn.GELU(),nn.Linear(2*latent,latent))
    def forward(self,x):
        score=torch.einsum("pd,btd->bpt",self.q,self.k(x))/math.sqrt(x.shape[-1]); pooled=torch.einsum("bpt,btd->bpd",score.softmax(-1),self.v(x)); return self.mlp(torch.cat((x.mean(1),pooled.flatten(1)),-1))


class _SummaryStack(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.layers=nn.ModuleList([nn.TransformerEncoderLayer(cfg.latent,cfg.summary_heads,4*cfg.latent,batch_first=True,norm_first=True) for _ in range(cfg.summary_layers)])
    def forward(self,x):
        m=x.shape[1]; mask=torch.triu(torch.ones(m,m,device=x.device,dtype=torch.bool),1)
        for layer in self.layers: x=layer(x,src_mask=mask)
        return x


class GlobalLatentRelay(nn.Module):
    """Causal global relay with true partial-chunk state between streaming calls."""
    def __init__(self,cfg: GlobalRelayConfig):
        super().__init__(); self.cfg=cfg; self.encoder=_SummaryEncoder(cfg.d_model,cfg.latent,cfg.n_queries); self.stack=_SummaryStack(cfg); self.q,self.k,self.v=nn.Linear(cfg.d_model,cfg.latent,False),nn.Linear(cfg.latent,cfg.latent,False),nn.Linear(cfg.latent,cfg.d_model,False); self.gate=nn.Linear(cfg.d_model,cfg.d_model); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,cfg.gate_init_bias)
    def init_state(self,b,device,dtype): return {"buffer":torch.empty(b,0,self.cfg.d_model,device=device,dtype=dtype),"raw_summaries":torch.empty(b,0,self.cfg.latent,device=device,dtype=dtype)}
    def forward(self,x,state=None):
        b,t,d=x.shape; state=self.init_state(b,x.device,x.dtype) if state is None else state; oldbuf,oldraw=state["buffer"],state["raw_summaries"]; c=self.cfg.chunk_size; joined=torch.cat((oldbuf,x),1); full=joined.shape[1]//c; raw_new=torch.stack([self.encoder(joined[:,i*c:(i+1)*c]) for i in range(full)],1) if full else joined.new_empty(b,0,self.cfg.latent); raw=torch.cat((oldraw,raw_new),1); ctx=self.stack(raw) if raw.shape[1] else raw; y=torch.zeros_like(x); q=self.q(x); prior=oldraw.shape[1]; offset=oldbuf.shape[1]
        # Vectorized per chunk: only summaries from strictly earlier completed chunks are injected.
        first=(offset//c); last=((offset+t-1)//c)
        for chunk in range(first,last+1):
            lo_t=max(0,chunk*c-offset); hi_t=min(t,(chunk+1)*c-offset); hi=prior+(chunk-first); lo=max(0,hi-self.cfg.recency_window)
            if hi>lo:
                kk,vv=self.k(ctx[:,lo:hi]),self.v(ctx[:,lo:hi]); p=torch.softmax(torch.einsum("btd,bsd->bts",q[:,lo_t:hi_t],kk)/math.sqrt(self.cfg.latent),-1); y[:,lo_t:hi_t]=torch.einsum("bts,bsd->btd",p,vv)
        return y*torch.sigmoid(self.gate(x)),{"buffer":joined[:,full*c:],"raw_summaries":raw}
    @staticmethod
    @torch.no_grad()
    def sparse_retrieve(query, summaries, k_proj, v_proj, top_k=16):
        keys=k_proj(summaries); score=torch.einsum("bd,bmd->bm",query,keys); _,idx=torch.topk(score,min(top_k,score.shape[1]),-1); gk=torch.gather(keys,1,idx[...,None].expand(-1,-1,keys.shape[-1])); values=v_proj(summaries); gv=torch.gather(values,1,idx[...,None].expand(-1,-1,values.shape[-1])); p=torch.softmax(torch.einsum("bd,bkd->bk",query,gk)/math.sqrt(keys.shape[-1]),-1); return torch.einsum("bk,bkd->bd",p,gv),idx


class _RMS(nn.Module):
    def __init__(self,d): super().__init__(); self.w=nn.Parameter(torch.ones(d))
    def forward(self,x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*self.w
class _MLP(nn.Module):
    def __init__(self,d): super().__init__(); h=int(d*8/3); self.a,self.b,self.c=nn.Linear(d,h,False),nn.Linear(d,h,False),nn.Linear(h,d,False)
    def forward(self,x): return self.c(F.silu(self.a(x))*self.b(x))
class _Residual(nn.Module):
    def __init__(self,d,mixer,kind): super().__init__(); self.n1,self.n2,self.mixer,self.ff,self.kind=_RMS(d),_RMS(d),mixer,_MLP(d),kind
    def forward(self,x,state=None):
        out,state=self.mixer(self.n1(x),state); x=x+out; return x+self.ff(self.n2(x)),state


class BUEORMDX(nn.Module):
    """Spec-complete [BUEORM, BUEORM, local, relay] macrocycle language model."""
    def __init__(self,cfg:BUEORMDXConfig):
        super().__init__(); self.config=cfg; self.embed=nn.Embedding(cfg.vocab_size,cfg.d_model); nn.init.normal_(self.embed.weight,std=.02); self.blocks=nn.ModuleDict(); self.order=[]; builders={"bueorm":BUEORMCore,"local":LocalWindowAttention,"relay":GlobalLatentRelay}
        for name,kind,block_cfg in cfg.resolve_blocks(): self.blocks[name]=_Residual(cfg.d_model,builders[kind](block_cfg),kind); self.order.append((name,kind))
        self.norm,self.head=_RMS(cfg.d_model),nn.Linear(cfg.d_model,cfg.vocab_size,False); self.head.weight=self.embed.weight
    def forward(self,tokens,states=None):
        states={} if states is None else states; x=self.embed(tokens)
        for name,kind in self.order: x,states[name]=self.blocks[name](x,states.get(name))
        return self.head(self.norm(x)),states
    def block_summary(self): return "\n".join(f"{name}: {kind}" for name,kind in self.order)


def compile_bueorm(model, **kwargs):
    """Optional PyTorch compiler path. Fall back to eager mode if compilation fails."""
    if not hasattr(torch,"compile"): return model
    try: return torch.compile(model, dynamic=False, mode="reduce-overhead", **kwargs)
    except Exception: return model
