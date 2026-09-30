#!/usr/bin/env python3
# sbmoth.py: sbmoth.c's model in PyTorch, for pretraining on a GPU. Its checkpoints are sbmoth.c's ("SBMT"), so a
# model trained here runs on sbmoth.c's CPU inference (-g ckpt -e / -b / -p) unchanged, and a checkpoint from
# sbmoth.c loads here.
#
# The forward pass is sbmoth.c's, step for step: absmean-ternary Monarchs on per-token absmax int8 codes (rms
# norm folded into the scale), the int16 -> int8 requantisation between R and L, static-scale int8 q, k, v and
# u (EMA amax) around ternary short convs and the ternarised implicit long-conv kernels, both directions, the
# int8 head, hash n-gram embeddings, patches from the boundary predictor with thetap, max-pooling, the global
# stack over patches, entropy-gated decoder depth (SK1, SK2) and logits for the masked bytes only. The integer
# parts are exact in fp32 (codes and trits are small integers, their sums stay below 2^24), so the values match
# sbmoth.c's up to float rounding in the norms, the FFTs and the head.
#
# What differs is only how training gets gradients: sbmoth.c runs its backward in int8 codes with stochastic
# rounding (a CPU speed trick); this runs it in fp32 with straight-through estimators through every rounding,
# which is what sbmoth.c's int8 backward approximates. The sparsity is kept as masks (a skipped layer passes its
# input through and adds nothing to the convs; the head runs on the masked bytes only): on a GPU the dense work
# is cheaper than gathering, and the model it trains is the one the C engine then runs sparsely.
#
# Not here (yet): sbmoth.c's -DLCTX chunking. Sequences are TB bytes (LCTX = TB).
#
#   python3 data/prep_bert.py --wiki-mb 900 --books-mb 300 | python3 sbmoth.py train - --M 32 --HV 512 --LG 3 -o run.ck
#   python3 sbmoth.py score run.ck text.txt mask.bin out.bin   (per masked byte: log p, log p either case, argmax)
#   cc -O3 -march=native -fopenmp -DM=32 -DHV=512 -DLG=3 sbmoth.c -o sbmoth -lm && ./sbmoth -g run.ck -e text.txt
import argparse, math, struct, sys, time
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False       # the Monarchs' integer sums are exact in fp32; keep them so
torch.backends.cudnn.allow_tf32 = False

V, NG, FE, FO, PMAX, MASK = 256, 6, 17, 64, 16, 0xFF
PRIMES = [1000000007, 5915587277, 1500450271, 3267000013, 5754853343, 4093082899]
CKMAGIC = 0x544D4253                                 # "SBMT"
LR = 3e-3


def ste(a, b):
    """b's value exactly, a's gradient: a - a.detach() is exactly 0."""
    return b.detach() + (a - a.detach())


def rinv(x):                                         # 1 / rms, as sbmoth.c's rinv
    return 1 / torch.sqrt((x * x).sum(-1) / x.shape[-1] + 1e-5)


def q8t(x, mul):
    """Per-token absmax int8 codes (value exact, straight-through gradient) and their scale m * mul / 127."""
    m = x.detach().abs().amax(-1).clamp_min(1e-8)
    inv = 127 / m
    codes = ste(x * inv[..., None], torch.round(x.detach() * inv[..., None]))
    return codes, m * mul / 127


def tern(w, n=None):
    """Absmean ternary: trits (straight-through to w / s) and the scale s (not differentiated)."""
    s = (w.detach().abs().sum() + 1e-8) / (n or w.numel())
    t = (w.detach() >= 0.5 * s).float() - (w.detach() <= -0.5 * s).float()
    return ste(w / s, t), s


def sq8(x, s):
    """Static-scale int8 fake quant: s * clip(round(x / s)), straight-through."""
    q = torch.clamp(torch.round(x.detach() * (1 / s)), -127, 127) * s
    return ste(x, q)


def tanh_(u):                                        # sbmoth.c's clamped [7/6] Pade tanh
    v = u.clamp(-4.97, 4.97); v2 = v * v
    return v * (135135 + v2 * (17325 + v2 * (378 + v2))) / (135135 + v2 * (62370 + v2 * (3150 + v2 * 28)))


def gelu(x):
    return 0.5 * x * (1 + tanh_(0.7978846 * (x + 0.044715 * x * x * x)))


class Cfg:
    def __init__(self, M=16, TB=128, LE=1, LG=4, LD=2, LH=1, HV=2048, PSZ=4.0, PH=32, SK1=0.3, SK2=0.2, WDECAY=0.1):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.D, self.W3, self.LCTX, self.TG = M * M, M ** 3, TB, TB // 2


class Stack(torch.nn.Module):
    """A stack of Monarch Mixer layers over sequences of T steps (bytes or patches), as sbmoth.c's Stack."""
    MONS = "qkvogud"

    def __init__(self, cfg, T, L, bi, plist, gen):
        super().__init__()
        self.c, self.T, self.L, self.bi = cfg, T, L, bi
        D, M, W3 = cfg.D, cfg.M, cfg.W3
        self.layers = []
        for l in range(L):
            ly = {}
            ro = 1 / math.sqrt(2 * L)
            for nm in self.MONS:                   # R then L of each Monarch, q k v o g u d (sbmoth.c's order)
                sd = ro if nm in "od" else 1.0
                ly[nm + "r"] = plist.new(W3, sd / math.sqrt(M), gen)
                ly[nm + "l"] = plist.new(W3, 1 / math.sqrt(M), gen)
            for nm, n, sd in (("w1", FO * FE, 1 / math.sqrt(FE)), ("b1", FO, 1 / math.sqrt(FE)), ("w2", FO * FO, 1 / math.sqrt(FO)),
                              ("b2", FO, 1 / math.sqrt(FO)), ("w3", D * FO, 1 / math.sqrt(FO)), ("bias", D, 1.0),
                              ("sw", 9 * D, 1 / 3), ("sb", 3 * D, 1 / 3)):
                ly[nm] = plist.new(n, sd, gen)
            if bi:
                ly["w3b"] = plist.new(D * FO, 1 / math.sqrt(FO), gen)
            self.layers.append(ly)
        for l, ly in enumerate(self.layers):
            for k, p in ly.items():
                self.register_parameter(f"l{l}_{k}", p)
        self.register_buffer("amax", torch.zeros(L, 4))          # static int8 scales for q, k, v, u (EMA amax)
        self.register_buffer("gprev", torch.zeros(L, 7, 2))      # sbmoth.c's delayed gradient scales (kept for its checkpoints)
        t = torch.arange(T, dtype=torch.float64); tl = t / (T - 1)
        pz = [tl]
        for f in range(8):
            fr = 1e-4 + f * (7 - 1e-4) / 7
            pz.append(torch.cos(2 * math.pi * fr * t / T))
        for f in range(8):
            fr = 1e-4 + f * (7 - 1e-4) / 7
            pz.append(-torch.sin(2 * math.pi * fr * t / T))
        self.register_buffer("pz", torch.stack(pz, 1).float())                                          # [T][FE]
        lo, hi = math.log(1e-2) / 1.5, math.log(1e-2) / 0.3
        rate = torch.abs(lo + (hi - lo) * torch.arange(D, dtype=torch.float64) / (D - 1))
        self.register_buffer("mod", torch.exp(-tl[:, None] * rate[None, :]).float())                   # [T][D]

    def mon(self, ly, nm, codes, sx):
        """Monarch on int8 codes with scale sx: R (row blocks), int16 -> int8 requantisation, L (column blocks)."""
        M = self.c.M; N = codes.shape[0]
        Rt, rs = tern(ly[nm + "r"]); Lt, ls = tern(ly[nm + "l"])
        z = torch.einsum("nbi,bio->nbo", codes.view(N, M, M), Rt.view(M, M, M))
        mx = z.detach().abs().amax((1, 2)).clamp_min(1)
        qs = 127 / mx
        q = ste(z * qs[:, None, None], torch.round(z.detach() * qs[:, None, None]))
        acc = torch.einsum("nib,oib->nob", q, Lt.view(M, M, M))
        sm = sx * rs * mx / 127
        return (acc * sm[:, None, None] * ls).reshape(N, M * M)

    def filt(self, ly, w3):
        """The implicit long-conv kernel: sine FFN on positional features, decayed, ternarised per channel."""
        D, T = self.c.D, self.T
        a1 = torch.sin(self.pz @ ly["w1"].view(FO, FE).T + ly["b1"])
        a2 = torch.sin(a1 @ ly["w2"].view(FO, FO).T + ly["b2"])
        h = (a2 @ w3.view(D, FO).T) * self.mod
        hs = (h.detach().abs().sum(0) + 1e-8) / T
        t = (h.detach() >= 0.5 * hs).float() - (h.detach() <= -0.5 * hs).float()
        return ste(h, t * hs)

    def conv_params(self, ly):
        D = self.c.D
        sw = ly["sw"].view(3, 3, D)                              # [tap j][stream w][ch]
        s = (sw.detach().abs().sum((0, 2)) + 1e-8) / (3 * D)
        t = (sw.detach() >= 0.5 * s[None, :, None]).float() - (sw.detach() <= -0.5 * s[None, :, None]).float()
        swe = ste(sw, t * s[None, :, None])
        sb = ly["sb"].view(3, D)
        s = (sb.detach().abs().sum(1) + 1e-8) / D
        t = (sb.detach() >= 0.5 * s[:, None]).float() - (sb.detach() <= -0.5 * s[:, None]).float()
        sbe = ste(sb, t * s[:, None])
        bt, bs = tern(ly["bias"])
        return swe, sbe, bt * bs

    def lconv(self, u, h):                                       # causal: y[t] = sum_j h[j] u[t - j], via FFT
        T = self.T
        U = torch.fft.rfft(u, n=2 * T, dim=1); H = torch.fft.rfft(h, n=2 * T, dim=0)
        return torch.fft.irfft(U * H[None], n=2 * T, dim=1)[:, :T]

    def forward(self, X, lens=None, skip=None, train=False):
        """X [N][T][D] -> [N][T][D]. lens [N]: valid steps (the rest padding); skip [N][T]: last layers skipped."""
        N, T, D = X.shape
        tt = torch.arange(T, device=X.device)
        pad = (tt[None, :] >= lens[:, None]) if lens is not None else torch.zeros(N, T, dtype=torch.bool, device=X.device)
        x = X
        for l, ly in enumerate(self.layers):
            sk = (skip >= self.L - l) if skip is not None else torch.zeros_like(pad)
            off = pad | sk                                       # no q, k, v: padding, or a skipped layer
            xf = x.reshape(N * T, D)
            codes, s1 = q8t(xf, rinv(xf))
            pre = [self.mon(ly, nm, codes, s1).view(N, T, D).masked_fill(off[..., None], 0) for nm in "qkv"]
            # mixer: static int8 q, k, v; ternary short convs; u = k * v, static int8; long conv, both directions
            amax = self.amax[l]; sc = torch.where(amax > 0, amax, torch.ones_like(amax)) / 127
            swe, sbe, be = self.conv_params(ly)
            mw = [p.detach().abs().amax() for p in pre]
            pre = [sq8(pre[w], sc[w]) for w in range(3)]
            post = []
            for w in range(3):
                o = sbe[w].expand(N, T, D)
                for j in range(3):                               # tap j reads step t - j + bi
                    sh = j - self.bi
                    if sh > 0: p = torch.nn.functional.pad(pre[w][:, :T - sh], (0, 0, sh, 0))
                    elif sh < 0: p = torch.nn.functional.pad(pre[w][:, -sh:], (0, 0, 0, -sh))
                    else: p = pre[w]
                    o = o + swe[j, w] * p
                post.append(o)
            vb = post[1] * post[2]
            vb = vb.masked_fill(((pad if self.bi else torch.zeros_like(pad)) | sk)[..., None], 0)
            m3 = vb.detach().abs().amax()
            u = sq8(vb, sc[3])
            cc = self.lconv(u, self.filt(ly, ly["w3"]))
            if self.bi:
                cc = cc + self.lconv(u.flip(1), self.filt(ly, ly["w3b"])).flip(1)
            if train:
                with torch.no_grad():
                    m = torch.stack(mw + [m3])
                    self.amax[l] = torch.where(amax > 0, 0.99 * amax + 0.01 * m, m)
            g = (post[0] * (cc + be * u)).reshape(N * T, D)
            # channel mixer: gate -> Mon_o, residual, rms -> Mon_g, Mon_u, GELU GLU -> Mon_d, residual
            cg, sg = q8t(g, 1)
            x1 = xf + self.mon(ly, "o", cg, sg)
            c2, s2 = q8t(x1, rinv(x1))
            gg = gelu(self.mon(ly, "g", c2, s2)) * self.mon(ly, "u", c2, s2)
            c3, s3 = q8t(gg, 1)
            xo = (x1 + self.mon(ly, "d", c3, s3)).view(N, T, D)
            x = torch.where(sk[..., None], x, xo).masked_fill(pad[..., None], 0)
        return x


def head(X, W):
    """rms + logits over an int8 [V][D] matrix (per row absmax) on int8 per-token inputs, as sbmoth.c's head."""
    mw = W.detach().abs().amax(1).clamp_min(1e-8)
    Wd = ste(W, torch.round(W.detach() * (127 / mw)[:, None]) * (mw / 127)[:, None])
    codes, sx = q8t(X, rinv(X))
    return (codes * sx[:, None]) @ Wd.T


class Params:
    """Parameters in sbmoth.c's build order (its checkpoints store them in that order), with their weight decay."""
    def __init__(self, dev):
        self.ps, self.dev = [], dev

    def new(self, n, sd, gen):
        p = torch.nn.Parameter((torch.randn(n, generator=gen, dtype=torch.float32) * sd).to(self.dev))
        p.wd = 0.0
        self.ps.append(p)
        return p


class Model(torch.nn.Module):
    def __init__(self, cfg, dev, seed=1):
        super().__init__()
        self.c = c = cfg
        gen = torch.Generator().manual_seed(seed)
        self.P = P = Params(dev)
        self.Eh = P.new(V * c.D, 0.02, gen)                                   # entropy model (the teacher)
        self.SH = Stack(c, c.TB, c.LH, 0, P, gen)
        self.h1 = len(P.ps)
        self.Eb = P.new(V * c.D, 0.02, gen)                                   # BLT
        self.Hs = [P.new(c.HV * c.D, 0.02, gen) for _ in range(NG)]
        self.Wo = P.new(V * c.D, 0.02, gen)
        bt = len(P.ps)
        self.SE = Stack(c, c.TB, c.LE, 1, P, gen)
        self.SG = Stack(c, c.TG, c.LG, 1, P, gen)
        self.SD = Stack(c, c.TB, c.LD, 1, P, gen)
        for p in P.ps[bt:]: p.wd = c.WDECAY                                   # AdamW on the stacks and the head
        self.Wo.wd = c.WDECAY
        self.Pw1 = P.new(c.PH * c.D, 1 / math.sqrt(c.D), gen); self.Pb1 = P.new(c.PH, 0.0, gen)
        self.Pw2 = P.new(c.PH, 1 / math.sqrt(c.PH), gen); self.Pb2 = P.new(1, 0.0, gen)
        with torch.no_grad(): self.Pb2.fill_(1.5)
        for i, p in enumerate(P.ps): self.register_parameter(f"p{i}", p)
        self.theta, self.thetap, self.thsk = 0.0, 0.0, [-1e30, -1e30]
        self.stacks = [self.SH, self.SE, self.SG, self.SD]
        self.to(dev); self.dev = dev

    # ---- the entropy model (teacher): a small causal moth; its next-byte entropies -----------------------
    def teacher_logits(self, bytes_):                                        # [N][TB] -> [N][TB][V]
        N, T = bytes_.shape
        E = self.Eh.view(V, -1)
        x = self.SH(E[bytes_.long()])
        return head(x.reshape(N * T, -1), E).view(N, T, V)

    @torch.no_grad()
    def teacher_entropy(self, bytes_):                                       # ent[b][t + 1]: entropy of byte t + 1
        z = self.teacher_logits(bytes_)
        lp = torch.log_softmax(z, -1)
        ent = -(lp.exp() * lp).sum(-1)
        return torch.nn.functional.pad(ent, (1, 0))                          # [N][TB + 1], ent[:, 0] unused

    # ---- BLT -------------------------------------------------------------------------------------------------
    def hash_rows(self, buf):
        """buf [N][8 + TB] uint8 (numpy): the hash rows of each byte's 3..8-grams, [NG][N][TB]."""
        N, TB = buf.shape[0], self.c.TB
        out = np.empty((NG, N, TB), np.int64)
        with np.errstate(over="ignore"):
            for k in range(NG):
                n = k + 3; h = np.zeros((N, TB), np.uint64); pw = np.uint64(1); pr = np.uint64(PRIMES[k])
                for i in range(n):
                    h += (buf[:, 8 - n + 1 + i:8 - n + 1 + i + TB].astype(np.uint64) + np.uint64(4)) * pw
                    pw = pw * pr
                out[k] = (h % np.uint64(self.c.HV)).astype(np.int64)
        return out

    def patchify(self, pe, th):
        """pe [N][TB + 1] (predicted entropy of byte t): patch starts s [N][TB + 1] and patch counts, as sbmoth.c."""
        N, T, TG = pe.shape[0], self.c.TB, self.c.TG
        s = np.zeros((N, T + 1), np.int64); s[:, 0] = s[:, 1] = 1
        npat = np.full(N, 2, np.int64); ln = np.ones(N, np.int64)
        for t in range(2, T + 1):
            st = (pe[:, t] > th) | (ln >= PMAX)
            if t < T: st &= npat < TG
            s[:, t] = st; npat += st & (t < T); ln = np.where(st, 1, ln + 1)
        return s, npat

    def predictor(self, e):                                                  # e [N][TB][D] (detached) -> [N][TB]
        x = e.detach()
        a = (x @ self.Pw1.view(self.c.PH, -1).T) * rinv(x)[..., None] + self.Pb1
        return self.Pb2 + torch.relu(a) @ self.Pw2

    def forward(self, buf, msk, ent=None, train=False, allout=False):
        """buf [N][8 + TB] uint8 (masked input, 8 bytes of context), msk [N][TB] bool. Returns the logits of the
        bytes that get one (the masked ones, or all with allout) in batch order, and the predictor's MSE."""
        c, dev = self.c, self.dev
        N, T, D = buf.shape[0], c.TB, c.D
        hid = torch.from_numpy(self.hash_rows(buf)).to(dev)
        byt = torch.from_numpy(buf[:, 8:].astype(np.int64)).to(dev)
        x = self.Eb.view(V, D)[byt]
        for k in range(NG): x = x + self.Hs[k].view(c.HV, D)[hid[k]]
        e = self.SE(x, train=train)
        pr = self.predictor(e)                                               # pr[:, t]: entropy of byte t + 1
        pe = torch.nn.functional.pad(pr, (1, 0), value=1e30)                  # pe[:, t]: entropy of byte t
        ploss = torch.zeros((), device=dev)
        if train:
            ploss = ((pr - ent[:, 1:]) ** 2).mean()
            with torch.no_grad():                                            # the teacher's boundary rate, matched
                tmp = pe[:, 2:].flatten().sort().values; k = tmp.numel()
                hi = int((ent[:, 2:] > self.theta).sum()); q = float(tmp[max(k - 1 - hi, 0)])
                self.thetap = 0.95 * self.thetap + 0.05 * q if self.thetap else q
                q1, q2 = float(tmp[int((c.SK1 + c.SK2) * (k - 1))]), float(tmp[int(c.SK2 * (k - 1))])
                self.thsk[0] = 0.95 * self.thsk[0] + 0.05 * q1 if self.thsk[0] > -1e29 else q1
                self.thsk[1] = 0.95 * self.thsk[1] + 0.05 * q2 if self.thsk[1] > -1e29 else q2
                if not c.SK1 + c.SK2 > 0: self.thsk = [-1e30, -1e30]
                if not c.SK2 > 0: self.thsk[1] = -1e30
        pen = pe.detach().cpu().numpy()
        s, npat = self.patchify(pen, self.thetap)
        pid = np.cumsum(s[:, :T], 1) - 1                                     # each byte's patch
        skd = np.minimum(np.where(pen[:, :T] < self.thsk[1], 2, (pen[:, :T] < self.thsk[0]).astype(np.int64)), c.LD)
        # max-pool the encoder states per patch (empty patch slots stay 0)
        slot = torch.from_numpy((np.arange(N)[:, None] * c.TG + pid).reshape(-1)).to(dev)
        g = torch.zeros(N * c.TG, D, device=dev).scatter_reduce(0, slot[:, None].expand(-1, D), e.reshape(N * T, D),
                                                                reduce="amax", include_self=False).view(N, c.TG, D)
        z = self.SG(g, lens=torch.from_numpy(npat).to(dev), train=train)
        d = e + z[torch.arange(N, device=dev)[:, None], torch.from_numpy(pid).to(dev)]
        y = self.SD(d, skip=torch.from_numpy(skd).to(dev), train=train)
        out = torch.ones_like(msk) if allout else msk
        oi = torch.from_numpy(np.flatnonzero(out)).to(dev)
        logits = head(y.reshape(N * T, D)[oi], self.Wo.view(V, D))
        self.last = dict(npat=npat, skd=skd)
        return logits, oi, ploss

    # ---- sbmoth.c checkpoints ------------------------------------------------------------------------------------
    def header(self):
        c = self.c
        return [CKMAGIC, 1, c.M, c.TB, c.LE, c.LG, c.LD, c.LH, c.HV, NG, V, PMAX, len(self.P.ps)]

    def save(self, path, step, opt=None, rs=0x9E3779B97F4A7C15):
        with open(path + ".tmp", "wb") as f:
            f.write(struct.pack("<13i", *self.header()))
            f.write(struct.pack("<iQffff f", step, rs, self.theta, self.thetap, self.thsk[0], self.thsk[1], self.c.PSZ))
            for i, p in enumerate(self.P.ps):
                f.write(struct.pack("<i", p.numel()))
                for t in (p.detach(), *(opt.state_of(i) if opt else (torch.zeros_like(p), torch.zeros_like(p)))):
                    f.write(t.float().cpu().numpy().tobytes())
            for S in self.stacks:
                for l in range(S.L):
                    f.write(S.amax[l].float().cpu().numpy().tobytes())
                    f.write(S.gprev[l].float().cpu().numpy().tobytes())
        import os; os.replace(path + ".tmp", path)

    def load(self, path, opt=None):
        with open(path, "rb") as f:
            h = struct.unpack("<13i", f.read(52))
            if list(h) != self.header():
                raise SystemExit(f"checkpoint header {h} does not match this model's {self.header()} (same --M --HV --LG ... ?)")
            step, rs, self.theta, self.thetap, t0, t1, psz = struct.unpack("<iQffff f", f.read(32))
            self.thsk = [t0, t1]
            with torch.no_grad():
                for i, p in enumerate(self.P.ps):
                    n = struct.unpack("<i", f.read(4))[0]
                    if n != p.numel(): raise SystemExit(f"param {i}: {n} values, expected {p.numel()}")
                    w, m, v = (torch.from_numpy(np.frombuffer(f.read(4 * n), np.float32).copy()) for _ in range(3))
                    p.copy_(w.to(p.device))
                    if opt: opt.set_state(i, m.to(p.device), v.to(p.device))
                for S in self.stacks:
                    for l in range(S.L):
                        S.amax[l] = torch.from_numpy(np.frombuffer(f.read(16), np.float32).copy())
                        S.gprev[l] = torch.from_numpy(np.frombuffer(f.read(56), np.float32).copy()).view(7, 2)
        return step


class Adam:
    """sbmoth.c's optimiser: Adam (0.9, 0.95, 1e-8) with bias correction and decoupled weight decay per parameter;
    its lr schedule is warmup 100 steps, then cosine from LR to 0.1 LR. Steps <= 0 update nothing."""
    def __init__(self, params):
        self.ps = params; self.m = [torch.zeros_like(p) for p in params]; self.v = [torch.zeros_like(p) for p in params]

    def state_of(self, i): return self.m[i], self.v[i]
    def set_state(self, i, m, v): self.m[i].copy_(m); self.v[i].copy_(v)

    @torch.no_grad()
    def step(self, step, steps, k0, k1):
        lr = LR * step / 100 if step < 100 else LR * (0.1 + 0.45 * (1 + math.cos(math.pi * step / steps)))
        c1, c2 = 1 - 0.9 ** step, 1 - 0.95 ** step
        for i in range(k0, k1):
            p = self.ps[i]
            if p.grad is None or step <= 0:
                p.grad = None; continue
            g = p.grad; m, v = self.m[i], self.v[i]
            m.mul_(0.9).add_(g, alpha=0.1); v.mul_(0.95).addcmul_(g, g, value=0.05)
            p.sub_(lr * ((m / c1) / (torch.sqrt(v / c2) + 1e-8) + p.wd * p))
            p.grad = None


# ---- data: windows, BERT-style span masking ----------------------------------------------------------------
MRATE = 0.15

def mask_windows(data, offs, TB, rng):
    """sbmoth.c's masking: spans of 1..8 starting at rate MRATE / 4.5; masked bytes become MASK 80% of the time,
    a random byte of the window 10%, and stay 10%. Returns the masked buffers (8 bytes of context first), masks."""
    N = len(offs)
    buf = np.stack([data[o - 8:o + TB] for o in offs]).copy()
    msk = np.zeros((N, TB), bool)
    starts = rng.random((N, TB)) < MRATE / 4.5
    lens = 1 + (rng.random((N, TB)) * 8).astype(np.int64)
    for b, t in zip(*np.nonzero(starts)): msk[b, t:t + lens[b, t]] = True
    for b in np.flatnonzero(~msk.any(1)): msk[b, rng.integers(TB)] = True
    r = rng.random((N, TB)); rnd = msk & (r >= 0.8) & (r < 0.9)
    body = buf[:, 8:]
    body[msk & (r < 0.8)] = MASK
    src = np.array(offs)[:, None] + (rng.random((N, TB)) * TB).astype(np.int64)
    body[rnd] = data[src[rnd]]
    return buf, msk


def xent_masked(logits, oi, tgt, msk):
    """Mean cross-entropy over the masked bytes among the outputs."""
    flat = msk.reshape(-1)[oi]
    y = tgt.reshape(-1)[oi]
    lp = torch.log_softmax(logits, -1)
    nll = -lp.gather(1, y[:, None])[:, 0]
    return (nll * flat).sum() / flat.sum()


def train(a):
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Cfg(M=a.M, LE=a.LE, LG=a.LG, LD=a.LD, LH=a.LH, HV=a.HV, WDECAY=a.wd)
    raw = sys.stdin.buffer.read() if a.data == "-" else open(a.data, "rb").read()
    data = np.frombuffer(raw, np.uint8); ndata = len(data); ntrain = ndata * 9 // 10
    print(f"data: {ndata / 1e6:.1f} MB ({ntrain / 1e6:.1f} MB train); device {dev}"
          + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""), flush=True)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    model = Model(cfg, dev, a.seed); opt = Adam(model.P.ps)
    TB, B = cfg.TB, a.B
    pick = lambda n, val: rng.integers(ntrain if val else 8, (ndata if val else ntrain) - TB - 2, n)
    nall = sum(p.numel() for p in model.P.ps[model.h1:]); nemb = cfg.D * V * 2 + NG * cfg.HV * cfg.D
    trits = (cfg.LE + cfg.LG + cfg.LD) * (7 * 2 * cfg.W3 + 13 * cfg.D) + 2 * ((cfg.LE + cfg.LD) * TB * cfg.D + cfg.LG * cfg.TG * cfg.D)
    print(f"BLT, bidirectional (masked bytes): encoder {cfg.LE}, global {cfg.LG}, decoder {cfg.LD} layers, d {cfg.D}; "
          f"{nall / 1e6:.2f}M params to train ({nemb / 1e6:.2f}M embeddings); inference on {trits / 1e6:.2f}M trits; "
          f"{B} x {TB} bytes a step", flush=True)
    step0 = -1
    if a.resume:
        step0 = model.load(a.resume, opt) + 1; print(f"resumed {a.resume} at step {step0}", flush=True)
    # 1. the entropy model (teacher) -------------------------------------------------------------------------
    if not a.resume:
        t0 = time.time()
        for step in range(-1, a.hsteps + 1):
            offs = pick(B, False)
            w = torch.from_numpy(np.stack([data[o:o + TB + 1] for o in offs]).astype(np.int64)).to(dev)
            z = model.teacher_logits(w[:, :TB])
            loss = torch.nn.functional.cross_entropy(z.reshape(-1, V), w[:, 1:].reshape(-1))
            loss.backward(); opt.step(step, a.hsteps, 0, model.h1)
            if step % 250 == 0 and step > 0:
                print(f"  teacher step {step:5d} | loss {loss.item():.4f} | {(time.time() - t0) * 1e3 / 250:.0f} ms/step", flush=True); t0 = time.time()
        tmp = []                                     # threshold: the entropy quantile giving PSZ-byte patches, masked input
        for _ in range(32):
            buf, _m = mask_windows(data, pick(B, False), TB, rng)
            tmp.append(model.teacher_entropy(torch.from_numpy(buf[:, 8:].astype(np.int64)).to(dev))[:, 2:].flatten())
        tmp = torch.cat(tmp).sort().values; model.theta = float(tmp[int((1 - 1 / cfg.PSZ) * tmp.numel())])
        print(f"patch threshold {model.theta:.3f} nats", flush=True)
    for p in model.P.ps[:model.h1]: p.requires_grad_(False)
    # 2. BLT ------------------------------------------------------------------------------------------------------
    def batch(val):
        offs = pick(B, val)
        buf, msk = mask_windows(data, offs, TB, rng)
        tgt = torch.from_numpy(np.stack([data[o:o + TB] for o in offs]).astype(np.int64)).to(dev)
        return buf, torch.from_numpy(msk).to(dev), tgt

    def val():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for _ in range(8):
                buf, msk, tgt = batch(True); lg, oi, _ = model(buf, msk); tot += xent_masked(lg, oi, tgt, msk).item() / 8
        model.train(); return tot

    t0 = time.time(); nb = 0
    for step in range(step0, a.steps + 1):
        buf, msk, tgt = batch(False)
        ent = model.teacher_entropy(torch.from_numpy(buf[:, 8:].astype(np.int64)).to(dev))
        lg, oi, ploss = model(buf, msk, ent=ent, train=True)
        loss = xent_masked(lg, oi, tgt, msk)
        (loss + ploss).backward(); opt.step(step, a.steps, model.h1, len(model.P.ps)); nb += B * TB
        if step % 100 == 0 and step > 0:
            if dev.type == "cuda": torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"step {step:6d} | masked loss {loss.item():.4f} | boundary mse {ploss.item():.3f}, patch {B * TB / model.last['npat'].sum():.2f} bytes, "
                  f"decoder {cfg.LD - model.last['skd'].mean():.2f} layers | {dt * 1e3 / 100:.0f} ms/step, {nb / dt / 1e3:.0f}K bytes/s"
                  + (f", {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB" if dev.type == "cuda" else ""), flush=True)
            t0 = time.time(); nb = 0
        if step % a.val_every == 0 and step > 0:
            vl = val(); print(f"step {step:6d} | val masked loss {vl:.4f} nats/byte ({vl / math.log(2):.3f} bits/byte)", flush=True); t0 = time.time()
        if a.out and step > 0 and (step % a.ck_every == 0 or step == a.steps):
            model.save(a.out, step, opt); print(f"saved {a.out} at step {step}", flush=True); t0 = time.time()


def score(a):
    """sb_maskfile's output for a checkpoint: per masked byte, log p(true), log p(true either case), argmax."""
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = Cfg(M=a.M, LE=a.LE, LG=a.LG, LD=a.LD, LH=a.LH, HV=a.HV)
    model = Model(cfg, dev); model.load(a.ckpt); model.eval()
    txt = np.fromfile(a.text, np.uint8); mk = np.fromfile(a.mask, np.uint8).astype(bool)
    TB = cfg.TB; nw = (len(txt) - 8) // TB; out = []
    if a.limit: nw = min(nw, a.limit)
    with torch.no_grad():
        for w0 in range(0, nw, a.B):
            offs = [8 + (w0 + b) * TB for b in range(min(a.B, nw - w0))]
            buf = np.stack([txt[o - 8:o + TB] for o in offs]).copy(); msk = np.stack([mk[o:o + TB] for o in offs])
            buf[:, 8:][msk] = MASK
            lg, oi, _ = model(buf, torch.from_numpy(msk).to(dev))
            lp = torch.log_softmax(lg, -1)
            tgt = torch.from_numpy(np.stack([txt[o:o + TB] for o in offs]).astype(np.int64)).to(dev).reshape(-1)[oi]
            alt = torch.where((tgt >= 65) & (tgt <= 90), tgt + 32, torch.where((tgt >= 97) & (tgt <= 122), tgt - 32, tgt))
            l1 = lp.gather(1, tgt[:, None])[:, 0]
            lc = torch.where(alt != tgt, torch.logaddexp(l1, lp.gather(1, alt[:, None])[:, 0]), l1)
            lg2 = lg.clone(); lg2[:, MASK] = -float("inf")
            out.append(torch.stack([l1, lc, lg2.argmax(1).float()], 1).cpu())
    r = torch.cat(out).numpy()
    rec = np.zeros(len(r), dtype=[("lp", "<f4"), ("lc", "<f4"), ("am", "<i4")])
    rec["lp"], rec["lc"], rec["am"] = r[:, 0], r[:, 1], r[:, 2].astype(np.int32)
    rec.tofile(a.out)
    print(f"{len(r)} masked bytes: {-r[:, 0].sum() / math.log(2) / len(r):.4f} bits a masked byte")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("train", "score"):
        p = sub.add_parser(nm)
        p.add_argument("--M", type=int, default=16); p.add_argument("--HV", type=int, default=2048)
        p.add_argument("--LE", type=int, default=1); p.add_argument("--LG", type=int, default=4)
        p.add_argument("--LD", type=int, default=2); p.add_argument("--LH", type=int, default=1)
        p.add_argument("--device", default=None)
    t = sub.choices["train"]
    t.add_argument("data"); t.add_argument("-o", "--out"); t.add_argument("-r", "--resume")
    t.add_argument("--B", type=int, default=16); t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--hsteps", type=int, default=3000); t.add_argument("--wd", type=float, default=0.1)
    t.add_argument("--val-every", type=int, default=500); t.add_argument("--ck-every", type=int, default=5000)
    t.add_argument("--seed", type=int, default=1)
    s = sub.choices["score"]
    s.add_argument("ckpt"); s.add_argument("text"); s.add_argument("mask"); s.add_argument("out")
    s.add_argument("--B", type=int, default=64); s.add_argument("--limit", type=int, default=0, help="first N windows only")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else score(a)
