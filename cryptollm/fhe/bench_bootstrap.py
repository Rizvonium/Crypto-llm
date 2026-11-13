"""Measure 128-bit-secure CKKS costs with OpenFHE, incl. bootstrapping (the noise refresh
that deep circuits such as transformer layers need).

Needs: pip install openfhe  (current wheels contain a CPython 3.12 binary) and ~8 GB RAM.
Measured on a 4-core cloud VM (OMP_NUM_THREADS=4), ring dim 2^16, 1024 sparse slots:
    keygen 7.6 s / 7.1 GB RAM, bootstrap 11.9 s, ct*ct mult 148 ms, ct*pt mult 8.9 ms,
    rotation 155 ms.   (Plaintext float ops on the same CPU: nanoseconds.)
"""
import resource
import time

from openfhe import *
p = CCParamsCKKSRNS()
p.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
p.SetScalingModSize(50); p.SetFirstModSize(59); p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
levelBudget=[2,2]; levelsAfter=5; slots = 1<<10
depth = levelsAfter + CryptoContext.GetBootstrapDepth(levelBudget, SecretKeyDist.UNIFORM_TERNARY)
p.SetMultiplicativeDepth(depth); p.SetBatchSize(slots)
t=time.time(); cc = GenCryptoContext(p)
for f in [PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE, PKESchemeFeature.ADVANCEDSHE, PKESchemeFeature.FHE]: cc.Enable(f)
print(f"ring dim {cc.GetRingDimension()} depth {depth} slots {slots}", flush=True)
cc.EvalBootstrapSetup(levelBudget, [0,0], slots)
kp = cc.KeyGen(); cc.EvalMultKeyGen(kp.secretKey); cc.EvalBootstrapKeyGen(kp.secretKey, slots)
cc.EvalRotateKeyGen(kp.secretKey, [1])
print(f"setup+keygen {time.time()-t:.1f}s  maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)
x = [0.001*i for i in range(slots)]
c = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(x, 1, depth-1, None, slots))
t=time.time(); c2 = cc.EvalBootstrap(c); bt=time.time()-t
r = cc.Decrypt(c2, kp.secretKey); r.SetLength(slots); vals=r.GetRealPackedValue()
err = max(abs(a-b) for a,b in zip(vals, x))
print(f"bootstrap {bt:.2f}s  max err {err:.2e}  levels left {depth - c2.GetLevel()}", flush=True)
a = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(x, 1, 0, None, slots)); b = cc.Encrypt(kp.publicKey, cc.MakeCKKSPackedPlaintext(x, 1, 0, None, slots))
t=time.time(); m = cc.EvalMult(a,b); print(f"ct*ct mult (top level) {1000*(time.time()-t):.1f} ms")
ptw = cc.MakeCKKSPackedPlaintext(x, 1, 0, None, slots)
t=time.time(); m = cc.EvalMult(a,ptw); print(f"ct*pt mult (top level) {1000*(time.time()-t):.1f} ms")
t=time.time(); rr = cc.EvalRotate(a,1); print(f"rotation (top level) {1000*(time.time()-t):.1f} ms")
print(f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
