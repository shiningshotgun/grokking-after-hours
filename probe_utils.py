import hashlib


def canonical_state_hash(sd):
    h = hashlib.sha256()
    for name, t in sorted(sd.items()):
        a = t.detach().to(device="cpu", copy=True).contiguous()
        h.update(name.encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(str(a.dtype).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()
