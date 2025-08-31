class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.next = None
        self._load_next()

    def _move(self, obj):
        # tensors / PyG Data/HeteroData / LBPEdges/NMFEdges should all implement .to(...)
        if hasattr(obj, "to"):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        return obj

    def _load_next(self):
        try:
            nxt = next(self.loader)
            # nxt may be a single graph batch, or a tuple like (batch, y) or (pbs, batch, y)
            self.next = self._move(nxt)
        except StopIteration:
            self.next = None

    def __iter__(self): return self
    def __next__(self):
        if self.next is None:
            raise StopIteration
        cur = self.next
        self._load_next()
        return cur

