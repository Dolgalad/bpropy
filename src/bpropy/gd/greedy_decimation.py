from bpropy.gd.inference_backend import InferenceBackend
from bpropy.gd.problem_adapter import ProblemAdapter

import time

def greedy_decimation(pb, backend: InferenceBackend, adapter: ProblemAdapter, *, max_steps: int = 10**9, return_log: bool = False):

    backend.reset(pb)
    adapter.reset(pb)

    log = {"step": [], "evidence": [], "info": [], "backend_out": []} if return_log else None

    steps = 0

    while not adapter.is_done() and steps < max_steps:
        out = backend.observe()              # mu, scores, or None
        
        evidence, info = adapter.step(out, return_log=return_log)   # adapter decides based on out + its own state

        backend.update(evidence)             # backend clamps/updates internally

        if return_log:
            log["step"].append(steps)
            log["evidence"].append(evidence)
            log["info"].append(info)
            # store out optionally; might be big
            log["backend_out"].append(None if out is None else "stored")

        steps += 1
    
    sol = adapter.solution()
    return (sol, log) if return_log else sol
