# bpropy/adapters/registry.py
from bpropy.co.mis import MIS
from bpropy.co.scp import SCP
from bpropy.co.mis_data import data_to_mis
from bpropy.co.scp_data import data_to_scp

PROBLEM_REGISTRY = {
    "MIS": {"from_data": data_to_mis, "cls": MIS},
    "SCP": {"from_data": data_to_scp, "cls": SCP},
}
CODE2NAME = {0: "MIS", 1: "SCP"}
NAME2CODE = {v: k for k, v in CODE2NAME.items()}

def recover_problem(data):
    name = getattr(data, "problem_type_name", None)
    if name is None and hasattr(data, "problem_type_code"):
        name = CODE2NAME[int(data.problem_type_code[0])]
    return PROBLEM_REGISTRY[name]["from_data"](data)

def solve_data(data, **solver_kwargs):
    pb = recover_problem(data)
    # call the right .solve(): MIS.maximize with clique constraints, SCP.minimize with coverage
    return pb.solve(**solver_kwargs)

