import os
import tempfile
import pickle as pkl

from bpropy.co.mis import random_mis
from bpropy.adapters.mis_data import mis_to_data

def create_in_memory_dataset(N=10, n=100, p=0.15, formulation="EB", with_lbp_edges=True, with_nmf_edges=False):
        # create a small dataset of problems
        if callable(n):
            pbs = [random_mis(n(), p) for _ in range(N)]
        elif hasattr(n, "__iter__"):
            pbs = [random_mis(v, p) for v in n]
        else:
            pbs = [random_mis(n,p) for _ in range(N)]
        # create the data objects
        data = [mis_to_data(pb, formulation, with_lbp_edges=with_lbp_edges, with_nmf_edges=with_nmf_edges) for pb in pbs]
        # save to temporary pickle file
        data_file = tempfile.NamedTemporaryFile("w+b", delete=False)
        pkl.dump(data, data_file, protocol=pkl.HIGHEST_PROTOCOL)
        data_file.close()
        return data_file.name, data

def create_on_disk_dataset(N=10, n=100, p=0.15, with_lbp_edges=True, with_nmf_edges=False):
        # create a small dataset of problems
        pbs = [random_mis(n,p) for _ in range(N)]
        # create the data objects
        data = [mis_to_data(pb, "EB", with_lbp_edges=with_lbp_edges, with_nmf_edges=with_nmf_edges) for pb in pbs]
        # save to temporary pickle file
        data_dir = tempfile.mkdtemp()
        for i,d in enumerate(data):
            with open(os.path.join(data_dir, f"{i}.dat"), "wb") as f:
                pkl.dump(d, f, protocol=pkl.HIGHEST_PROTOCOL)
        return data_dir, data

