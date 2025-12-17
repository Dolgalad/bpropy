"""
- see Iterative Proportional Fitting
"""
import torch

from bpropy.lbp import *
from dopy.maxcut import *
from progress.bar import Bar

import matplotlib.pyplot as plt


# train device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# number of nodes
n = 100

# maximum BP iterations
max_iter = 1000
convergence_tol = 1e-3
pool_size = 100
lr = 1e-2
loss = torch.nn.BCELoss()

# Randomly generate a graph
g = nx.generators.erdos_renyi_graph(n, 0.15)
edge_list = [e for e in g.edges]


# Solve ILP
w = np.random.random_sample(n)
w = np.ones(n)
pb = MaxCut(g, weights=w)
print("Solving problem...")
t_sol = time.time()
ilp_sol = pb.solve()
t_sol = time.time() - t_sol
ilp_val = np.dot(w, ilp_sol.x > 0.5)
print(f"Done in {t_sol}s, val = {ilp_val}")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def rand_maxcut_greedy(pb, node_scores):
    queue = list(range(pb.nv()))
    sol = np.zeros(pb.nv())
    while len(queue)>0:
        #u = queue[np.argmax(node_scores[np.array(queue)])]
        u = np.random.choice(queue, p=softmax(node_scores[np.array(queue)]))
        sol[u] = 1
        queue.remove(u)
        for v in pb.g.neighbors(u):
            if v in queue:
                sol[v] = 0
                queue.remove(v)
    return sol


def get_solution_pool(pb, n=10):
    bar = Bar("Getting pool: ", max=n)
    pool = []
    values = []
    while len(pool) < n:
        nw = np.random.random_sample(pb.nv())
        sol = maxcut_greedy(pb, nw)
        have_it_already = False
        for s in pool:
            if np.all(s == sol):
                have_it_already = True
                break
        if not have_it_already:
            pool.append(sol)
            bar.next()
    bar.finish()
    return pool

pool = get_solution_pool(pb, n=pool_size)
#pool.append(ilp_sol.x > 0.5)

# Solve using inference
fg = create_factor_graph(g)
n_ss = sufficient_statistics_dim(fg)

# create the computation edges
f2v_idx, f2v_subsum_idx,ss_idx = f2v_index_map(fg)
# creating message passing edges
f2v_edges,r_edges = f2v_subsum_indexes(fg, f2v_idx, ss_idx)
f2v_edges = torch.from_numpy(f2v_edges).to(device)
r_edges = torch.from_numpy(r_edges).to(device)
#f2v_edges2,r_edges2 = f2v_subsum_indexes2(fg)
#f2v_edges2 = torch.from_numpy(f2v_edges2)
#r_edges2 = torch.from_numpy(r_edges2)
# scatter indices
q_edges = torch.from_numpy(v2f_indexes(fg, f2v_idx)).to(device)
#q_edges2 = torch.from_numpy(v2f_indexes2(fg))
m_edges = torch.from_numpy(marginal_indexes(fg, f2v_idx, ss_idx)).to(device)
#m_edges2 = torch.from_numpy(marginal_indexes2(fg))
vm_edges = torch.from_numpy(variable_marginal_indexes(fg, f2v_idx, ss_idx)).to(device)

# stack sufficient stats and v2f messages
v2f = torch.zeros(len(f2v_idx), dtype=torch.float).to(device)
theta = torch.zeros(len(ss_idx), dtype=torch.float)
c = 0
for i in range(pb.nv()):
    theta[c+1] = 1
    c += 2
for e in pb.g.edges:
    theta[c+1] = 1. / len(list(g.neighbors(e[0])))
    theta[c+2] = 1. / len(list(g.neighbors(e[1])))
    theta[c+3] = -n
    c += 4
theta = theta.detach().to(device).requires_grad_()


loss_values  = []
gap_values = []


optimizer = torch.optim.Adam([theta], lr=lr)

pool_values = np.array([np.dot(w, s) for s in pool])
Z = np.sum(np.exp(pool_values))

target = np.stack([np.exp(np.dot(w,s))*s for s in pool])
target = np.sum(target, axis=0)/Z
target = torch.from_numpy(target).to(torch.float).to(device)

trg_sol = maxcut_greedy(pb, target.cpu().numpy())
trg_val = np.dot(w, trg_sol > 0.5)
trg_gap = ( ilp_val - trg_val ) / ilp_val

print("Target gap = ", trg_gap)

for it in range(100000):
    def closure():
        global target
        optimizer.zero_grad()
        beliefs = residual_loopy_belief_propagation(theta, v2f.detach().clone(), fg, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6)
        l = loss(beliefs[1:2*n+1:2], target)

        # solve greedy
        grd_sol = rand_maxcut_greedy(pb, beliefs[1:2*n+1:2].detach().cpu().numpy())
        grd_val = np.dot(w, grd_sol > 0.5)
        gap = ( ilp_val - grd_val ) / ilp_val
        sol_found = False
        for s in pool:
            if np.all(s == grd_sol):
                sol_found = True
                break
        if not sol_found:
            pool.append(grd_sol.copy())
            pool_values = np.array([np.dot(w, s) for s in pool])
            Z = np.sum(np.exp(pool_values))
            
            target = np.stack([np.exp(np.dot(w,s))*s for s in pool])
            target = np.sum(target, axis=0)/Z
            target = torch.from_numpy(target).to(torch.float).to(device)
            
        print(it, l.item(), beliefs.shape, gap)

        loss_values.append(l.item())
        gap_values.append(gap)

        l.backward()

        fig,ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(loss_values)
        ax[0].set_ylabel("BCELoss")
        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Step")
        ax[1].plot(gap_values)
        ax[1].set_ylabel("Gap")
        ax[1].set_xlabel("Step")
        ax[1].set_xscale("log")

        plt.savefig(f"learning_rbp_parameters_n{n}_lr{lr}.png")
        plt.close()
        return loss
    optimizer.step(closure)
    #optimizer.zero_grad()
    #target = torch.from_numpy(ilp_sol.x)

    #beliefs = residual_loopy_belief_propagation(theta.clone(), v2f, fg, f2v_edges, f2v_subsum_idx, r_edges, q_edges, m_edges, max_iter=100, tol=1e-6)

    #l = loss(beliefs[1:2*n+1:2], target)
    #l.backward()

    #optimizer.step()



