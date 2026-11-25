"""Does TVB's initial history depend on the noise seed in the companion pipeline? (doc/LOG.md, 2026-11-24, burn-in check.)"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); REF = os.path.join(HERE, "pci_ref"); sys.path.insert(0, os.path.join(REF, "utils"))
import numpy as np
import simulation
H = {}
for seed in (1, 2, 3):
    conn = simulation.configure_conn(os.path.join(REF, "hcp-001.zip"), conn_speed=20.)
    sim = simulation.configure_sim(G=0.54, nsigma=0.030, conn=conn, dt=0.01, seed=seed)
    H[seed] = np.array(sim.history.buffer[:, :, :, 0]); n1 = sim.integrator.noise.generate(shape=(2, 84, 5)).copy()
    print("seed", seed, "| history shape", H[seed].shape, "| r mean %.4f, first values" % H[seed][:, 0].mean(), np.round(H[seed][0, 0, :4], 4).tolist(), "| first noise values", np.round(n1[0, 0, :3], 5).tolist())
print("history identical for seeds 1 and 2:", bool(np.array_equal(H[1], H[2])), "| 1 and 3:", bool(np.array_equal(H[1], H[3])))
print("max |difference| between seeds 1 and 2:", float(np.abs(H[1] - H[2]).max()))
