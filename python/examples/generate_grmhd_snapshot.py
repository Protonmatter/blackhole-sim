from blackhole_sim.grmhd import assert_four_velocity_normalization, generate_analytic_grmhd_torus

snap = generate_analytic_grmhd_torus(spin_a=0.85, nr=64, ntheta=36, nphi=40)
err = assert_four_velocity_normalization(snap, samples=256)
snap.to_npz("out/torus_fixture.npz")
print(f"wrote out/torus_fixture.npz, shape={snap.shape}, normalization error={err:.3e}")
