from ltspice_runner_snellius import LTSpiceSimulator

sim = LTSpiceSimulator()
df = sim.simulate("test_batch")
print(df)


