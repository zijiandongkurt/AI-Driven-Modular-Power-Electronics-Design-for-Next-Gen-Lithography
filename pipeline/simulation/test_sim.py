from pipeline.simulation.local.ltspice_runner import TopologySimulator

sim = TopologySimulator()
df = sim.simulate("test_batch")
print(df)


