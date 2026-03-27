from PyLTSpice import LTspice, SimRunner, SpiceEditor, RawRead
from pathlib import Path
import pandas as pd

class TopologySimulator():
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent

    def _onSimulationComplete(self, raw_file, log_file):
        """Event informing completion of a simulation of a particular netlist
        """
        print(f"SIMULATION COMPLETE: {raw_file}, {log_file}")
        raw_path = Path(raw_file)

        for suffix in [".net", ".db", ".op.raw"]:
            unwanted = raw_path.with_suffix(suffix) if not suffix.startswith(".op") \
                    else raw_path.with_name(raw_path.stem + suffix)
            if unwanted.exists():
                unwanted.unlink()
    
    def simulate(self, batchID):
        """Simulates a batch of valid topologies by reading its ID and mapping it onto the actual .net file
        
        PARAMS:
        batchID <string> : The ID of a Batch 

        RETURNS:
        results <DataFrame> : Results of the batch simulation
        
        """

        #Init paths and valid topologies location
        output_path = self.BASE_DIR / "output" / batchID
        netdir = self.BASE_DIR / "valid_topologies" / batchID
        assert netdir.exists(), f"Batch folder not found: {netdir.resolve()}"

        #Init runner for desired output path
        runner = SimRunner(output_folder=output_path, simulator=LTspice, parallel_sims=4)

        #Loop over all topologies within the batch and run simulations parallel
        netdir = self.BASE_DIR / "valid_topologies" / batchID

        for netpath in netdir.glob("*.net"):
            net = SpiceEditor(netpath)
            runner.run(net,
                       run_filename= netpath.name,
                       callback=self._onSimulationComplete)

        #Wait until batch simulation is complete
        runner.wait_completion()
        print(f"Batch '{batchID}' done — {runner.okSim}/{runner.runno} successful")
        
        #Format simulation data to dataframe
        results = self.__getResultsOf(batchID=batchID)
        return results

    def __getResultsOf(self, batchID):
        """Directly ran after a batch simulation, it is meant to compress the results into a dataframe for further processing
        within the reward function.

        PARAMS:
        batchID <string> : The ID of a Batch 

        RETURNS:
        combined <DataFrame> : The dataframe consisting of all simulation data read from the .raw output of each file.
        """


        #Get output directory of .raw files
        output_dir = self.BASE_DIR / "output" / batchID
        assert output_dir.exists(), f"No results found for batch: {batchID}"

        #get .raw files of batch
        raw_files = list(output_dir.glob("*.raw"))
        assert raw_files, f"No .raw files found in: {output_dir}"

        all_frames = []

        #go through each simulation result of that batch, and convert to data rows
        for raw_path in raw_files:
            raw = RawRead(raw_path)
            steps = raw.get_steps()       

            for step in steps:
                df = raw.to_dataframe(step=step)   # all traces for this step, as columns
                df["source_file"] = raw_path.stem  # tag which netlist it came from
                df["step"] = step
                all_frames.append(df)

        #combine dataframes per batch into one
        combined = pd.concat(all_frames, ignore_index=True)

        #save as .csv into results directory
        results_dir = self.BASE_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        csv_path = results_dir / f"{batchID}_out.csv"
        combined.to_csv(csv_path, index=False)
        print(f"Saved {len(all_frames)} run(s) -> {csv_path}")

        #return dataframe for further processing
        return combined

#testing
# simo = TopologySimulator()
# df = simo.simulate("batch_001")
