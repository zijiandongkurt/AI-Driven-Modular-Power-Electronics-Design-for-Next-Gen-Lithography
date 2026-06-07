def _run_one(filename):
            """Run a single netlist through the container and move the .raw output.

            Args:
                filename (str): Netlist filename (e.g. ``"top1.net"``) located
                    in the container's simulation directory.

            Returns:
                bool: True if the simulation succeeded and a .raw file was
                    produced; False otherwise.
            """
            container_path = f"Z:\\\\sim\\\\{filename}"
            result = subprocess.run(
                [str(self.RUN_SCRIPT), container_path],
                text=True
            )
            if result.returncode != 0:
                print(f"ERROR [{filename}]: {result.stderr.strip()}")
                return False

            # Move .raw from ltspice-files/ to output/<batchID>/
            raw_src = self.LTSPICE_FILES / filename.replace(".net", ".raw")
            if raw_src.exists():
                shutil.move(str(raw_src), str(output_path / raw_src.name))
                self._onSimulationComplete(Path(filename).stem)
                return True
            else:
                print(f"WARNING: No .raw output found for {filename}")
                return False

        # Run simulations in parallel
        ok, total = 0, len(net_files_to_run)
        with ThreadPoolExecutor(max_workers=self.PARALLEL_SIMS) as executor:
            futures = {executor.submit(_run_one, f): f for f in net_files_to_run}
            for future in as_completed(futures):
                if future.result():
                    ok += 1

        print(f"Batch '{batchID}' done — {ok}/{total} successful")
        if ok < total:
            print(f"WARNING: {total - ok} simulation(s) failed in batch '{batchID}'")

        return self.__getResultsOf(batchID=batchID, netlist_map=netlist_map)