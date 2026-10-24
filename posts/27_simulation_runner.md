Day 27 of 30: Simulation Workflow Runner

Each case in a simulation campaign is a chain: write the input, solve, post-process, check. When a node dies at case 37 of 60, the usual shell script cannot tell you which cases finished, and the results still end up in folders.

The Simulation Workflow Runner takes a table of cases (a Campaign Planner export works directly) and a workflow template, and runs every case as a DAG of steps. Each step keeps its state, tries, time, log and content hash.

In the video I ran the built-in slab-heating template (exact Fourier-series solution; CalculiX and OpenFOAM are not installed on this machine). An uploaded table with T_left = 650 K was refused before anything ran. The 6 example cases ran as 7 steps, 0 failed; running them again reused all 7 steps by content hash, with 0.00 s of compute. The results became a 6-row dataset with a DatasetSpec contract: 6 inputs and 5 outputs with units, the file's SHA-256 and the run step behind each row. A check on that table: 6 of 6 cases have T_mid and T_mean between T0 and T_left.

On a deliberately flaky workflow, one step passed on its second try, another failed and cancelled only the step that needed it, and Resume finished the run (4 of 4 done) without redoing the finished steps.

Testing it surfaced a bug, now fixed in the library: a CSV with T0 = 300 was refused for a template constant of 300.0.

Live: https://runner.pinneapple.org
Code: https://github.com/pinneapple-labs/pinneapple (apps/simulation_runner)

#Simulation #CAE #Workflow #HPC #ScientificComputing #DataEngineering #OpenSource
