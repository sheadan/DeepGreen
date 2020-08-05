#!/usr/bin/python3
import subprocess

experiments = ["List", "Experiment", "Filenames", "Without", ".py"]

save_logs = True

for expt in experiments:
    if ".py" in expt[-3:]:
        expt = expt[:-3]

    # Run the experiment script file
    cmd = "python3 {}.py".format(expt)
    cp = subprocess.run(cmd, stderr=subprocess.STDOUT)

    if save_logs:
        # Define where to save the log
        log = "../logs/{}.log".format(expt)
        stdout = cp.stdout

        # And write the logs
        with open(log, 'w') as f:
            f.write(stdout)

    print("Completed Experiment", expt)
