#!/usr/bin/python3
import subprocess
import os

experiments = ["S0-41",
               "S0-42",
               "S0-43",
               "S0-44",
               "S0-45",
               "S0-46",
               "S0-47",
               "S0-48",
               "S0-49",
               "S0-50"]

save_logs = True
logdir = "../logs/"


for expt in experiments:
    # Remove ".py" if in experiment string
    if ".py" in expt[-3:]:
        expt = expt[:-3]

    # Set up the log file
    if save_logs:
        # First check the logdir exists, create if it doesn't
        try:
            os.stat(logdir)
        except FileNotFoundError:
            os.mkdir(logdir)

        # Define where to save the log
        log = logdir + "{}.log".format(expt)
        log = open(log, 'w+')
    else:
        # If not saving logs, set 'log' to None
        log = None

    # Set the command to run
    cmd = "~/anaconda3/envs/NODE-Operator-TF2/bin/python ./{}.py".format(expt)
    # Run the command
    cp = subprocess.run(cmd,
                        shell=True,
                        stdout=log,
                        stderr=subprocess.STDOUT)

    if save_logs:
        log.close()

    print("Completed Experiment", expt)
