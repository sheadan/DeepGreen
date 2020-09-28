
import subprocess
import os

experiments = ["S0-61",
               "S0-62",
               "S0-63",
               "S0-64",
               "S0-65",
               "S0-66",
               "S0-67",
               "S0-68",
               "S0-69",
               "S0-70" ]

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
    cmd = "python3 ./{}.py".format(expt)
    # Run the command
    cp = subprocess.run(cmd,
                        shell=True,
                        stdout=log,
                        stderr=subprocess.STDOUT)

    if save_logs:
        log.close()

    print("Completed Experiment", expt)
