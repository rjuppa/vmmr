#!/usr/bin/env python

# it processes input files like input_1.txt
# Each line produces a separate job in Metacentrum using template_job.pbs

import os
import subprocess
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Need param: python generate_jobs.py 'input_1.txt'")
        exit(1)

    input_file = str(sys.argv[1])
    exists = os.path.isfile(input_file)
    if not exists:
        print("File '{}' not found.".format(input_file))
        exit(1)

    with open(input_file, "r") as file:
        lines = file.readlines()

    n = 0
    for f in lines:
        f = f.replace("\n", "")
        if os.path.isdir(f):
            source_dir = f.split("/")[-3]
            make = f.split("/")[-2]
            if make:
                model = f.split("/")[-1]
                if model:
                    qsub_command = """qsub -v MAKE={0},MODEL={1},SOURCE={2}  template_job.pbs""".format(make, model, source_dir)
                    print(qsub_command)
                    # Comment the following 3 lines when testing to prevent jobs from being submitted
                    exit_status = subprocess.call(qsub_command, shell=True)
                    if exit_status is 1:  # Check to make sure the job submitted
                        print("Job {0} failed to submit".format(qsub_command))
                    n += 1


    print("Done submitting jobs!")
