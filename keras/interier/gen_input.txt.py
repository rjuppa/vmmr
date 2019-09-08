#!/usr/bin/env python

# it generates 100 line input files (input_0.txt) with a name of path of folders
# those are processed by SVM separation script
# Each line produces a separate job in Metacentrum by generate_jobs.py

import os
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Need param: python gen_input.py 'vmmr/datasets/source/aaa_auto1/images'")
        exit(1)

    folder = str(sys.argv[1])
    exists = os.path.isdir(folder)
    if not exists:
        print("Folder '{}' not found.".format(folder))
        exit(1)

    if "/source/" not in folder:
        print("Folder '{}' must be in /source/ directory.".format(folder))
        exit(1)

    files = os.listdir(folder)
    dirs = list(map(lambda x: os.path.join(folder, x), files))
    lines = []
    for d in sorted(dirs):
        if os.path.isdir(d):
            files = os.listdir(d)
            files = list(map(lambda x: os.path.join(d, x), files))
            for f in sorted(files):
                if os.path.isdir(f):
                    lines.append(f)

    # split to parts
    batch_size = 200
    total = len(lines)
    batch_count = int(total / batch_size) + 1

    i = 0
    for b in range(batch_count):
        with open("input_{}.txt".format(b), "w") as file:
            for d in range(batch_size):
                try:
                    line = lines[i]
                    file.write("{}\n".format(line))
                    i += 1
                except IndexError:
                    print("Done.")
                    exit(0)

    print("Done.")
