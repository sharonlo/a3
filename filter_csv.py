from __future__ import division
import csv
import sys
from collections import defaultdict

IN_FILENAME = sys.argv[1]
OUT_FILENAME = sys.argv[2]

c = 0
cur = 0
with open(IN_FILENAME, 'r') as infile:
    reader = csv.reader(infile)
    headers = next(reader)
    with open(OUT_FILENAME, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)
        for line in reader:
            if line[0] != '':
               writer.writerow(line)
