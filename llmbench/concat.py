import glob
import pandas as pd

rfile="all_results.csv"

with open(rfile, "w") as rf:
   for f in sorted(glob.glob('test_results/*.csv')):
       with open(f, 'r') as i:
           rf.write(i.read())
           print(f'process: {f}')

print(f'saved to {rfile}!') 
