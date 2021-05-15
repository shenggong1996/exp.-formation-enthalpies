import numpy as np
import os

f=open('id_prop_with_mp.csv') # cleaned IIT dataset
scidata = {}
for i in range(644):
    if i == 0:
        f.readline()
        continue
    ID, exp, mp, diff = f.readline().split(',')
    print (exp)
    diff = diff.replace('\n','')
    exp =float(exp)
    mp=float(mp)
    diff=float(diff)
    scidata[ID] = [exp,mp,diff]
f.close()

qmpy = {}
f=open('mp_qmpy.csv') # cleaned SSUB dataset
for i in range(554):
    ID, _, exp,mp,diff = f.readline().split(',')
    diff = diff.replace('\n','')
    exp =float(exp)
    mp=float(mp)
    diff=float(diff)
    qmpy[ID] = [exp,mp,diff]
f.close()

cifs = os.listdir('sample/') # where you store the MP structures
for cif in cifs:
    cif=cif.replace('.cif','')
    if cif in scidata.keys(): # for overlaps, we use the IIT data as the priority.
        exp = scidata[cif][0]
        mp = scidata[cif][1]
        diff = scidata[cif][2]
        print ('%s,%f,%f,%f'%(cif,exp,mp,diff),file=open('data_cleaned.csv','a'))
    elif cif in qmpy.keys():
        exp = qmpy[cif][0]
        mp = qmpy[cif][1]
        diff = qmpy[cif][2]
        print ('%s,%f,%f,%f'%(cif,exp,mp,diff),file=open('data_cleaned.csv','a'))
