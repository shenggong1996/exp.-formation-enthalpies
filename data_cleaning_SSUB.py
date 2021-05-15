from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
import numpy as np
import os

f=open('qmpy.txt') # file of the downloaded SSUB dataset 
qmpy = {}
for i in range(2090):
    ID, exp = f.readline().split(' ')
    exp =float(exp)
    if ID in qmpy.keys():
        if exp < qmpy[ID]: # extract the lowest energy for each formula from SSUB
            qmpy[ID] = exp
    else:
        qmpy[ID]=exp
f.close()

mpdata = {}
n=0
f=open('../sample/id_prop.csv') # where you store data of materials project formation energy
for i in range(126356):
    ID, mp = f.readline().split(',')
    mp = float(mp)
    struct = Structure.from_file('../sample/%s.cif'%(ID))
    comp = struct.composition.reduced_formula
    if comp in qmpy.keys():
        if comp in mpdata.keys():
            if mp < mpdata[comp][3]: # extract the lowest energy for each formula from MP
                mpdata[comp][3] = mp
        else:
            mpdata[comp] = [ID,comp,qmpy[comp],mp,qmpy[comp]-mp]
            n+=1
        print (n,comp,qmpy[comp]-mp)
f.close()

for comp in mpdata.keys():
    ID = mpdata[comp][0]
    comp = mpdata[comp][1]
    exp = mpdata[comp][2]
    mp = mpdata[comp][3]
    os.system('cp ../sample/%s.cif sample/'%(ID))
    print ('%s,%s,%f,%f,%f'%(ID,comp,exp,mp,exp-mp),file=open('mp_qmpy.csv','a')) # where you store the cleaned SSUB dataset
