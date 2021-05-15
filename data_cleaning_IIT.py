import json
from pymatgen.ext.matproj import MPRester
import pymatgen as mg
import numpy as np
import os


def get_structure(ID):#get structures of different metal oxides
    with MPRester(") as m: # your token
        structure = m.get_structure_by_material_id("%s"%(ID))
        structure.to(filename="sample/%s.cif"%(ID))
        print ("%s"%(ID))
    return()

with open('enthalpy_formation.json') as f: # the downloaded IIT database
    data = json.load(f)

for matter in data:
    if not matter['standard_enthalpy_formation']:
        continue
    if matter['materials_project']['mp_id'] and matter['materials_project']['mp_formation_energy'][1][0]:
        ID = matter['materials_project']['mp_id']
        get_structure(ID) # different from the script for the SSUB database, here we provide a function to download structures from materials project
        print ('%s,%f,%f,%f'%(matter['materials_project']['mp_id'],matter['standard_enthalpy_formation'][1][0],matter['materials_project']['mp_formation_energy'][1][0],matter['standard_enthalpy_formation'][1][0]-matter['materials_project']['mp_formation_energy'][1][0]), file=open('id_prop_with_mp.csv','a'))
    else:
        print ('%s,%f'%(matter['id'],matter['standard_enthalpy_formation'][1][0]), file=open('id_prop_no_mp.csv','a')) # Here we separate the materials in the IIT database that have corresponding materials project structures from those that don't have corresponding materials project structures. 

