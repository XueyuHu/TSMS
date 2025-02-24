import os
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV

mpr = MPRester("3Llvxl8s9cnyDiZfbTNjyGH3uZcIZ1zz")
entries = mpr.get_entry_by_material_id("mp-18748")
compatibility = MaterialsProjectCompatibility()
processed_entries = compatibility.process_entries(entries)
cwd = os.getcwd()
costdb_path = "%s\costdb_elements_2021.csv" %cwd
costdb = CostDBCSV(costdb_path)
costanalyzer = CostAnalyzer(costdb)
#comp = Composition(processed_entries['Composition'])
for entry in processed_entries:
    print(entry.composition.reduced_formula)
    print(costanalyzer.get_cost_per_kg(entry.composition.reduced_formula))
#print(processed_entries['Composition'])
#material_data = mpr.get_entry_by_material_id("mp-18748")
#a = material_data['Composition']
#print(a)
#print(material_data)
#comp = Composition("La16Co14Fe2O48").reduced_formula
#print(comp)