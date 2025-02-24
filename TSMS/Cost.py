import os
from collections import Counter
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV

vector = ["Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Al"]
element_counts = Counter(vector)

formatted_string = "".join(f"{element}{count}" for element, count in element_counts.items()) + "O48"

cwd = os.getcwd()
costdb_path = "%s\costdb_elements_2021.csv" %cwd
costdb = CostDBCSV(costdb_path)
costanalyzer = CostAnalyzer(costdb)
print(costanalyzer.get_cost_per_kg(formatted_string))

print(costanalyzer.get_cost_per_kg("Ba16Mn15AlO48"))
print(costanalyzer.get_cost_per_kg("Ba16Mn15Al1O48"))