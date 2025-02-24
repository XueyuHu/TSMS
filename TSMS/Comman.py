import pandas as pd
import os
import time
start_time = time.time()
#os.system("python .\G+V(Single).py")
os.system("python .\G+V(Double).py")
#应当更改四个标注之处

targets = ["Ehull","p-band center","EV","EH","d-band center","exp Ehull","exp p-band center","exp EV","exp EH","exp d-band center","a","b","c","alpha","beta","gamma","Volume","ShrinkageV","ShrinkageH","FreeVolume","OverlappingArea","OverlappingCenter"]
targets_1 = ["Ehull","p-band center","EV","EH","d-band center","exp Ehull","exp p-band center","exp EV","exp EH","exp d-band center","a","b","c","alpha","beta","gamma","Volume","ShrinkageV","ShrinkageH","FreeVolume"]
targets_2 = ["OverlappingArea","OverlappingCenter"]
for target in (targets):
    command = f'python ML.py -model xgb -stage 1 -train Data/dataset_1.csv -test vector_data.csv -targ "{target}" -pth "SmSr_{target}"' #这里可以改一下K
    os.system(command)

vector_data = pd.read_csv('vector_data.csv').iloc[:,:155]

additional_columns = []
os.chdir("./Result")
for target in targets_1:
    file_name = f'SmSr_{target}_pred.csv' #一定要改对了
    quoted_file_name = f'"{file_name}"'
    column = pd.read_csv(file_name, usecols=[11])
    column.columns = [target]
    additional_columns.append(column)

#num_rows = additional_columns.shape[0]
num_rows = additional_columns[0].shape[0]
SpaceGroup_column = pd.DataFrame(["P4/mmm"]*num_rows, columns=["SpaceGroup"])
additional_columns.append(SpaceGroup_column)
PointGroup_column = pd.DataFrame(["15[D4h]"]*num_rows, columns=["PointGroup"])
additional_columns.append(PointGroup_column)
International_column = pd.DataFrame(["123"]*num_rows, columns=["International"])
additional_columns.append(International_column)
SymmetryOperations_column = pd.DataFrame(["256"]*num_rows, columns=["SymmetryOperations"])
additional_columns.append(SymmetryOperations_column)

#file_name = 'All_Add_pred.csv'
#columns = pd.read_csv(file_name, usecols=range(175,179))
#additional_columns.append(columns)

for target in targets_2:
    file_name = f'SmSr_{target}_pred.csv' #一定要改
    quoted_file_name = f'"{file_name}"'
    column = pd.read_csv(file_name, usecols=[11])
    column.columns = [target]
    additional_columns.append(column)

SG_1_column = pd.DataFrame(["0"]*num_rows, columns=["SG_1"])
additional_columns.append(SG_1_column)
SG_2_column = pd.DataFrame(["0"]*num_rows, columns=["SG_2"])
additional_columns.append(SG_2_column)
SG_3_column = pd.DataFrame(["1"]*num_rows, columns=["SG_3"])
additional_columns.append(SG_3_column)
SG_4_column = pd.DataFrame(["0"]*num_rows, columns=["SG_4"])
additional_columns.append(SG_4_column)
SG_5_column = pd.DataFrame(["0"]*num_rows, columns=["SG_5"])
additional_columns.append(SG_5_column)
SG_6_column = pd.DataFrame(["0"]*num_rows, columns=["SG_6"])
additional_columns.append(SG_6_column)
SG_7_column = pd.DataFrame(["0"]*num_rows, columns=["SG_7"])
additional_columns.append(SG_7_column)
PG_1_column = pd.DataFrame(["0"]*num_rows, columns=["PG_1"])
additional_columns.append(PG_1_column)
PG_2_column = pd.DataFrame(["1"]*num_rows, columns=["PG_2"])
additional_columns.append(PG_2_column)
PG_3_column = pd.DataFrame(["0"]*num_rows, columns=["PG_3"])
additional_columns.append(PG_3_column)
PG_4_column = pd.DataFrame(["0"]*num_rows, columns=["PG_4"])
additional_columns.append(PG_4_column)
PG_5_column = pd.DataFrame(["0"]*num_rows, columns=["PG_5"])
additional_columns.append(PG_5_column)

os.chdir("./..")

Temperature_column = pd.DataFrame(["0.998855377"]*num_rows, columns=["Temperature"])
additional_columns.append(Temperature_column)
PolarizationResistance_column = pd.DataFrame(["0"]*num_rows, columns=["Polarization Resistance"])
additional_columns.append(PolarizationResistance_column)

result = pd.concat([vector_data] + additional_columns, axis=1)

result.to_csv('concatenated_data.csv',index=False)

os.system('python ML.py -model xgb -stage 2 -train Data/dataset_2.csv -test concatenated_data.csv -targ "Polarization Resistance" -pth SmSr_Rp') #这里可以改一下K

end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)