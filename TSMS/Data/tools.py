import os
import pandas as pd
import numpy as np

def category_encode(data, categories):
    encoded = []
    for datum in data:
        if datum not in categories:
            raise ValueError('Category not found!: %s' % datum)
        encoded.append(categories.index(datum))
    return np.array(encoded)


# path = 'B'
# element_col = ['Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
#                'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5']
# B_col = ['']
# C1_col = ['Ehull', 'p-band center', 'EV', 'EH', 'd-band center']
# C2_col = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'Volume', 'ShrinkageV',
#           'ShrinkageH', 'FreeVolume', 'SpaceGroup', 'PointGroup', 'International',
#           'SymmetryOperations', 'OverlappingArea', 'OverlappingCenter']
# Y_col = ['Temperature', 'Polarization Resistance']

def txt_to_csv(path, col):
    for file in os.listdir(path):
        if file.endswith('txt') or file.endswith('xlsx'):
            filename = os.path.splitext(file)[0]
            df = pd.read_csv(path+'/'+file, sep='\t', header=None)
            df.dropna(how='all', inplace=True)
            df.fillna(0, inplace=True)
            df.columns = col
            df.to_csv(path+'_'+filename+'.csv', index=False)
# txt_to_csv('Y', element_col+Y_col)

def concat_df(headname):
    DS = pd.DataFrame()
    for file in os.listdir():
        if file.endswith('csv'):
            if headname in file:
                df = pd.read_csv(file)
                DS = pd.concat((DS, df))
        elif file.endswith('xlsx'):
            if headname in file:
                df = pd.read_excel(file)
                DS = pd.concat((DS, df))
    DS.dropna(how='all', inplace=True)
    DS.reset_index(drop=True, inplace=True)
    return DS

# DS2.drop([14, 15, 16], axis=1, inplace=True)
# DS2[10].fillna(0, inplace=True)

# df_c1 = concat_df('C1')

###加指数feature###
data = pd.read_csv('dataset_1_v2.csv')
data.dropna(how='any', inplace=True)

exp = lambda x, y: np.exp(-x[y])
exp_Ehull = exp(data, 'Ehull')
exp_p_band_center = exp(data, 'p-band center')
exp_EV = exp(data, 'EV')
exp_EH = exp(data, 'EH')
exp_d_band_center = exp(data, 'd-band center')
data['exp Ehull'] = exp_Ehull
data['exp p-band center'] = exp_p_band_center
data['exp EV'] = exp_EV
data['exp EH'] = exp_EH
data['exp d-band center'] = exp_d_band_center

# df_c2 = concat_df('C2')
# df_b = concat_df('B_')

###加absfeature###
data['absToleranceFactor'] = np.abs(data['ToleranceFactor']-1)

###加element one-hot###
data[['Element_1_1', 'Element_1_2', 'Element_1_3',
      'Element_4_1', 'Element_4_2', 'Element_4_3', 'Element_4_4', 'Element_4_5']] = int(0)
for ind in data.index:
    x = data['Element_1'][ind]
    y = data['Ratio_1'][ind]
    if x == 'Sr':
        data['Element_1_1'][ind] = int(1)
    elif x == 'Ba' and y == 1:
        data['Element_1_2'][ind] = int(1)
    elif x == 'Ba' and y == 0.5:
        data['Element_1_3'][ind] = int(1)
    else:
        print('error')
# df_b.fillna(int(0), inplace=True)
Element4 = ['Mn', 'Fe', 'Co', 'Ni', 'Cu']
Element4_col = ['Element_4_1', 'Element_4_2', 'Element_4_3', 'Element_4_4', 'Element_4_5']
for ind in data.index:
    x = data['Element_4'][ind]
    ix = Element4.index(x)
    for i in range(len(Element4)):
        if i != ix:
            data[Element4_col[i]][ind] = int(0)
        elif i == ix:
            data[Element4_col[i]][ind] = int(1)
        else:
            print('error')

# data = data.reindex(columns=element_col+C2_col+SG_col+PG_col)


# df_y = concat_df('Y_')

###转换类别变量###
SG = ['P4mm', 'I4/mmm', 'P4/mmm', 'P1', 'C2/m', 'Cmcm', 'P4/nmm']
PG = ['1[C1]', '15[D4h]', '13[C4v]', '8[D2h]', '5[C2h]']
SG_col = ['SG_1', 'SG_2', 'SG_3', 'SG_4', 'SG_5', 'SG_6', 'SG_7']
PG_col = ['PG_1', 'PG_2', 'PG_3', 'PG_4', 'PG_5']

col = data.columns
data = data.reindex(columns=list(col)+SG_col+PG_col)
for ind in data.index:
    sg = data['SpaceGroup'][ind]
    pg = data['PointGroup'][ind]
    isg = SG.index(sg)
    ipg = PG.index(pg)
    for i in range(len(SG)):
        if i != isg:
            data[SG_col[i]][ind] = int(0)
        elif i == isg:
            data[SG_col[i]][ind] = int(1)
    for i in range(len(PG)):
        if i != ipg:
            data[PG_col[i]][ind] = int(0)
        elif i == ipg:
            data[PG_col[i]][ind] = int(1)
################

# DS1 = pd.read_excel('data1.xlsx', header=None).append(pd.read_excel('data2.xlsx', header=None))
# DS_C = pd.merge(df_c1, df_c2, on=('Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
#                                   'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5'), how='left')
# DS = pd.merge(df_b, DS_C, on=('Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
#                               'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5'), how='right')
# DS2 = pd.merge(DS, df_y, on=('Element_1', 'Element_2', 'Element_3', 'Element_4', 'Element_5',
#                             'Ratio_1', 'Ratio_2', 'Ratio_3', 'Ratio_4', 'Ratio_5'), how='right')
# DS.to_csv('dataset_1.csv', index=False)
# DS2.to_csv('dataset_2.csv', index=False)

