#########################
# Xueyu Hu, March 29th, Generating the Vectors List for Large Chemical Space Simulation
import numpy as np
import os
import pandas as pd
import csv
from tqdm import tqdm
from collections import Counter
from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV

class data():
	O={'radius6':1.4,'radius12':0,'electronAffinity':-141,'oxidationstate':-2,'meltingpointSS':55,'boilingpointSS':90,'atomicMass':15.9994,'density':0.00143,'ionizationEnergy':1314,'electronnegativity':3.44,'meltingpointO':55,'chemicalPotential':0}
	Na={'radius6':1.02,'radius12':1.39,'electronAffinity':-53,'oxidationstate':1,'meltingpointSS':371,'boilingpointSS':1156,'atomicMass':22.98977,'density':0.968,'ionizationEnergy':496,'electronnegativity':0.93,'meltingpointO':1132,'chemicalPotential':-147.565}
	Mg={'radius6':0.72,'radius12':1.10,'electronAffinity':-0.1,'oxidationstate':2,'meltingpointSS':923,'boilingpointSS':1363,'atomicMass':24.305,'density':1.738,'ionizationEnergy':738,'electronnegativity':1.31,'meltingpointO':2825,'chemicalPotential':-507.672}
	Al={'radius6':0.535,'radius12':0,'electronAffinity':-43,'oxidationstate':3,'meltingpointSS':933,'boilingpointSS':2792,'atomicMass':26.98154,'density':2.7,'ionizationEnergy':578,'electronnegativity':1.61,'meltingpointO':2050,'chemicalPotential':-700.950}
	K={'radius6':1.38,'radius12':1.64,'electronAffinity':-48,'oxidationstate':1,'meltingpointSS':337,'boilingpointSS':1032,'atomicMass':39.0983,'density':0.856,'ionizationEnergy':419,'electronnegativity':0.82,'meltingpointO':490,'chemicalPotential':-119.692}
	Ca={'radius6':1,'radius12':1.34,'electronAffinity':-2,'oxidationstate':2,'meltingpointSS':1115,'boilingpointSS':1757,'atomicMass':40.078,'density':1.55,'ionizationEnergy':590,'electronnegativity':1,'meltingpointO':2615,'chemicalPotential':-543.558}
	Sc={'radius6':0.745,'radius12':0,'electronAffinity':-18,'oxidationstate':3,'meltingpointSS':1814,'boilingpointSS':3103,'atomicMass':44.95591,'density':2.985,'ionizationEnergy':633,'electronnegativity':1.36,'meltingpointO':2400,'chemicalPotential':-824.905}
	Ti={'radius6':0.605,'radius12':0,'electronAffinity':-8,'oxidationstate':4,'meltingpointSS':1941,'boilingpointSS':3560,'atomicMass':47.867,'density':4.507,'ionizationEnergy':659,'electronnegativity':1.54,'meltingpointO':2020,'chemicalPotential':-785.128}
	V={'radius6':0.54,'radius12':0,'electronAffinity':-51,'oxidationstate':5,'meltingpointSS':2183,'boilingpointSS':3680,'atomicMass':50.9415,'density':6.11,'ionizationEnergy':651,'electronnegativity':1.63,'meltingpointO':670,'chemicalPotential':-584.745}
	Cr={'radius6':0.44,'radius12':0,'electronAffinity':-64,'oxidationstate':6,'meltingpointSS':2180,'boilingpointSS':2944,'atomicMass':51.9961,'density':7.14,'ionizationEnergy':653,'electronnegativity':1.66,'meltingpointO':185,'chemicalPotential':-370.126}
	Mn={'radius6':0.58,'radius12':0,'electronAffinity':-0.1,'oxidationstate':3,'meltingpointSS':1519,'boilingpointSS':2334,'atomicMass':54.93805,'density':7.47,'ionizationEnergy':717,'electronnegativity':1.55,'meltingpointO':1875,'chemicalPotential':-366.471}
	Fe={'radius6':0.55,'radius12':0,'electronAffinity':-16,'oxidationstate':3,'meltingpointSS':1811,'boilingpointSS':3134,'atomicMass':55.845,'density':7.874,'ionizationEnergy':763,'electronnegativity':1.83,'meltingpointO':1594,'chemicalPotential':-295.384}
	Co={'radius6':0.545,'radius12':0,'electronAffinity':-64,'oxidationstate':3,'meltingpointSS':1768,'boilingpointSS':3200,'atomicMass':58.9332,'density':8.9,'ionizationEnergy':760,'electronnegativity':1.88,'meltingpointO':1805,'chemicalPotential':-191.238}
	Ni={'radius6':0.56,'radius12':0,'electronAffinity':-112,'oxidationstate':3,'meltingpointSS':1728,'boilingpointSS':3186,'atomicMass':58.6934,'density':8.908,'ionizationEnergy':737,'electronnegativity':1.91,'meltingpointO':1955,'chemicalPotential':-159.648}
	Cu={'radius6':0.54,'radius12':0,'electronAffinity':-118,'oxidationstate':3,'meltingpointSS':1358,'boilingpointSS':3200,'atomicMass':63.546,'density':8.92,'ionizationEnergy':746,'electronnegativity':1.9,'meltingpointO':1236,'chemicalPotential':-76.978}
	Zn={'radius6':0.74,'radius12':0,'electronAffinity':-0.1,'oxidationstate':2,'meltingpointSS':693,'boilingpointSS':1180,'atomicMass':65.38,'density':7.14,'ionizationEnergy':906,'electronnegativity':1.65,'meltingpointO':1975,'chemicalPotential':-261.407}
	Ga={'radius6':0.62,'radius12':0,'electronAffinity':-29,'oxidationstate':3,'meltingpointSS':303,'boilingpointSS':2477,'atomicMass':69.723,'density':5.904,'ionizationEnergy':579,'electronnegativity':1.81,'meltingpointO':1795,'chemicalPotential':-403.356}
	Ge={'radius6':0.53,'radius12':0,'electronAffinity':-119,'oxidationstate':4,'meltingpointSS':1211,'boilingpointSS':3093,'atomicMass':72.64,'density':5.323,'ionizationEnergy':762,'electronnegativity':2.01,'meltingpointO':1116,'chemicalPotential':-410.699}
	As={'radius6':0.46,'radius12':0,'electronAffinity':-78,'oxidationstate':5,'meltingpointSS':1090,'boilingpointSS':887,'atomicMass':74.9216,'density':5.727,'ionizationEnergy':947,'electronnegativity':2.18,'meltingpointO':312.2,'chemicalPotential':-222.203}
	Rb={'radius6':1.52,'radius12':1.72,'electronAffinity':-47,'oxidationstate':1,'meltingpointSS':312,'boilingpointSS':961,'atomicMass':85.4678,'density':1.532,'ionizationEnergy':403,'electronnegativity':0.82,'meltingpointO':500,'chemicalPotential':-111.682}
	Sr={'radius6':1.18,'radius12':1.44,'electronAffinity':-5,'oxidationstate':2,'meltingpointSS':1050,'boilingpointSS':1655,'atomicMass':87.62,'density':2.63,'ionizationEnergy':550,'electronnegativity':0.95,'meltingpointO':2460,'chemicalPotential':-504.711}
	Y={'radius6':0.9,'radius12':0,'electronAffinity':-30,'oxidationstate':3,'meltingpointSS':1799,'boilingpointSS':3618,'atomicMass':88.90585,'density':4.472,'ionizationEnergy':600,'electronnegativity':1.22,'meltingpointO':2704,'chemicalPotential':-824.705}
	Zr={'radius6':0.72,'radius12':0,'electronAffinity':-41,'oxidationstate':4,'meltingpointSS':2128,'boilingpointSS':4682,'atomicMass':91.224,'density':6.511,'ionizationEnergy':640,'electronnegativity':1.33,'meltingpointO':2680,'chemicalPotential':-933.198}
	Nb={'radius6':0.64,'radius12':0,'electronAffinity':-86,'oxidationstate':5,'meltingpointSS':2750,'boilingpointSS':5017,'atomicMass':92.90638,'density':8.57,'ionizationEnergy':652,'electronnegativity':1.6,'meltingpointO':1510,'chemicalPotential':-757.921}
	Mo={'radius6':0.59,'radius12':0,'electronAffinity':-72,'oxidationstate':6,'meltingpointSS':2896,'boilingpointSS':4912,'atomicMass':95.96,'density':10.28,'ionizationEnergy':684,'electronnegativity':2.16,'meltingpointO':800,'chemicalPotential':-523.646}
	In={'radius6':0.8,'radius12':0,'electronAffinity':-29,'oxidationstate':3,'meltingpointSS':430,'boilingpointSS':2345,'atomicMass':114.818,'density':7.31,'ionizationEnergy':558,'electronnegativity':1.78,'meltingpointO':1910,'chemicalPotential':-319.764}
	Sn={'radius6':0.69,'radius12':0,'electronAffinity':-107,'oxidationstate':4,'meltingpointSS':505,'boilingpointSS':2875,'atomicMass':118.71,'density':7.31,'ionizationEnergy':709,'electronnegativity':1.96,'meltingpointO':2000,'chemicalPotential':-399.976}
	Sb={'radius6':0.6,'radius12':0,'electronAffinity':-103,'oxidationstate':5,'meltingpointSS':904,'boilingpointSS':1860,'atomicMass':121.76,'density':6.697,'ionizationEnergy':834,'electronnegativity':2.05,'meltingpointO':656,'chemicalPotential':-287.544}
	Te={'radius6':0.56,'radius12':0,'electronAffinity':-190,'oxidationstate':6,'meltingpointSS':723,'boilingpointSS':1261,'atomicMass':127.6,'density':6.24,'ionizationEnergy':869,'electronnegativity':2.1,'meltingpointO':733,'chemicalPotential':-161.153}
	Cs={'radius6':1.67,'radius12':1.88,'electronAffinity':-46,'oxidationstate':1,'meltingpointSS':302,'boilingpointSS':944,'atomicMass':132.90545,'density':1.879,'ionizationEnergy':376,'electronnegativity':0.79,'meltingpointO':490,'chemicalPotential':-116.104}
	Ba={'radius6':1.35,'radius12':1.61,'electronAffinity':-14,'oxidationstate':2,'meltingpointSS':1000,'boilingpointSS':2143,'atomicMass':137.327,'density':3.51,'ionizationEnergy':503,'electronnegativity':0.89,'meltingpointO':1920,'chemicalPotential':-468.011}
	La={'radius6':1.032,'radius12':1.36,'electronAffinity':-48,'oxidationstate':3,'meltingpointSS':1193,'boilingpointSS':3737,'atomicMass':138.90547,'density':6.146,'ionizationEnergy':538,'electronnegativity':1.1,'meltingpointO':2320,'chemicalPotential':-772.207}
	Ce={'radius6':0.87,'radius12':1.34,'electronAffinity':-50,'oxidationstate':4,'meltingpointSS':1071,'boilingpointSS':3633,'atomicMass':140.116,'density':6.689,'ionizationEnergy':534,'electronnegativity':1.12,'meltingpointO':2400,'chemicalPotential':-908.255}
	Pr={'radius6':0.99,'radius12':1.32,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1204,'boilingpointSS':3563,'atomicMass':140.90765,'density':6.64,'ionizationEnergy':527,'electronnegativity':1.13,'meltingpointO':2500,'chemicalPotential':-778.997}
	Nd={'radius6':0.983,'radius12':1.31,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1294,'boilingpointSS':3373,'atomicMass':144.242,'density':7.01,'ionizationEnergy':533,'electronnegativity':1.14,'meltingpointO':2272,'chemicalPotential':-779.983}
	Sm={'radius6':0.958,'radius12':1.28,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1345,'boilingpointSS':2076,'atomicMass':150.36,'density':7.353,'ionizationEnergy':545,'electronnegativity':1.17,'meltingpointO':2350,'chemicalPotential':-784.204}
	Eu={'radius6':0.947,'radius12':1.28,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1095,'boilingpointSS':1800,'atomicMass':151.964,'density':5.244,'ionizationEnergy':547,'electronnegativity':1.2,'meltingpointO':623,'chemicalPotential':-689.959}
	Gd={'radius6':0.938,'radius12':1.27,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1586,'boilingpointSS':3523,'atomicMass':157.25,'density':7.901,'ionizationEnergy':593,'electronnegativity':1.2,'meltingpointO':2340,'chemicalPotential':-793.802}
	Tb={'radius6':0.923,'radius12':1.25,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1629,'boilingpointSS':3503,'atomicMass':158.92535,'density':8.219,'ionizationEnergy':566,'electronnegativity':1.2,'meltingpointO':2340,'chemicalPotential':-807.674}
	Dy={'radius6':0.912,'radius12':1.24,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1685,'boilingpointSS':2840,'atomicMass':162.5,'density':8.551,'ionizationEnergy':573,'electronnegativity':1.22,'meltingpointO':2380,'chemicalPotential':-801.290}
	Ho={'radius6':0.901,'radius12':1.23,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1747,'boilingpointSS':2973,'atomicMass':164.93032,'density':8.795,'ionizationEnergy':581,'electronnegativity':1.23,'meltingpointO':2415,'chemicalPotential':-813.591}
	Er={'radius6':0.89,'radius12':1.22,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1770,'boilingpointSS':3141,'atomicMass':167.259,'density':9.066,'ionizationEnergy':589,'electronnegativity':1.24,'meltingpointO':2400,'chemicalPotential':-820.766}
	Tm={'radius6':0.88,'radius12':1.21,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1818,'boilingpointSS':2223,'atomicMass':168.93421,'density':9.321,'ionizationEnergy':597,'electronnegativity':1.25,'meltingpointO':2341,'chemicalPotential':-810.485}
	Yb={'radius6':0.868,'radius12':1.2,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1092,'boilingpointSS':1469,'atomicMass':173.054,'density':6.57,'ionizationEnergy':603,'electronnegativity':1.1,'meltingpointO':2439,'chemicalPotential':-782.429}
	Lu={'radius6':0.861,'radius12':1.19,'electronAffinity':-50,'oxidationstate':3,'meltingpointSS':1936,'boilingpointSS':3675,'atomicMass':174.9668,'density':9.841,'ionizationEnergy':524,'electronnegativity':1.27,'meltingpointO':2490,'chemicalPotential':-810.622}
	Hf={'radius6':0.71,'radius12':0,'electronAffinity':-0.1,'oxidationstate':4,'meltingpointSS':2506,'boilingpointSS':4876,'atomicMass':178.49,'density':13.31,'ionizationEnergy':659,'electronnegativity':1.3,'meltingpointO':2810,'chemicalPotential':-954.156}
	Ta={'radius6':0.64,'radius12':0,'electronAffinity':-31,'oxidationstate':5,'meltingpointSS':3290,'boilingpointSS':5731,'atomicMass':180.94788,'density':16.65,'ionizationEnergy':761,'electronnegativity':1.5,'meltingpointO':1900,'chemicalPotential':-829.533}
	W={'radius6':0.6,'radius12':0,'electronAffinity':-79,'oxidationstate':6,'meltingpointSS':3695,'boilingpointSS':5828,'atomicMass':183.84,'density':19.25,'ionizationEnergy':770,'electronnegativity':2.36,'meltingpointO':1472,'chemicalPotential':-616.947}
	Pb={'radius6':0.775,'radius12':0,'electronAffinity':-35,'oxidationstate':4,'meltingpointSS':601,'boilingpointSS':2022,'atomicMass':207.2,'density':11.34,'ionizationEnergy':716,'electronnegativity':2.33,'meltingpointO':290,'chemicalPotential':-113.590}
	Bi={'radius6':0.76,'radius12':1.17,'electronAffinity':-91,'oxidationstate':5,'meltingpointSS':544,'boilingpointSS':1837,'atomicMass':208.9804,'density':9.78,'ionizationEnergy':703,'electronnegativity':2.02,'meltingpointO':824,'chemicalPotential':-167.088}

cwd = os.getcwd()
costdb_path = "%s\costdb_elements_202311.csv" %cwd
costdb = CostDBCSV(costdb_path)
costanalyzer = CostAnalyzer(costdb)

variables = {'electronAffinity':{},'oxidationstate':{},'meltingpointSS':{},'boilingpointSS':{},'atomicMass':{},'density':{},'ionizationEnergy':{},'electronnegativity':{},'meltingpointO':{},'chemicalPotential':{}}

#vector = ["Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Na", "Cr", "Cr", "Cr", "Cr", "Cr", "Cr", "Cr", "Cr", "Zr", "Zr", "Zr", "Zr", "Zr", "Cr", "Cr", "Cr"]
#vector = ["Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Ba", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Mn", "Al"]
A = ["K","Rb","Cs","Ba","Na","Mg","Ca","Sr","La","Pr","Nd","Sm"]
B1 = ["Cr","Mn","Fe","Co","Ni","Cu"]
B2 = ["Al","As","Bi","Ce","Dy","Er","Eu","Ga","Gd","Ge","Hf","Ho","In","Lu","Mo","Nb","Pb","Sb","Sc","Sn","Ta","Tb","Te","Ti","Tm","V","W","Y","Yb","Zn","Zr"]
A1 = ["K","Rb","Cs","Ba"]
A2 = ["Na","Mg","Ca","Sr","La","Pr","Nd","Sm"]

def create_vector(first_element="Na", second_element="Cr", third_element="Al", vector_size=32, split_first=16, split_second=24, num_replacements=0):
    """Create a vector with specified elements and replacements."""
    vector = np.empty(vector_size, dtype=object)  # Create an empty vector
    vector[:split_first] = first_element  # Fill the first part with the first element
    vector[split_first:split_second] = second_element  # Fill the middle part with the second element
    vector[split_second:] = second_element  # Fill the last part with the second element
    if num_replacements > 0:  # If replacements are specified, replace the last part's elements accordingly
        vector[split_second:split_second + num_replacements] = third_element
    return vector

def create_vector2(first_element="Na", second_element="Cr", third_element="Al", fourth_element="As", vector_size=32, split_first=16, split_second=24, num_replacements=0, num_replacements2=1):
    """Create a vector with specified elements and replacements."""
    vector = np.empty(vector_size, dtype=object)  # Create an empty vector
    vector[:split_first] = first_element  # Fill the first part with the first element
    vector[split_first:split_second] = second_element  # Fill the middle part with the second element
    vector[split_second:] = second_element  # Fill the last part with the second element
    if num_replacements > 0:  # If replacements are specified, replace the last part's elements accordingly
        vector[split_second:split_second + num_replacements] = third_element
        vector[split_second:split_second + num_replacements2] = fourth_element
    return vector

def Enlargement(vector):
    element_counts = {}
    for element in vector:
        if element in element_counts:
            element_counts[element] += 1
        else:
            element_counts[element] = 1
    counts_A = 0
    counts_B1 = 0
    counts_B2 = 0
    Element_1_3 = 0  # double with 1
    Element_4_1 = 0  # Cr and Mn
    Element_4_2 = 0  # Fe
    Element_4_3 = 0  # Co
    Element_4_4 = 0  # Ni
    Element_4_5 = 0  # Cu
    for element, count in element_counts.items():
        if element in A1:
            counts_A += count
            Ratio_1 = 1
            Ratio_2 = 0
            Element_1_1 = 0
            Element_1_2 = 1
        if element in A2:
            counts_A += count
            Ratio_1 = 0
            Ratio_2 = 1
            Element_1_1 = 1
            Element_1_2 = 0
        if element in B1:
            counts_B1 += count
        if element in B2:
            counts_B2 += count
        if element == 'Cr':
            Element_4_1 = 1
        elif element == "Mn":
            Element_4_1 = 1
        elif element == "Fe":
            Element_4_2 = 1
        elif element == "Co":
            Element_4_3 = 1
        elif element == "Ni":
            Element_4_4 = 1
        elif element == "Cu":
            Element_4_5 = 1

    #Ratio_1 = 1  # radius larger than Sr with 1, otherwise 0
    #Ratio_2 = 0  # radius smaller and equals to Sr with 1, otherwise 0
    Ratio_3 = 0  # double with 1 and single with 0
    Ratio_4 = counts_B1 / (counts_B1 + counts_B2)
    Ratio_5 = counts_B2 / (counts_B1 + counts_B2)

    for i in range(1, 6):
        vector.append(locals()[f'Ratio_{i}'])

    radius_dict = {}
    for i in range(1, 13):
        radius_dict[f'radius_{i}'] = 0
        #locals()[f'radius_{i}'] = 0
    for element, count in element_counts.items():
        if element in A:
            radius_dict['radius_1'] += getattr(data(), element)['radius12'] * count  # 需要除一下
            radius_dict['radius_5'] += getattr(data(), element)['radius12'] * count  # 就这样吧 无所谓了
        if element in B1:
            radius_dict['radius_2'] += getattr(data(), element)['radius6'] * count
            radius_dict['radius_6'] += getattr(data(), element)['radius6'] * count
        if element in B2:
            radius_dict['radius_2'] += getattr(data(), element)['radius6'] * count
            radius_dict['radius_4'] += getattr(data(), element)['radius6'] * count
    vector.append(radius_dict['radius_1'] / counts_A)
    vector.append(radius_dict['radius_2'] / (counts_B1 + counts_B2))
    vector.append(radius_dict['radius_3'])
    vector.append(radius_dict['radius_4'] / (counts_B1 + counts_B2))
    vector.append(radius_dict['radius_5'] / counts_A)
    vector.append(radius_dict['radius_6'] / (counts_B1 + counts_B2))
    vector.append(radius_dict['radius_5'] / counts_A + radius_dict['radius_6'] / (counts_B1 + counts_B2))  # 7
    vector.append((radius_dict['radius_1'] / counts_A) / (radius_dict['radius_2'] / (counts_B1 + counts_B2)))
    vector.append((radius_dict['radius_5'] / counts_A) / (radius_dict['radius_6'] / (counts_B1 + counts_B2)))
    vector.append(radius_dict['radius_3'] / (radius_dict['radius_1'] / counts_A))
    #vector.append(0)
    vector.append((radius_dict['radius_4'] / (counts_B1 + counts_B2)) / (radius_dict['radius_2'] / (counts_B1 + counts_B2)))
    vector.append((radius_dict['radius_3'] + (radius_dict['radius_4'] / (counts_B1 + counts_B2))) / (
                (radius_dict['radius_5'] / counts_A) + (radius_dict['radius_6'] / (counts_B1 + counts_B2))))
    # for i in range(1,13):
    #	print(eval(f'radius_{i}'))
    # for indexD in range(len(listData)):
    #	for i in range(1,13):
    #		locals()[f'%s_{i}' %listData[indexD]] = 0

    for i in variables:
        for j in range(1, 7):
            variables[i][j] = 0
    for element, count in element_counts.items():
        for i in variables:
            if element in A:
                variables[i][1] += getattr(data(), element)[i] * count  # 需要除一下
                variables[i][5] += getattr(data(), element)[i] * count  # 就这样吧 无所谓了
            if element in B1:
                variables[i][2] += getattr(data(), element)[i] * count
                variables[i][6] += getattr(data(), element)[i] * count
            if element in B2:
                variables[i][2] += getattr(data(), element)[i] * count
                variables[i][4] += getattr(data(), element)[i] * count
    for i in variables:
        vector.append(variables[i][1] / counts_A)
        vector.append(variables[i][2] / (counts_B1 + counts_B2))
        vector.append(variables[i][3])
        vector.append(variables[i][4] / (counts_B1 + counts_B2))
        vector.append(variables[i][5] / counts_A)
        vector.append(variables[i][6] / (counts_B1 + counts_B2))
        vector.append(variables[i][5] / counts_A + variables[i][6] / (counts_B1 + counts_B2))  # 7
        vector.append((variables[i][1] / counts_A) / (variables[i][2] / (counts_B1 + counts_B2)))
        vector.append((variables[i][5] / counts_A) / (variables[i][6] / (counts_B1 + counts_B2)))
        vector.append(variables[i][3] / (variables[i][1] / counts_A))
        vector.append((variables[i][4] / (counts_B1 + counts_B2)) / (variables[i][2] / (counts_B1 + counts_B2)))
        vector.append((variables[i][3] + (variables[i][4] / (counts_B1 + counts_B2))) / (
                    (variables[i][5] / counts_A) + (variables[i][6] / (counts_B1 + counts_B2))))

    vector.append((radius_dict['radius_1'] / counts_A + 1.4) / ((radius_dict['radius_2'] / (counts_B1 + counts_B2) + 1.4) * 2 ** 0.5))  # TF
    vector.append(radius_dict['radius_2'] / (counts_B1 + counts_B2) / 1.4)  # OF
    vector.append((variables['atomicMass'][1] / counts_A + variables['atomicMass'][2] / (counts_B1 + counts_B2) +
                   getattr(data(), 'O')['atomicMass'] * 3) * (
                              (radius_dict['radius_1'] / counts_A) * (radius_dict['radius_2'] / (counts_B1 + counts_B2))) ** -0.5)  # density
    vector.append(
        (6 - variables['oxidationstate'][1] / counts_A - variables['oxidationstate'][2] / (counts_B1 + counts_B2)) / 2)
    vector.append(abs(((radius_dict['radius_1'] / counts_A + 1.4) / ((radius_dict['radius_2'] / (counts_B1 + counts_B2) + 1.4) * 2 ** 0.5)) - 1))

    #Element_1_1 = 0  # equals or smaller than Sr with 1
    #Element_1_2 = 1  # larger than Sr with 1
    #Element_1_3 = 0  # double with 1
    #Element_4_1 = 1  # Cr and Mn
    #Element_4_2 = 0  # Fe
    #Element_4_3 = 0  # Co
    #Element_4_4 = 0  # Ni
    #Element_4_5 = 0  # Cu
    for i in range(1, 4):
        vector.append(locals()[f'Element_1_{i}'])
    for i in range(1, 6):
        vector.append(locals()[f'Element_4_{i}'])
    return vector

def print_vector(vector):
    """Print the vector elements."""
    print("[" + ", ".join(vector) + "]")

def Cost(vector):
    element_counts = Counter(vector)
    formatted_string = "".join(f"{element}{count}" for element, count in element_counts.items()) + "O48"
    cost = costanalyzer.get_cost_per_kg(formatted_string)
    return(cost)


# Configuration
first_elements = ["Ba"]#,"K","Rb","Cs","Ba","Na","Mg","Ca","Sr","La","Pr","Nd","Sm"]
second_elements = ["Cr","Mn","Fe","Co","Ni","Cu"]

third_elements = ["Al","As","Bi","Ce","Dy","Er","Eu","Ga","Gd","Ge","Hf","Ho","In","Lu","Mo","Nb","Pb","Sb","Sc","Sn","Ta","Tb","Te","Ti","Tm","V","W","Y","Yb","Zn","Zr"]#['Al','As','Bi','Ca','Cr','Cr','Dy','Er','Eu','Ga','Gd','Ge','Hf','Ho','In','La','Lu','Mg','Mo','Nb','Nd','Pb','Sb','Sc','Sm','Sn','Ta','Tb','Te','Ti','Tm','V','W','Y','Yb','Zn','Zr']

#third_elements = ["Al","As","Bi","Ca","Ce","Dy","Er","Eu","Ga","Gd","Ge","Hf","Ho","In","La","Lu","Mg","Mo","Nb","Nd","Pb","Sb","Sc","Sm","Sn","Ta","Tb","Te","Ti","Tm","V","W","Y","Yb","Zn","Zr"]#['Al','As','Bi','Ca','Cr','Cr','Dy','Er','Eu','Ga','Gd','Ge','Hf','Ho','In','La','Lu','Mg','Mo','Nb','Nd','Pb','Sb','Sc','Sm','Sn','Ta','Tb','Te','Ti','Tm','V','W','Y','Yb','Zn','Zr']
vector_size = 32
split_first = 16
split_second = 24
index3 = 0
n = 0
element_counts = {}
######################## Initialization of the csv file ##################################
First_Row = []
for i in range(1,6):
    First_Row.append(f'Element_{i}')
for i in range(1,6):
    First_Row.append(f'Ratio_{i}')
for i in range(1,13):
    First_Row.append(f'radius_{i}')
for j in variables:
    for i in range(1,13):
        First_Row.append(f'%s_{i}' %j)
for i in (["ToleranceFactor","OctahedralFactor","MaterialDensity","delta","absToleranceFactor"]):
    First_Row.append(i)
for i in range(1,4):
    First_Row.append(f'Element_1_{i}')
for i in range(1,6):
    First_Row.append(f'Element_4_{i}')
for i in (["Ehull","p-band center","EV","EH","d-band center","exp Ehull","exp p-band center","exp EV","exp EH","exp d-band center","a","b","c","alpha","beta","gamma","Volume","ShrinkageV","ShrinkageH","FreeVolume","SpaceGroup","PointGroup","International","SymmetryOperations","OverlappingArea","OverlappingCenter"]):
    First_Row.append(i)
for i in range(1,8):
    First_Row.append(f'SG_{i}')
for i in range(1,6):
    First_Row.append(f'PG_{i}')
with open('vector_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(First_Row)
######################## End of Initialization ###########################################
# Generate and print vectors
for first_element in tqdm(first_elements):
    for second_element in tqdm(second_elements):
        index3 = 0
        for third_element in tqdm(third_elements):
            for num_replacements in range(1, 9):  # From 1 to 8 replacements in the last 8 positions
                vector = create_vector(first_element, second_element, third_element, vector_size, split_first, split_second,
                                               num_replacements)
                #print(f"Vector with {num_replacements} replacement(s) by '{third_element}':")
                List_Vector = vector.tolist()
                #print(List_Vector)
                with open('element.csv','a',newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(List_Vector)
                ##cost = Cost(List_Vector)
                ##with open('cost.csv', 'a', newline='') as file:
                ##    writer = csv.writer(file)
                ##    writer.writerow([cost])
                #Output = Enlargement(List_Vector)
                #Output[5:32] = []
                #Output.extend([0]*38)
                #with open('vector_data.csv', 'a', newline='') as file:
                #    writer = csv.writer(file)
                #    writer.writerow(Output)
                #print(Output)
                #n = n+1
                #print(n)
                #for element in vector:
                #    if element in element_counts:
                #        element_counts[element] += 1
                #    else:
                #        element_counts[element] = 1
                #print(element_counts)

                #print("This is the first trials")  # Add an empty line for better readability between vectors
                if num_replacements >= 2:
                    for fourth_element in third_elements[index3:]:
                        if fourth_element == third_element:
                            continue
                        for num_replacements2 in range(1, num_replacements):
                            vector = create_vector2(first_element, second_element, third_element, fourth_element, vector_size, split_first, split_second, num_replacements, num_replacements2)
                            List_Vector = vector.tolist()
                            with open('element.csv','a',newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(List_Vector)
                            ##cost = Cost(List_Vector)
                            ##with open('cost.csv', 'a', newline='') as file:
                            ##    writer = csv.writer(file)
                            ##    writer.writerow([cost])
                            #Output = Enlargement(List_Vector)
                            #Output[5:32] = []
                            #Output.extend([0] * 38)
                            #with open('vector_data.csv', 'a', newline='') as file:
                            #    writer = csv.writer(file)
                            #    writer.writerow(Output)
                            #print(f"Vector with {num_replacements} replacement(s) by '{third_element}':")
                            #print_vector(vector)
                            #print(vector)
                            #n = n+1
                            #print(n)