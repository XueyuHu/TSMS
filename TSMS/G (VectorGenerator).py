#########################
# Xueyu Hu, March 28th, Generating the Vectors List for Large Chemical Space Simulation
import numpy as np
import csv
from tqdm import tqdm

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

def print_vector(vector):
    """Print the vector elements."""
    print("[" + ", ".join(vector) + "]")

# Configuration
first_elements = ["K","Rb"]
second_elements = ["Cr","Mn"]
#third_elements = ["Al","Dy","Er","Eu","Ga","Gd","Ge","Hf","Ho","In","La","Lu","Mg","Mo","Nb","Nd","Pb","Sb","Sc","Sm","Sn","Ta","Tb","Te","Ti","Tm","V","W","Y","Yb","Zn","Zr"]#['Al','As','Bi','Ca','Cr','Cr','Dy','Er','Eu','Ga','Gd','Ge','Hf','Ho','In','La','Lu','Mg','Mo','Nb','Nd','Pb','Sb','Sc','Sm','Sn','Ta','Tb','Te','Ti','Tm','V','W','Y','Yb','Zn','Zr']

third_elements = ["Al","As","Bi","Ca","Ce","Dy","Er","Eu","Ga","Gd","Ge","Hf","Ho","In","La","Lu","Mg","Mo","Nb","Nd","Pb","Sb","Sc","Sm","Sn","Ta","Tb","Te","Ti","Tm","V","W","Y","Yb","Zn","Zr"]#['Al','As','Bi','Ca','Cr','Cr','Dy','Er','Eu','Ga','Gd','Ge','Hf','Ho','In','La','Lu','Mg','Mo','Nb','Nd','Pb','Sb','Sc','Sm','Sn','Ta','Tb','Te','Ti','Tm','V','W','Y','Yb','Zn','Zr']
vector_size = 32
split_first = 16
split_second = 24

n = 0
element_counts = {}
# Generate and print vectors
for first_element in first_elements:
    for second_element in second_elements:
        index3 = 0
        for third_element in third_elements:
            index3 = index3 + 1
            for num_replacements in range(1, 9):  # From 1 to 8 replacements in the last 8 positions
                vector = create_vector(first_element, second_element, third_element, vector_size, split_first, split_second,
                                               num_replacements)
                print(f"Vector with {num_replacements} replacement(s) by '{third_element}':")
                print_vector(vector)
                n = n+1
                print(n)
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
                            print(f"Vector with {num_replacements} replacement(s) by '{third_element}':")
                            print_vector(vector)
                            n = n+1
                            print(n)