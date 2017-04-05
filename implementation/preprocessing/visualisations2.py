#from dataset import load_data
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import random  
import numpy as np

def load_data(from_url):
    data = np.genfromtxt(from_url,dtype="i4,U256,U256,U256",
                         comments=None,skip_header=1,names=['num','name','p_np','smiles'],
                         converters={k: lambda x: x.decode("utf-8") for k in range(1,4,1)})
    fail_idx = []
    for idx,entry in enumerate(data):
        smiles = entry[3]
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            fail_idx.append(idx)
            continue
    data = np.delete(data,fail_idx)
    print("Fail Count: ", len(fail_idx))
    print("{} molecules used in the calculations".format(len(data)))
    return data

data_url = "./../datasets/bbb_penetration_modified.txt"
#descriptor_output_url = "./../datasets/bbb_penetration_molecular_descriptors.csv"

dataset = load_data(data_url)

random_entries = [dataset[x] for x in random.sample(range(0,2047), 6)]
random_molecules = [Chem.MolFromSmiles(entry[3]) for entry in random_entries]

"""
Draw a chart of random molecules and also create a file containing the chemical 
descriptors for the molecules 
"""
mol_images = Draw.MolsToGridImage(random_molecules, molsPerRow=3, subImgSize=(200,200),legends=[entry[1] for entry in random_entries])
mol_images.save("./../visualisations/random_molecules.png")
#create a file containing the chemical descriptors for the molecules
chem_descriptors = [desc[0] for desc in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(chem_descriptors) 
with open('./random_molecules_descriptors.csv','w') as f:
    f.write("Molecule_Name," + 
            ",".join(["{}".format(name) for name in calculator.descriptorNames]) + 
            "\n")
    for entry in random_entries:
        smiles = entry[3]
        molecule = Chem.MolFromSmiles(smiles)
        print("Calculating chemical descriptors for",smiles)
        f.write( entry[1] + "," +
                ", ".join(["{}".format(value)
                            for value in calculator.CalcDescriptors(molecule)]) +
                "\n")
    print("-> Finished writing to the file")
    print("Using",len(chem_descriptors), "chemical Descriptors")


"""
Generating fingerprint information from the molecules
-> Generate images for each molecule with the smiles-format as its file name
-> Generate a txt file containing all the smiles and their fingerprint format  
"""
print("Generating Fingerprint information")
random_entries = random_entries[:2]
random_molecules = random_molecules[:2]
# Generating images for each molecule 
with open('./random_fingerprints.txt','w') as f:
    for idx in range(0,2):
        filename = random_entries[idx][1] #The molecular name is the filename  
        Draw.MolToFile(random_molecules[idx],"./../visualisations/fingerprint_stub/" + filename + ".png")
        # Generate the entry in the txt file  
        f.write("{:} -> {:}\n".format(filename, random_entries[idx][3]))
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(random_molecules[idx],2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fingerprint,arr)
        fingerprint = arr  
        f.write("{:}\n".format("=" * 10))
        f.write("{:}\n".format(list(fingerprint)))

        

