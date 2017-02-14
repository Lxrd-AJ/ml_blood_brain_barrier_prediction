import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

def load_data(from_url,desc_url):
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
    calc_descriptors(desc_url, data)

def calc_descriptors(file_url,data):
    chem_descriptors = [desc[0] for desc in Descriptors._descList]
    
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(chem_descriptors)
    print("Using",len(chem_descriptors), "chemical Descriptors")

    with open(file_url,'w') as f:
        f.write("smiles," +
                ", ".join(["{}".format(name) for name in calculator.descriptorNames]) +
                ",p_np\n")
        for entry in data:
            smiles = entry[3]
            molecule = Chem.MolFromSmiles(smiles)
            print("Calculating chemical descriptors for",smiles)
            f.write( smiles + "," +
                    ", ".join(["{}".format(value)
                               for value in calculator.CalcDescriptors(molecule)]) +
                    ",{}\n".format(entry[2]))
