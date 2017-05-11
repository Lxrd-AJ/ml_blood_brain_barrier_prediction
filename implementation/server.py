#!/usr/bin/python
from flask import Flask
from flask import request, jsonify, abort
from sklearn.externals import joblib
from models.voting_classifier import train_voting_clf
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem, DataStructs
from collections import defaultdict 
import sys, getopt
import os
import pandas as pd    
import numpy as np     

app = Flask(__name__)

"""
Co-ordinate this function to utilise models/voting_classifier.py to return a fully trained model
"""
def train_classifier():
    dataset_dir = "./datasets/"
    return train_voting_clf(dataset_dir)


def parse_arguments(argv):
    useage_info = "{:} --model=<model_filename>".format(sys.argv[0])
    arguments = {}
    try:
        opts, args = getopt.getopt(argv, "hm:", ["model="])
    except getopt.GetoptError as err:
        print(err) 
        print(useage_info)
        sys.exit(2)        
    for opt, arg in opts:
        if opt in ("-h","--help"):
            print(useage_info)
            sys.exit(2)
        elif opt in ("-m","--model"):
            arguments["model_filename"] = arg
        else:
            assert False, "unhandled option"
    if len(arguments.keys()) > 0:
        print("Starting the program with the following arguments")
    for arg,val in arguments.items():
        print("-> {:} = {:}".format(arg, val))
    return arguments
            


@app.route("/")
def information():
    return "Blood Brain Barrier Prediction Project!"




@app.route('/api/prediction', methods=['POST'])
def prediction():
    chem_descriptors = [desc[0] for desc in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(chem_descriptors)

    smile = request.get_json()["smile"]
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        abort(400)
        #return "SMILE is malformed, failed to parse it"

    input_descriptors = np.array(list(calculator.CalcDescriptors(molecule)))
    input_ = np.reshape(input_descriptors, (1,-1))
    result = voting_clf.predict(input_)[0]
    probability = voting_clf.predict_proba(input_)[0]
    
    json_result = defaultdict(dict)
    json_result["smile"] = smile
    json_result["category"] = result
    
    for category,proba in zip(voting_clf.classes_,probability):
        json_result["probability"][category] = proba
    
    return jsonify(json_result)




if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])
    if "model_filename" in arguments:
        print("** Using loaded model {:}".format(arguments["model_filename"]))
        voting_clf = joblib.load(arguments["model_filename"])
    else:
        print("** No model passed, will have to train classsifiers\n\t- This should take ~30 minutes")
        voting_clf = train_classifier()
        print("-> Training and Validation of Classifier complete")
        print(voting_clf)
    
    #Fit the classifier on the simple molecular descriptor dataset 
    descriptor_output_url = "./datasets/bbb_penetration_molecular_descriptors.csv"
    mol_df = pd.read_csv(descriptor_output_url)
    x_cols = [col for col in mol_df.columns if col not in ['p_np','smiles']]
    # Simple molecular descriptors
    A = mol_df[x_cols]
    y_a = mol_df.p_np

    print("Fitting the Ensemble classifier on the simple molecular dataset")
    voting_clf.fit(A,y_a)

    app.run()

