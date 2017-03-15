#!/usr/bin/python
from flask import Flask
from sklearn.externals import joblib
import sys, getopt
import os

app = Flask(__name__)

"""
Co-ordinate this function to utilise models/voting_classifier.py to return a fully trained model
"""
def train_classifier():
    return NotImplemented

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
def hello():
    return "Hello Cruel World!"

if __name__ == "__main__":
    arguments = parse_arguments(sys.argv[1:])
    if "model_filename" in arguments:
        print("** Using loaded model {:}".format(arguments["model_filename"]))
        voting_clf = joblib.load(arguments["model_filename"])
    else:
        print("** No model passed, will have to train classsifiers\n\t- This should take ~30 minutes")
        voting_clf = train_classifier()
    app.run()

