=====================================================================================================================================
# Machine learning in drug discovery and design: Ensemble techniques for predicting blood brain barrier Penetration of drugs
- Predicting the probability of a drug passing through the blood brain barrier
    * __Investigation/visualisations__
        - [ ] What simple chemical descriptors are important in predicting BBB
            - [ ] Plot estimated relative prior density probability charts for each simple molecular descriptor (Ana et al,BBB_modelling)
    * __Preprocessing__
        * Data preprocessing on the simple molecular descriptors
    * BBB Penetration __Prediction__ implementation - http://b3pp.lasige.di.fc.ul.pt (Nov)
        * Chemical Molecular Descriptors (numerical values that characterize properties of molecules)
            - [x] Simple Molecular Descriptors
            - 2D graphs: [x] Morgan Fingerprints, [ ] Atom Pair 2D Fingerprints
            - 3D representations: Pharmacophore descriptors __time_consuming__
        * Random Forests
            - [ ] Highlight the important chemical descriptor in predicting BBB from the RF
        * Other ML Models ((Bayesian techniques - Ana et al,BBB_modelling),SVM,NN)
    * __Ensemble techniques__ for BBB Penetration as opposed to a specific technique (Apply the same data to different ML models running on
        distributed clusters and compile your results) on Apache Spark
    * Dockerise the application
    __Extensions I__:
    * Deep learning in VS [Explore the benefits of using deep learning vs bayesian methods, as deep learning shows promises in virtual screening (VS)] and if possible show Any implementation using TensorFlow.
    * Feature engineering to help improve the prediction accuracy
    * Prediction Optimizations:
        * Dataset is biased towards positives, look for papers on the best ML technique to handle imbalanced datasets
            - http://www.cs.ox.ac.uk/people/vasile.palade/papers/Class-Imbalance-SVM.pdf
            - http://sci2s.ugr.es/keel/pdf/algorithm/congreso/kubat97addressing.pdf
            - http://www.ele.uri.edu/faculty/he/PDFfiles/ImbalancedLearning.pdf
        * Random feature rotation on random forests http://jmlr.org/papers/volume17/blaser16a/blaser16a.pdf
        * Sequential Minimal Optimisation for training SVMs http://web.iitd.ac.in/~sumeet/tr-98-14.pdf
        * Can we apply boosting techniques to any of our ML algorithms

## Extensions II
- Predicting the ADMET properties of a drug
- Evaluate NAMS for predicting similarity of drugs
    * NAMS Website and implementation - http://nams.lasige.di.fc.ul.pt
- Evaluate structural and semantic similarity techniques separately
    * Semantic similarity - https://en.wikipedia.org/wiki/Semantic_similarity using the CheBI ontology


# BBB Datasets
* ZINC - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1360656/
* Ana et al - http://pubs.acs.org/doi/suppl/10.1021/ci300124c/suppl_file/ci300124c_si_001.txt

# Deliverables and deadlines
[X] Project Proposal (Thursday, 19th October)
[ ] Ethics form (Thursday, 30th November)
[ ] Interim Planning and Investigation Report (Thursday, 30th November)
[ ] Viva (Between 5th December - Friday 11th December)
[ ] Final Deliverables (Thursday 11th May)
    [ ] Project Report and Software
    [ ] Project Log
    [ ] Email records
    [ ] Examiner Report
    [ ] Student Folio Entry
[ ] Project Exhibition day (24th May)


# Notable Companies
* http://www.atomwise.com

# Examples
* https://iwatobipen.wordpress.com/2014/01/23/build-qsar-model-using-rdkit/
* https://github.com/arthuc01/HERG-QSAR
