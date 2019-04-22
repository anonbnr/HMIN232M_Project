#CLASSIFIERS LOADING USING PICKLE

#loading the pipeline containing the trained GaussianNB classifier
gnb_loaded = pickle.load(open(GAUSSIANNB_PATH, 'rb'))

#loading the pipeline containing the trained LogisticRegression classifier
lr_loaded = pickle.load(open(LOGISTICREGRESSION_PATH, 'rb'))
