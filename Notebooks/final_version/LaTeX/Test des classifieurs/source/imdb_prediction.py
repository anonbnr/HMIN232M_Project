#PREDICTION ON IMDB DATASET

#prediction of data on the IMDB Dataset using the LogisticRegression classifier
start_time = time()
print('\nLogisticRegression classifier prediction of IMDB dataset started at {}'.format(datetime.now()))
prediction_results = lr_loaded.predict(imdb_df['Avis'])
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#accuracy, confusion matrix and classification report of the classifier
accuracy = accuracy_score(prediction_results, imdb_df['Score'])
conf = confusion_matrix(imdb_df['Score'], prediction_results)
report = classification_report(imdb_df['Score'], prediction_results)
print('''
Accuracy: {}
Confusion Matrix
{}

Classification Report
{}
'''.format(accuracy, conf, report))

#prediction of data on the IMDB Dataset using the GaussianNB classifier
start_time = time()
print('\nGaussianNB classifier prediction of IMDB dataset started at {}'.format(datetime.now()))
prediction_results = gnb_loaded.predict(imdb_df['Avis'])
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#accuracy, confusion matrix and classification report of the classifier
accuracy = accuracy_score(prediction_results, imdb_df['Score'])
conf = confusion_matrix(imdb_df['Score'], prediction_results)
report = classification_report(imdb_df['Score'], prediction_results)
print('''
Accuracy: {}
Confusion Matrix
{}

Classification Report
{}
'''.format(accuracy, conf, report))
