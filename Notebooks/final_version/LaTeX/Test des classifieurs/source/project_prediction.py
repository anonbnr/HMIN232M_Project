#PREDICTION ON PROJECT DATASET

#prediction of data on the project Dataset using the LogisticRegression classifier
start_time = time()
print('\nLogisticRegression classifier prediction of project dataset started at {}'.format(datetime.now()))
prediction_results = lr_loaded.predict(project_df['Avis'])
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#accuracy, confusion matrix and classification report of the classifier
accuracy = accuracy_score(prediction_results, project_df['Score'])
conf = confusion_matrix(project_df['Score'], prediction_results)
report = classification_report(project_df['Score'], prediction_results)
print('''
Accuracy: {}
Confusion Matrix
{}

Classification Report
{}
'''.format(accuracy, conf, report))

#prediction of data on the project Dataset using the GaussianNB classifier
start_time = time()
print('\nGaussianNB classifier prediction of project dataset started at {}'.format(datetime.now()))
prediction_results = gnb_loaded.predict(project_df['Avis'])
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#accuracy, confusion matrix and classification report of the classifier
accuracy = accuracy_score(prediction_results, project_df['Score'])
conf = confusion_matrix(project_df['Score'], prediction_results)
report = classification_report(project_df['Score'], prediction_results)
print('''
Accuracy: {}
Confusion Matrix
{}

Classification Report
{}
'''.format(accuracy, conf, report))
