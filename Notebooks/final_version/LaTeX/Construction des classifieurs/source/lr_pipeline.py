#PIPELINE CREATION FOR LOGISTICREGRESSION CLASSIFIER
#creating the pipeline instance
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess, min_df=12, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=11.288378916846883, penalty='l2'))
])

#choosing data and target columns from initial dataset
X = df['Avis']
y = df['Score']

#generating the training/test sets from the initial dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = TTS_VALIDATION_SIZE,
    test_size = TTS_TEST_SIZE,
    random_state = TTS_SEED
)

#learning the model using the pipeline
start_time = time()
print('LogisticRegression classifier pipeline execution started at {}'.format(datetime.now()))
lr_pipeline.fit(X_train, y_train)
print('\nTime taken to complete pipeline execution: {} seconds'.format(time() - start_time))

#predicting the targets of test data
start_time = time()
print('\nLogisticRegression classifier prediction started at {}'.format(datetime.now()))
prediction_result = lr_pipeline.predict(X_test)
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#accuracy, confusion matrix and classification report of the classifier
accuracy = accuracy_score(prediction_result, y_test)
conf = confusion_matrix(y_test, prediction_result)
report = classification_report(y_test, prediction_result)
print('''
Accuracy: {}
Confusion Matrix
{}

Classification Report
{}
'''.format(accuracy, conf, report))

#SAVING LOGISTICREGRESSION PIPELINE
print('Saving the Logistic Regression pipeline')
pickle.dump(lr_pipeline, open(LOGISTICREGRESSION_PATH, 'wb'))
