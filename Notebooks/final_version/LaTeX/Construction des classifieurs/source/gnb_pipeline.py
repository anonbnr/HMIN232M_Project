#PIPELINE CREATION FOR GAUSSIANNB CLASSIFIER
#creating the pipeline instance
gnb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocess, min_df=12, ngram_range=(1, 2))),
    ('to_dense', DenseTransformer()),
    ('clf', GaussianNB())
])

#choosing data and target columns from initial dataset
df_pipeline = df
X = df_pipeline['Avis']
y = df_pipeline['Score']

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
print('GaussianNB classifier pipeline execution started at {}'.format(datetime.now()))
gnb_pipeline.fit(X_train, y_train)
print('\nTime taken to complete pipeline execution: {} seconds'.format(time() - start_time))

#predicting the targets of test data
start_time = time()
print('\nGaussianNB classifier prediction started at {}'.format(datetime.now()))
prediction_result = gnb_pipeline.predict(X_test)
print('\nTime taken to complete prediction: {} seconds'.format(time() - start_time))

#printing the accuracy, confusion matrix and classification report
#of the classifier in the pipeline
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

#SAVING GAUSSIANNB PIPELINE
print('Saving the Gaussian Naive Bayes pipeline')
pickle.dump(gnb_pipeline, open(GAUSSIANNB_PATH, 'wb'))
