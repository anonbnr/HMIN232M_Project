#CROSS VALIDATION USING ACCURACY METRIC
#choosing the data (opinions) and target (score) columns in the dataset
X = vectors.toarray()
y = df['Score']

#dictionary containing the models to cross validate using their default parameters
models = {
    'LogisticRegression': LogisticRegression(),
    'SGDClassifier': SGDClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GaussianNB': GaussianNB(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'LinearSVC': LinearSVC()
}
#configuring the parameters used by the cross validation function
k_fold = KFold(n_splits=10, shuffle=True, random_state=CV_SEED)

#cross validation using accuracy metric
#for each defined model
for name, model in models.items():
    start_time = time()
    print('Cross validation started at {}'.format(datetime.now()))
    cv_score = cross_val_score(model, X, y, cv=k_fold, scoring=CV_SCORING)
    output = """
    Time taken to complete cross validation of {}: {} seconds
    Accuracy scores over 10 evaluations: {}
    Mean score: {}
    Standard deviation of scores: {}
    """.format(name, time() - start_time, cv_score, cv_score.mean(), cv_score.std())

    print(output)
