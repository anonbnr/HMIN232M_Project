#GRIDSEARCH USING THE ACCURACY METRIC FOR PARAMETERS TUNING

#based on the cross-validation results, using KFold over 10 partitions
#the models LogisticRegression and LinearSVC are best suited for the job
#However, using GaussianNB should also be taken into account, since it's among the most adapted
#for sentiment analysis

#dictionary containing the candidate models that will be used
#for parameters tuning using a GridSearchCV
candidates = {
    'LogisticRegression': models['LogisticRegression'],
    'LinearSVC': models['LinearSVC']
}

#dictionary of the hyperparameters to be tuned for each model
grid_params = {
    'LogisticRegression': {
        'C': np.logspace(-4,4,20),
        'penalty': ['l1','l2']
    },
    'LinearSVC': {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
}

#generation of training/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = TTS_VALIDATION_SIZE,
    test_size = TTS_TEST_SIZE,
    random_state = TTS_SEED
)

#GridSearchCV for every candidate classifier
grid_search_results = []
for name, model in candidates.items():
    #creation of the gridsearch
    grd_sr = GridSearchCV(
        estimator = model,
        param_grid = grid_params[name],
        scoring = GRDSR_SCORING,
        cv = 5,
        n_jobs = -1,
        iid = True,
        return_train_score = True
    )

    #execution of the gridsearch
    start_time = time()
    print('Grid search started at {}'.format(datetime.now()))
    grd_sr.fit(X_train, y_train)
    print('\nTime taken to complete Grid search of {}: {} seconds'.format(name, time() - start_time))
    grd_sr_result = GridSearchResult(name, grd_sr.best_score_, grd_sr.best_estimator_)
    print(grd_sr_result)
    grid_search_results.append(grd_sr_result)

#sorting the results by descending order on the score column of the GridSearchResult objects
grid_search_results = sorted(grid_search_results, key=lambda result: result.score, reverse=True)
print('The best model with the best hyperparameters:\n{}'.format(grid_search_results[0]))
