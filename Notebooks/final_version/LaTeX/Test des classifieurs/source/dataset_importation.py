#IMPORTATION OF DATASETS

#importation of project Dataset
test_data = import_dataset(TEST_DATA_PATH, 'Avis Test Dataset', sep='\t', names=['Avis'])
test_labels = import_dataset(TEST_TARGET_PATH, 'Score Test Dataset', sep='\t', names=['Score'])
project_df = merge_datasets(test_data, test_labels)
project_df = shuffle_dataset(project_df)

#importation of IMDB Dataset
imdb_df = import_dataset(IMDB_DATA_PATH, 'IMDB Opinions Dataset', sep='\t', names=['Avis', 'Score'])
