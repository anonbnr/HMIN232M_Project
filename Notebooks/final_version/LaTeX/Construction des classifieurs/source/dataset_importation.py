#IMPORTATION OF DATASETS
#importation of opinion dataset
df_avis = import_dataset(DATA_PATH, importation_message="\nDataframe des avis", sep='\t', names=['Avis'])

#importation of scores dataset
df_score = import_dataset(TARGET_PATH, importation_message='\nDataframe des scores', sep='\t', names=['Score'])

#merging of both datasets
df = merge_datasets(df_avis, df_score)

#shuffling of the merged dataset
df = shuffle_dataset(df)
