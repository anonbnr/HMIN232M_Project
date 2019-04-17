import re
import unicodedata
import warnings

import pandas as pd
import numpy as np
import contractions
import inflect

from collections import defaultdict

from nltk import pos_tag
from nltk import punkt
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.utils import shuffle
from sklearn.base import TransformerMixin

warnings.filterwarnings('ignore', category=FutureWarning)

#DEFINITION OF SOME OF THE RESOURCES USED THROUGHOUT THIS NOTEBOOK
DATA_PATH = 'Datasets/dataset.csv'
TARGET_PATH = 'Datasets/labels.csv'
TEST_DATA_PATH = 'Datasets/test_data.csv'
TEST_TARGET_PATH = 'Datasets/test_labels.csv'
GAUSSIANNB_PATH = 'Classifiers/gaussianNB.pkl'
LOGISTICREGRESSION_PATH = 'Classifiers/logisticRegression.pkl'

#stop words set adapted to the context of the dataset
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS_EXCEPTIONS = set(('no', 'nor', 'not',))
STOP_WORDS_ADDITIONS = [
    'film', 'films', 'filmed',
    'movie', 'movies',
    'character', 'characters',
    'story', 'stories',
    'scene', 'scenes',
    'actor', 'actors', 'actress', 'actresses', 'act', 'acts', 'acted', 'acting'
    'direct', 'directs', 'directed', 'directing', 'director', 'directors',
    'script', 'scripts',
    'plot', 'plots'
]
STOP_WORDS.update(STOP_WORDS_ADDITIONS)
STOP_WORDS = STOP_WORDS - STOP_WORDS_EXCEPTIONS

#POS-TAG dictionary that will be used during the lemmatization process
POS_TAG_MAP = defaultdict(lambda : wn.NOUN)
POS_TAG_MAP['J'] = wn.ADJ
POS_TAG_MAP['V'] = wn.VERB
POS_TAG_MAP['R'] = wn.ADV

#parameters used by the cross validation score function
CV_SEED = 7 #seed used for random selection of partitions during cross validation
CV_SCORING = 'accuracy'

#parameters used by the training/set generator
TTS_VALIDATION_SIZE = 0.3 #30% of dataset used for training
TTS_TEST_SIZE = 1 - TTS_VALIDATION_SIZE #70% of dataset used for testing
TTS_SEED = 30 #seed used for random selection of training/test sets

#parameters used by the gridsearch function
GRDSR_SCORING = 'accuracy'

#DEFINITION OF IMPORTATION, MERGING AND SHUFFLING FUNCTIONS OF DATASETS
def import_dataset(dataset_path, importation_message=None, sep='\t', names=None):
    """
    imports a dataset from a given path

    returns the dataframe containing the imported dataset"""

    print("\n{}".format(importation_message))
    df = pd.read_csv(dataset_path, sep=sep, header=None, names=names, encoding='utf-8')
    print('Size : {}'.format(df.shape))
    print('Head of imported dataset :')
    display(df.head())

    return df

def merge_datasets(df1, df2):
    """
    merges datasets contained within dataframes df1 and df2

    returns a new dataframe containing the merged datasets"""

    df = df1.join(df2)

    print('Size : {}'.format(df.shape))
    print('Head of merged dataset :')
    display(df.head())

    return df

def shuffle_dataset(df):
    """
    shuffles dataset entries and reset indexes

    returns a new dataframe containing the shuffled dataset with the reset indexes"""

    shuffled_df = shuffle(df)
    shuffled_df.reset_index(inplace = True, drop = True)

    print('Head of shuffled dataset :')
    display(shuffled_df.head())

    return shuffled_df

#DEFINITION OF PREPROCESSING FUNCTIONS
def replace_contractions(document):
    """
    replaces contracted expressions in a document

    returns document with no contracted expressions"""
    return contractions.fix(document)

def remove_urls(document):
    """
    removes all urls in the document

    returns a document without any urls"""
    return re.sub(r'https?://(www\.)?[-\w@:%.\+~#=]{2,256}\.[a-z]{2,6}\b([-\w@:%_\+.~#?&/=;]*)', '', document)

def remove_empty_html_tags(document):
    """
    removes empty html tags like <br />, <hr />, etc.

    returns a document without filtered from empty html tags"""
    return re.sub(r'(<\w+\s*/?>)', ' ', document)

def clean_sentence_anchors(document):
    """
    cleans all sentences within a document, such that
    the end of a sentence and the beginning of a new one is separated by a period (or many)
    followed by a whitespace
    This cleaning is required because upon removing punctuation,
    some words get concatenated and create new meaningless terms

    example of a dirty document: "This is a dirty sentence.Another dirty sentence begins"
    cleaned version: "This is a cleaned sentence. Another cleaned sentence begins"

    This pattern repeats with a sentence ending with a lowercase/uppercase letter and
    another one beginning with a lowercase/uppercase letter
    The beginning sentence could also end with a digit and the next sentence could begin with
    a digit. Hence we get three different patterns:
    word.*word
    word.*digit
    digit.*word

    returns a document with cleaned sentences"""

    word_word = r'([a-zA-Z]+\.*)\.([a-zA-Z]+)' #word(.*)word pattern
    word_digit = r'([a-zA-Z]+\.*)\.(\d+)' #word(.*)digit pattern
    digit_word = r'(\d+\.*)\.([a-zA-Z]+)' #digit(.*)word pattern
    patterns = [
        word_word,
        word_digit,
        digit_word,
    ]

    for pattern in patterns:
        if re.search(pattern, document):
            document = re.sub(pattern, r'\1. \2', document)

    return document

def remove_non_ascii(tokens):
    """
    normalizes the tokens
    encodes tokens as ASCII characters from tokens
    and decodes as utf-8

    returns a list of normalized and encoded as ascii tokens"""
    return [unicodedata.normalize('NFKD', token)
           .encode('ascii', 'ignore')
           .decode('utf-8', 'ignore')
           for token in tokens]

def split_on_characterset(tokens, regex):
    """
    splits a token in tokens upon matching with the characterset defined by the regex
    and appends the tokens obtained from splitting the token to the tokens list

    returns a list of all tokens obtained after splitting problematic tokens"""

    new_tokens = []
    for token in tokens:
        if re.search(regex, token) :
            new_tokens += re.split(regex, token)
        else:
            new_tokens.append(token)

    return new_tokens

def to_lowercase(tokens):
    """returns a list of tokens in lowercase"""
    return [token.lower() for token in tokens]

def replace_numbers(tokens):
    """
    replaces tokens representing whole numeric values
    by their equivalent letter values

    returns a list of transformed tokens"""

    engine = inflect.engine()
    new_tokens = []
    for token in tokens:
        new_token = token
        if token.isdigit():
            new_token = engine.number_to_words(token)
        new_tokens.append(new_token)

    return new_tokens

def remove_punctuation(tokens):
    """
    removes tokens not in \w and \s classes of characters.
    By extension, all punctuation characters will be removed

    returns a list of tokens only in \w and \s"""

    new_tokens = []
    for token in tokens:
        new_token = re.sub(r'[^\w\s]', '', token)
        if new_token != '':
            new_tokens.append(new_token)
    return new_tokens

def remove_stopwords(tokens, stopwords=STOP_WORDS):
   """
   removes all stopwords (a set) from tokens (a list)
   except those in exceptions (a set)

   returns a list of tokens that are not stopwords"""
   return [token for token in tokens if token not in stopwords]

def lemmatize(tokens, lemmatizer, pos_tag_map):
    """
    lematizes all tokens using a lemmatizer and a POS-Tagging map

    returns the list of lemmatized tokens"""
    return [lemmatizer.lemmatize(token, pos_tag_map[tag[0]]) for token, tag in pos_tag(tokens)]

def normalize(tokens):
    """
    normalizes all tokens by:
    1. removing non ASCII characters
    2. converting to lowercase
    3. splitting wrongfully joined tokens
    4. replacing numbers with their equivalent letter representation
    5. removing punctuation
    6. removing stopwords
    7. lemmatizing using POS-Tags

    returns the list of normalized tokens"""

    tokens = remove_non_ascii(tokens)
    tokens = to_lowercase(tokens)
    tokens = split_on_characterset(tokens, r'[/\\~_-]')
    tokens = replace_numbers(tokens)
    tokens = remove_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens, WordNetLemmatizer(), POS_TAG_MAP)
    return tokens

def preprocess(document):
    """
    preprocesses the document for vectorization by:
    1. replacing contractions by their equivalent full expressions
    2. removing empty html tags
    3. removing urls
    4. cleaning sentences beginning and end anchors
    5. tokenizing the document
    6. normalizing its tokens
    7. joining normalized tokens back to recreate the document

    returns a preprocessed document, ready for vectorization"""

    document = replace_contractions(document)
    document = remove_empty_html_tags(document)
    document = remove_urls(document)
    document = clean_sentence_anchors(document)
    tokens = word_tokenize(document)
    tokens = normalize(tokens)
    document = ''.join([" " + token for token in tokens]).strip()

    return document

def preprocess_dataset(dataset):
    """
    preprocesses all documents in a dataset

    returns a dataset with preprocessed documents
    and ready for vectorization"""
    return [preprocess(document) for document in dataset]

#DEFINITION OF UTILITY CLASSES
#class used to encapsulate the results of the gridsearch
class GridSearchResult:

    def __init__(self, name, score, estimator):
        self.name = name
        self.score = score
        self.estimator = estimator

    def __str__(self):
        return """
        Model: {}
        Best Accuracy Score: {}
        Best Estimator: {}
        """.format(self.name, self.score, self.estimator)

#class used to transform a sparse matrix into a dense matrix
#to be used by some pipelines during fitting stages
class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
