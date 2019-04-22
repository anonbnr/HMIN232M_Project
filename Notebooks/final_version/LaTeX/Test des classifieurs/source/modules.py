#MODULE IMPORTATION AND ENVIRONMENT CONFIGURATION
import re
import unicodedata
import pickle
import warnings

from time import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contractions
import inflect

from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict

from nltk import pos_tag
from nltk import punkt
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utility_ML import *

##### UNCOMMENT THIS SECTION ON FIRST EXECUTION
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
#####

plt.style.use('fivethirtyeight') #choose fivethirtyeight style for plt
warnings.filterwarnings('ignore', category=FutureWarning) #filter FutureWarnings
