import nltk, re
import pandas as pd
# spacy has 219 danish stopwords as nltk only has 94
from spacy.lang.da.stop_words import STOP_WORDS
import lemmy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import bz2
import pickle
import _pickle as cPickle


# Load data
data = bz2.BZ2File('data/picl_law_data.pbz2', 'rb')

data = cPickle.load(data)


# Cleanup formatting
data['tokens'] = [re.sub(r'\xa0|\\r\\n|\\r|\\n',' ',w.lower()) for w in data['full_text']]

# Tokenize and remove punctuation and numbers
data['tokens'] = [nltk.word_tokenize(re.sub(r'[^\w\s]|\d',' ',w.lower())) for w in data['tokens']]

# Danish lemmatizer without word tags
lemmatizer = lemmy.load("da")
data['tokens'] = [[min(lemmatizer.lemmatize("", w)) for w in ws] for ws in data['tokens']]

# Remove stopwords
data['tokens'] = [[w for w in ws if w not in STOP_WORDS] for ws in data['tokens']]

# Remove words with less than 2 characters
data['tokens'] = [[w for w in ws if len(w) > 1] for ws in data['tokens']]

data['tokens'] = [[w.lower() for w in ws] for ws in data['tokens']]

# Create string out of tokens
data['string'] = [[] for _ in range(len(data))]
for i in range(len(data)):
    data['string'].values[i] = ' '.join(data['tokens'].values[i])

# Calculate TF-IDF for each token
# Code from : https://medium.com/analytics-vidhya/demonstrating-calculation-of-tf-idf-from-sklearn-4f9526e7e78b
cv = CountVectorizer()
word_count_vector = cv.fit_transform(data['string'])
tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names())

tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(word_count_vector)
idf = pd.DataFrame({'feature_name':cv.get_feature_names(), 'idf_weights':tfidf_transformer.idf_})

tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names())

# Now make dictionary with token:TF-IDF value
data['tf-idf'] = [{} for _ in range(len(data))]
for i in range(len(data)):
    for t in set(data['tokens'].values[i]):
        data['tf-idf'][i][t] = tf_idf.loc[i][t]

# Save data as compressed bz2 pickle file
with bz2.BZ2File('data/picl_law_data_tokenized' + '.pbz2', 'w') as f:
    cPickle.dump(data, f)




