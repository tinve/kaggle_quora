from __future__ import division

import pandas as pd
import string
import datetime
import sframe

from nltk.corpus import stopwords  # Import stop word list


def is_true_duplicate(x, y):
    x = x.lower()
    y = y.lower()
    return x == y


def text_to_word_lists(raw_text):
    # Convert to lower case
    words = raw_text.lower()
    # Replace punctuation with spaces, split into words
    words = words.translate(string.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = words.split()
    # Set of stop words
    stops = set(stopwords.words("english"))
    # Return list with and without stop words
    return sframe.SArray([words, [w for w in words if w not in stops]])

print 'start'
print datetime.datetime.now()

train = pd.read_csv('train.csv').fillna('')
train = sframe.SFrame(train)
train = train.remove_columns(['id', 'qid1', 'qid2'])

test = pd.read_csv('test.csv').fillna('')
test = sframe.SFrame(test)
ids = test[['test_id']]
test = test.remove_columns(['test_id'])

print 'files reading complete'
print datetime.datetime.now()

# Create columns with word lists from Q1 and Q2 in train
result = train['question1'].apply(lambda x: text_to_word_lists(x)).unpack()
train.add_columns(result.rename({'X.0': 'word_list1', 'X.1': 'word_list_no_stops1'}))
result = train['question2'].apply(lambda x: text_to_word_lists(x)).unpack()
train.add_columns(result.rename({'X.0': 'word_list2', 'X.1': 'word_list_no_stops2'}))

# Create columns with word lists from Q1 and Q2 in test
result = test['question1'].apply(lambda x: text_to_word_lists(x)).unpack()
test.add_columns(result.rename({'X.0': 'word_list1', 'X.1': 'word_list_no_stops1'}))
result = test['question2'].apply(lambda x: text_to_word_lists(x)).unpack()
test.add_columns(result.rename({'X.0': 'word_list2', 'X.1': 'word_list_no_stops2'}))

print 'saving auxiliary files'
print datetime.datetime.now()

# Save target, train ids and true duplicates to csv
train[['is_duplicate']].save('target', format='csv')
ids.save('ids', format='csv')

print 'auxiliary saved, saving train'
print datetime.datetime.now()

# Save train with all lists to binary file
train[['question1',           'question2',
       'word_list1',          'word_list2',
       'word_list_no_stops1', 'word_list_no_stops2']].save('train_lists.sf')

print 'saved train, saving test'
print datetime.datetime.now()

# Save test with all lists to binary file
test[['question1',           'question2',
      'word_list1',          'word_list2',
      'word_list_no_stops1', 'word_list_no_stops2']].save('test_lists.sf')

print 'saved test, done'
print datetime.datetime.now()

