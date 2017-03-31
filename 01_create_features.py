from __future__ import division

import sframe
import datetime


def lower_identical(text1, text2):
    return text1.lower() == text2.lower()


def generate_features_from_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    feature0 = len(list1)
    feature1 = len(list2)

    feature2 = len(set1)
    feature3 = len(set2)

    feature4 = len(set1 & set2)
    feature5 = len(set1 | set2)
    feature6 = feature4 / max(feature5, 1)

    feature7 = sum(len(e) for e in list1) / max(feature0, 1)
    feature8 = sum(len(e) for e in list2) / max(feature1, 1)

    if feature0 == feature1:
        feature9 = (list1 == list2)
    else:
        feature9 = False

    return sframe.SArray([feature0,
                      feature1,
                      feature2,
                      feature3,
                      feature4,
                      feature5,
                      feature6,
                      feature7,
                      feature8,
                      feature9])


features = {
    'X.0': '#_of_words1',
    'X.1': '#_of_words2',

    'X.2': '#_of_unique_words1',
    'X.3': '#_of_unique_words2',

    'X.4': '#_of_words_in_intersection',
    'X.5': '#_of_words_in_union',
    'X.6': 'jaccard_similarity',

    'X.7': 'average_word_length1',
    'X.8': 'average_word_length2',

    'X.9': 'lists_identical'
}

features_no_stops = {
    'X.0': '#_of_words_no_stops1',
    'X.1': '#_of_words_no_stops2',

    'X.2': '#_of_unique_words_no_stops1',
    'X.3': '#_of_unique_words_no_stops2',

    'X.4': '#_of_words_in_intersection_no_stops',
    'X.5': '#_of_words_in_union_no_stops',
    'X.6': 'jaccard_similarity_no_stops',

    'X.7': 'average_word_length_no_stops1',
    'X.8': 'average_word_length_no_stops2',

    'X.9': 'lists_identical_no_stops'
}

feature_names = features.values() + features_no_stops.values() + ['lower_identical']

print 'start'
print datetime.datetime.now()

train = sframe.load_sframe('train_lists.sf')
test = sframe.load_sframe('test_lists.sf')

print 'files loaded'
print datetime.datetime.now()

# create features with and without stop words for train
result = train.apply(lambda row: generate_features_from_lists(row['word_list1'],
                                                              row['word_list2'])).unpack()
train.add_columns(result.rename(features))

result = train.apply(lambda row: generate_features_from_lists(row['word_list_no_stops1'],
                                                              row['word_list_no_stops2'])).unpack()
train.add_columns(result.rename(features_no_stops))
train['lower_identical'] = train.apply(lambda row: lower_identical(row['question1'], row['question2']))

# create features with and without stop words for test
result = test.apply(lambda row: generate_features_from_lists(row['word_list1'],
                                                             row['word_list2'])).unpack()
test.add_columns(result.rename(features))

result = test.apply(lambda row: generate_features_from_lists(row['word_list_no_stops1'],
                                                             row['word_list_no_stops2'])).unpack()
test.add_columns(result.rename(features_no_stops))
test['lower_identical'] = test.apply(lambda row: lower_identical(row['question1'], row['question2']))

print 'saving train features to sf'
print datetime.datetime.now()
train[feature_names].save('train_features.sf')

print 'saved train features to sf, saving test features to sf'
print datetime.datetime.now()

test[feature_names].save('test_features.sf')

print 'saved test features to sf, done'
print datetime.datetime.now()


