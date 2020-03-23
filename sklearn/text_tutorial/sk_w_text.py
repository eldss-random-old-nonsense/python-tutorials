'''An example of working with text in sklearn.'''

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# Limit categories for faster execution times
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)

# Tokenize text to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)

# Need to account for length of different documents
# Divide word count by total number of words => Term Frequencies
# Can also downscale weights of words that occur in many documents => Inverse Document Frequency
# tf-idf = Term Frequency times Inverse Document Frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)

# Using a classifier
# Naive Bayes - uses processed data and provided targets
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Example test predictions
# docs_new = ['God is love', 'OpenGL on the GPU is fast']
# X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
# predicted = clf.predict(X_new_tfidf)
# for doc, category in zip(docs_new, predicted):
#     print('%r => %s' % (doc, twenty_train.target_names[category]))

# Create a pipeline to do the above tasks more easily
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Train with one command now
text_clf.fit(twenty_train.data, twenty_train.target)

# Evaluate performance on test set
twenty_test = fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print("Accuracy with Naive Bayes", np.mean(predicted == twenty_test.target))
# More advanced metrics
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))

# Can we do better with a support vector machine?
text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss="hinge", penalty="l2",
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])
# text_clf_svm.fit(twenty_train.data, twenty_train.target)
# predicted = text_clf_svm.predict(docs_test)
# print("Accuracy with SVM", np.mean(predicted == twenty_test.target))
# print(metrics.classification_report(twenty_test.target, predicted,
#                                     target_names=twenty_test.target_names))
# print(metrics.confusion_matrix(twenty_test.target, predicted))

# Do automatic hyperparameter tuning
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}
# Takes our previous classifier and becomes one itself
gs_clf = GridSearchCV(text_clf_svm, parameters, cv=5,
                      n_jobs=-1)  # n_jobs = use all cores
# Get best params and create model
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
predicted = gs_clf.predict(docs_test)
# Show results
print(metrics.classification_report(twenty_test.target, predicted,
                                    target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))
