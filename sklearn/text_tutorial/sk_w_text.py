'''An example of working with text in sklearn.'''

from sklearn.datasets import fetch_20newsgroups

# Limit categories for faster execution times
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories,
                                  shuffle=True,
                                  random_state=42)

# Examples of accessing data from the object above
print("Target categories:")
print(twenty_train.target_names)
print()
print("Number of records by data in memory and by filenames:")
print(len(twenty_train.data))
print(len(twenty_train.filenames))
print()
print("First lines of the first file and its target category:")
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])
print()
