import os
import numpy
import string
import io
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.cross_validation import KFold
from pandas import DataFrame

def read_file_hierarchy(path):
    for root, dir_names, file_names in os.walk(path):
        for fName in file_names:
            file_path = os.path.join(root, fName)
            if os.path.isfile(file_path):
                array_lines = []
                f = open(file_path)
                for line in f:
                	array_lines.append(line)
                f.close()
                content = "\n".join(array_lines)
                yield file_path, content


def lable_data(fName):
    if (fName.startswith('spm', 11) or fName.startswith('spm', 12)):
        data_class = 'spam'
    else:
        data_class = 'ham'
    return data_class

dataFiles = [
    ('bare/part1'),
    ('bare/part2'),
    ('bare/part3'),
    ('bare/part4'),
    ('bare/part5'),
    ('bare/part6'),
    ('bare/part7'),
    ('bare/part8'),
    ('bare/part9'),
    ('bare/part10')
]

X = DataFrame({'text': [], 'class': []})
for path in dataFiles:
    rows = []
    index = []
    flag = 0
    for fName, text in read_file_hierarchy(path):
        if flag:
            data_class = lable_data(fName)
            rows.append({'text': text, 'class': data_class})
            index.append(fName)
        flag = 1

    data_frame = DataFrame(rows, index=index)
    X = X.append(data_frame)

X = X.reindex(numpy.random.permutation(X.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    # ('classifier',         LinearSVC(loss='hinge'))
    ('classifier',         LinearSVC())         ## compare
])

kCrossValidate = KFold(n=len(X), n_folds=5)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in kCrossValidate:
    # print test_indices
    trainX = X.iloc[train_indices]['text'].values
    trainY = X.iloc[train_indices]['class'].values.astype(str)

    test_text = X.iloc[test_indices]['text'].values
    test_y = X.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(trainX, trainY)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    print confusion_matrix(test_y, predictions)
    

print'Total emails classified:', len(X)
print 'Confusion matrix:', confusion