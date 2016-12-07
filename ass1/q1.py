import os
import numpy
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

NEWLINE = '\n'
HAM = 'ham'
SPAM = 'spam'

SOURCES = [
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

SKIP_FILES = {'cmds'}

def read_files(path):
    for root, dir_names, file_names in os.walk(path):
        for path in dir_names:
            read_files(os.path.join(root, path))
        for file_name in file_names:
            if file_name not in SKIP_FILES:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    past_header, lines = False, []
                    f = open(file_path)
                    for line in f:
                        if past_header:
                            lines.append(line)
                        elif line == NEWLINE:
                            past_header = True
                    f.close()
                    content = NEWLINE.join(lines)
                    yield file_path, content


def build_data_frame(path):
    rows = []
    index = []  
    cachedStopWords = stopwords.words("english")
    for file_name, text in read_files(path):
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])
        port = PorterStemmer()
       # text = ' '.join([port.stem(i) for i in text.split()])

        if file_name.find("spm")!=-1:
            classification='spam'
        else:
            classification='ham'

        rows.append({'text': text, 'class': classification})
        index.append(file_name)

    data_frame = DataFrame(rows, index=index)
    return data_frame


data = DataFrame({'text': [], 'class': []})
for path in SOURCES:
    data = data.append(build_data_frame(path))

data = data.reindex(numpy.random.permutation(data.index))

pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
    ('classifier',         MultinomialNB())
])

k_fold = KFold(n=len(data), n_folds=6)
scores = []
confusion = numpy.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = data.iloc[train_indices]['text'].values
    train_y = data.iloc[train_indices]['class'].values.astype(str)

    test_text = data.iloc[test_indices]['text'].values
    test_y = data.iloc[test_indices]['class'].values.astype(str)

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, pos_label=SPAM)
    scores.append(score)

print('Total emails classified:', len(data))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)