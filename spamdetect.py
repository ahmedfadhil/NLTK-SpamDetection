import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# nltk.download_shell()

messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))

for mess_no, message in enumerate(messages[:10]):
    print(mess_no, message)

messages = pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'message'])
messages.head()
messages.describe()

messages.groupby('label').describe()

messages['length'] = messages['message'].apply(len)

messages['length'].plot.hist(bins=50)

messages['length'].describe()

messages[messages['length'] == 900]['message'].iloc[1]

messages.hist(column='length', by='label', bins=60, figsize=(12, 4))

mess = 'sample message ! notice that: it contains punctuations'

nopunc = [c for c in mess if c not in string.punctuation]
print(nopunc)

# stopwords.words('english')

nopunc = ''.join(nopunc)

nopunc.split()
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.word('englis')]


# Text processing part
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word not in stopwords.word('english')]


messages.head().apply(text_process)

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

print(len(bow_transformer.vocabulary_))

mess4 = messages['message'][3]
print(mess4)
bow4 = bow_transformer.transform([mess4])

print(bow4.shape)

bow_transformer.get_feature_names()[344]

# The TF IDF part

message_bow = bow_transformer.transform(messages['message'])
print('Shape of spare metrix:', message_bow.shape)

message_bow.nnz

sparcity = (100.0 * message_bow.nnz / (message_bow.shape[0] * message_bow.shape))
print('sparcity'.format((sparcity)))

# Term frequency index document frequency

tfidf_transformer = TfidfTransformer.fit(message_bow)

tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]

messages_tfidf = tfidf_transformer.transform(message_bow)

spam_detect_model = MultinomialNB().fit(message_bow, messages['label'])

spam_detect_model.predict(tfidf4)

all_pred = spam_detect_model.predict(messages_tfidf)

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print(classification_report((label_test, predictions)))

