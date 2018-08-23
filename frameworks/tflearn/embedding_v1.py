import numpy as np
from tensorflow.contrib import learn
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('./data/titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data_1, labels = load_csv('./data/titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

data = []
for it in data_1:
    data.append(it[1])

print data
tokenizer = learn.preprocessing.VocabularyProcessor(10)
data = list(tokenizer.fit_transform(data))
data = tflearn.data_utils.pad_sequences(data, maxlen=10)
print data

trainX = data
trainY = labels

testX = data
testY = labels

def run():
    net = tflearn.input_data(shape=[None, 1])
    print net
    # embed int vector to compact real vector
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    print net
    # fucking magic of rnn
    # if dynamic lstm, backprop thru time till the seq ends,
    # but padding is needed to feed input dim; tail not used
    net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             learning_rate=0.001,
                             loss='categorical_crossentropy')
 
    m = tflearn.DNN(net)
    m.fit(trainX, trainY, validation_set=(testX, testY),
          show_metric=True, batch_size=32)
    m.save('models/lstm.tfl')
 

run()
