'''
@brief: a simple word2vec model
@author: Shylock Hg
@time: 2017/9/2
@email: tcath2s@icloud.com
'''

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

#Download the data from the source website if necessary

#file macro
URL = 'http://mattmahoney.net/dc/'
FILE = 'text8.zip'
FILE_SIZE = 31344016

#model macro
#vocabulary size
VOCABULARY_SIZE = 50000


def maybe_download(file,expected_bytes):
    '''Download a file if not present,and make sure it's the right size'''
    if not os.path.exists(file):
        file,_ = urlretrieve(URL+file,file)
    statinfo = os.stat(file)
    if statinfo.st_size == expected_bytes:
        print('Found and verified {}'.format(file))
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify'+file+'.Can you get to it with hand?'
        )

def read_data(file):
    '''Extract the first file enclosed in a zip file as a list of words'''
    with zipfile.ZipFile(file) as f:
        #
        return tf.compat.as_str(f.read(f.namelist()[0])).split()

def build_dataset(words):
    '''Build the dictionary and replace rare words with UNK token
       @retval: data--list of index of words,if word in dictionary the index==dictionary[word] else index==0,size=words.size
                count--list of word&count couple , shape==(VOCABULARY_SIZE,2)
                dictionary--dict of [word:index] , index==count.index size==VOCABULARY_SIZE
                reverse_dictionary--dict of [index:word]
    '''
    count = [['UNK',-1]]
    #add best common words&count couple to 'count'
    count.extend(collections.Counter(words).most_common(VOCABULARY_SIZE-1))  #'UNK' exists
    #dictionary cache word&index couple
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    #
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0 #count to dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0

def generate_batch(batch_size,num_skips,skip_window):
    '''Function to generate training batch for the skip-gram model'''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    #read from words sequence
    span = 2*skip_window+1 #[skip_window+traget+skip_window]
    #def a double entry queue
    buffer = collections.deque(maxlen=span)
    #initialize buffer
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    #data->buffer->batch,labels
    for i in range(batch_size//num_skips):
        target = skip_window #target word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            batch[i*num_skips+j] = buffer[skip_window]
            labels[i*num_skips+j,0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    return batch,labels



#Download file
maybe_download(FILE,FILE_SIZE)
#Extract file to get list of words
words = read_data(FILE)
print('Words size is : {}'.format(len(words)))
#build data set
data,count,dictionary,reverse_dictionary = build_dataset(words)
print('Most common words (+UNK):',count[:5])
print('Sample data:',data[:10])
del words #save memery

print('data:',[reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2,1),(4,2)]:
    data_index = 0
    batch,labels = generate_batch(batch_size=8,num_skips=num_skips,skip_window=skip_window)
    print('\nwith num_skips = {} and skip_window = {}'.format(num_skips,skip_window))
    print('   batch:',[reverse_dictionary[bi] for bi in batch])
    print('   labels:',[reverse_dictionary[li] for li in labels.reshape(8)])

BATCH_SIZE = 128
EMBEDDING_SIZE = 128 #dimension of the word vector
SKIP_WINDOW = 1 #how many words to consider left and right
NUM_SKIPS = 2 #how many labels according to a input word
#We pick a random validation set to sample a nearest neighbors.
#Here we limit the validation samples to the words that have a
#low numeric ID, which by construction are also the most frequent.
VALID_SIZE = 16 #Random set of words to evaluate similarity on.
VALID_WINDOW = 100 #Only pick dev samples in the head of the distribution
VALID_EXAMPLES = np.array(random.sample(range(VALID_WINDOW),VALID_SIZE))
NUM_SAMPLED = 64 #Number of negative examples to sample


graph = tf.Graph()
with graph.as_default():
    #Input data
    train_dataset = tf.placeholder(tf.int32,shape=[BATCH_SIZE])
    train_labels = tf.placeholder(tf.int32,shape=[BATCH_SIZE,1])
    valid_dataset = tf.constant(VALID_EXAMPLES,dtype=tf.int32)
    #Variables
    embeddings = tf.Variable(
        tf.random_uniform([VOCABULARY_SIZE,EMBEDDING_SIZE],-1.0,1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([VOCABULARY_SIZE,EMBEDDING_SIZE],
                            stddev=1.0/math.sqrt(EMBEDDING_SIZE))
    )
    softmax_biases = tf.Variable(tf.zeros(VOCABULARY_SIZE))
    #Model.
    #Look up embedding for inputs
    embed = tf.nn.embedding_lookup(embeddings,train_dataset)
    #Compute the softmax loss, using a sample of the negative labels each time
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights,biases=softmax_biases,
            inputs=embed,labels=train_labels,num_sampled=NUM_SAMPLED,num_classes=VOCABULARY_SIZE)
    )

    #Optimizer
    #Note: The optimizer will optimize the softmax_weights AND the embeddings.
    #This is because the embeddings are defined as a Variable quantity and the
    #optimizer's `minimize` method will by default modify all variable quantities
    #that contribute to the tensor it is passed.
    #See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    #Compute the similarity between minibatch examples and all embeddings.
    #We use the cosine distance.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings,valid_dataset
    )
    similarity = tf.matmul(valid_embeddings,tf.transpose(normalized_embeddings))

NUM_STEPS = 100001


with tf.Session(graph=graph) as s:
    tf.global_variables_initializer().run()
    print('Initialized!')
    average_loss = 0
    for step in range(NUM_SKIPS):
        #run train
        batch_data,batch_labels = generate_batch(
            BATCH_SIZE,NUM_SKIPS,SKIP_WINDOW
        )
        feed_dict = {train_dataset:batch_data,train_labels:batch_labels}
        _,l = s.run([optimizer,loss],feed_dict=feed_dict)
        average_loss += 1
        #log
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss/2000
            #The average_loss is an estimate of the loss over the last 2000 batches
            print('Average loss at step {} : {}'.format(step,average_loss))
            average_loss=0
        #note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(VALID_SIZE):
                valid_word = reverse_dictionary[VALID_EXAMPLES[i]]
                top_k = 8 #number of nearest neighbors
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log = 'Nearest to {}:'.format(valid_word)
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '{} {},'.format(log,close_word)
                print(log)
    final_embeddings = normalized_embeddings.eval()

num_points = 400

tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000,method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1,:])

def plot(embeddings,labels):
    assert embeddings.shape[0] >= len(labels) , 'More labels than embeddings'
    pylab.figure(figsize=(15,15)) #in inches
    for i, label in enumerate(labels):
        x,y = embeddings[i,:]
        pylab.scatter(x,y)
        pylab.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    pylab.show()

words = [reverse_dictionary[i] for i in range(1,num_points+1)]
plot(two_d_embeddings,words)
