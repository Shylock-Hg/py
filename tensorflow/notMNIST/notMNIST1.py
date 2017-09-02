'''
@brief: a softmax classifier to notMNIST data
@author: Shylock Hg
@time: 2017/8/20
@email: tcath2s@icloud.com
'''

#import
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

'''
  First, we'll download the dataset to our local machine. The data consists of characters rendered in
a variety of fonts on a 28x28 image. The labels are limited to 'A' through 'J' (10 classes). The
training set has about 500k and the testset 19000 labeled examples. Given these sizes, it should
be possible to train models quickly on any machine.
'''

url = 'https://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None
data_root = '.'

TRAINING_DATA_NAME = 'notMNIST_large.tar.gz'
TRAINING_DATA_SIZE = 247336696

EVAL_DATA_NAME = 'notMNIST_small.tar.gz'
EVAL_DATA_SIZE = 8458043

def download_progress_hook(count,block_size,total_size):
    '''
      A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 5% change in download progress.
    '''

    global last_percent_reported
    #compute percent of download progress
    percent = int(count*block_size*100/total_size)

    #is percent equals last reported percent
    if percent != last_percent_reported :
        if percent%5 == 0:
            print('{}%'.format(percent))
        else:
            print('.')
    last_percent_reported = percent


def maybe_download(filename,expected_bytes,force=False):
    '''
    Download file if file not exsits.
    @param:
        filename: Name of file to download.
        expected: The size of file should download.
        force: Download without whether exsits.
    '''
    #destinate file path
    dest_filename = os.path.join(data_root,filename)

    #if download file
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:{}'.format(filename))
        urlretrieve(url+filename,dest_filename,reporthook = download_progress_hook)
        print('\nDownload Complete!')
    else:
        print('File {} exists!'.format(filename))

    #is file Complete
    statinfo = os.stat(dest_filename)
    if expected_bytes == statinfo.st_size:
        print('File {} is downloaded completely!'.format(filename))
    else:
        raise Exception(
            'File {} is not downloaded completely!'.format(filename)
        )
    return dest_filename

#download .tar.gz file
train_data = maybe_download(TRAINING_DATA_NAME,TRAINING_DATA_SIZE)
eval_data = maybe_download(EVAL_DATA_NAME,EVAL_DATA_SIZE)

num_classes = 10
np.random.seed(133)

def maybe_extract(filename,force = False):
    '''
    extract files to dir 'data_root/root/[class_name]'
    '''
    global data_root
    #get root dir
    root = os.path.splitext(os.path.splitext(filename)[0])[0] #remove .tar.gz
    #does dir exists?
    if not force and os.path.isdir(root):
        print('Dir {} exists!'.format(root))
    else:
        print('Dir {} don\'t exists! Extract the file {}'.format(root,filename))
        with tarfile.open(filename) as tf:
            sys.stdout.flush()
            tf.extractall(data_root)
    #does extract completely?
    data_folders = [os.path.join(root,d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root,d))]
    if(len(data_folders) == num_classes):
        print('The data extract completely!')
    else:
        raise Exception('The data extract uncompletely!')

    return data_folders

#extract .tar.gz file to ./DATA_NAME/{A-J}/{IMAGES}
eval_folders = maybe_extract(EVAL_DATA_NAME)
train_folders = maybe_extract(TRAINING_DATA_NAME)


#image property
image_size = 28 #height & width
pixel_depth = 255.0 #pixel size

def load_letter(folders,min_num_images):
    '''load data for a single letter label.'''
    images_list = os.listdir(folders)
    dataset = np.ndarray(shape=(len(images_list),image_size,image_size),dtype=np.float32)
    #apend image to dataset
    image_counter = 0
    for i in images_list:
        image = os.path.join(folders,i)
        try:
            image_data = (ndimage.imread(image).astype(float) - pixel_depth/2)/pixel_depth
            if image_data.shape != (image_size,image_size):
                raise Exception('Unexpected image size:{}'.format(image_data.shape))
            dataset[image_counter,:,:] = image_data
            image_counter += 1
        except IOError as e:
            print('Can\'t read image :{} , {} -it\'s ok! , skipping!'.format(image,e))
    #reshape dataset
    dataset = dataset[0:image_counter,:,:]
    #enough images
    if image_counter < min_num_images:
        raise Exception('Less images than expected!')
    print('Full dataset tensor:{}'.format(dataset.shape))
    print('Mean:{}'.format(np.mean(dataset)))
    print('Std:{}'.format(np.std(dataset)))
    return dataset

def maybe_pickle(data_folders,min_num_images_per_class,force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder+'.pickle'
        dataset_names.append(set_filename)
        #is pickle.dump
        if not force and os.path.exists(set_filename):
            print('{} existed!'.format(set_filename))
        else:
            print('Pickling {}'.format(set_filename))
            dataset = load_letter(folder,min_num_images_per_class)
            try:
                with open(set_filename,'wb') as f:
                    pickle.dump(dataset,f,pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save to {}:{}'.format(set_filename,e))
    return dataset_names

#pikel images to ./DATA_NAME/{CLASS.pickle}
train_datasets = maybe_pickle(train_folders, 45000)
eval_datasets = maybe_pickle(eval_folders, 1800)


#construct ndarray for images[images_num,image_size,image_size] and labels[images_num,10]
def make_ndarray(images_num,image_size):
    if images_num:
        return np.ndarray((images_num,image_size,image_size),dtype=np.float32),np.ndarray(images_num,dtype=np.int32)
    else:
        return None,None

#merge dataset
def merge_datasets(pickle_files,train_size,valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_ndarray(valid_size,image_size)
    train_dataset, train_labels = make_ndarray(train_size,image_size)
    valid_size_per_class = valid_size//num_classes
    train_size_per_class = train_size//num_classes

    t_start,v_start = 0,0
    t_end,v_end = train_size_per_class,valid_size_per_class
    end_l = train_size_per_class+valid_size_per_class
    for label,pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file,'rb') as f:
                #load a letter.pickle file
                letter_set = pickle.load(f)
                #
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:valid_size_per_class,:,:]
                    valid_dataset[v_start:v_end,:,:] = valid_letter
                    valid_labels[v_start:v_end] = label
                    v_start += valid_size_per_class
                    v_end += valid_size_per_class

                train_letter = letter_set[valid_size_per_class:end_l,:,:]
                train_dataset[t_start:t_end,:,:] = train_letter
                train_labels[t_start:t_end] = label
                t_start += train_size_per_class
                t_end += train_size_per_class
        except Exception as e:
            print('Can\'t process data from {} : {}'.format(pickle_file,e))
            raise
    return valid_dataset,valid_labels,train_dataset,train_labels


train_size = 200000
valid_size = 10000
eval_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, valid_size)
_, _, eval_dataset, eval_labels = merge_datasets(eval_datasets, eval_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Evaling:', eval_dataset.shape, eval_labels.shape)


#shuffle the dataset and labels
def randomize(dataset,labels):
    permutation = np.random.permutation(labels.shape[0])
    return dataset[permutation,:,:],labels[permutation]
#shuffle
train_dataset, train_labels = randomize(train_dataset, train_labels)
eval_dataset, eval_labels = randomize(eval_dataset, eval_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


#dump all dataset/labels to ./notMNIST.pickle
pickle_file = os.path.join(data_root, 'notMNIST.pickle')
if not os.path.exists(pickle_file):
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'eval_dataset': eval_dataset,
            'eval_labels': eval_labels,
        }
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

print('Compressed pickle file size:{}'.format(os.stat(pickle_file).st_size))
