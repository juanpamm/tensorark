#
# This python script converts a sample of the notMNIST dataset into
# the same file format used by the MNIST dataset. If you have a program
# that uses the MNIST files, you can run this script over notMNIST to
# produce a new set of data files that should be compatible with
# your program.
#
# Instructions:
#
# 1) if you already have a MNIST data/ directory, rename it and create
#    a new one
#
# $ mv data data.original_mnist
# $ mkdir convert_MNIST
#
# 2) Download and unpack the notMNIST data. This can take a long time
#    because the notMNIST data set consists of ~500,000 files
#
# $ curl -o notMNIST_small.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz
# $ curl -o notMNIST_large.tar.gz http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz
# $ tar xzf notMNIST_small.tar.gz
# $ tar xzf notMNIST_large.tar.gz
#
# 3) Run this script to convert the data to MNIST files, then compress them.
#    These commands will produce files of the same size as MNIST
#    notMNIST is larger than MNIST, and you can increase the sizes if you want.
#
# $ python convert_to_mnist_format.py notMNIST_small test 1000
# $ python convert_to_mnist_format.py notMNIST_large train 6000
# $ gzip convert_MNIST/*ubyte
#
# 4) After update, we cancel output path and replace with 'train', 'test' or test ratio number,
#    it not only work on 10 labels but more,
#    it depends on your subdir number under target folder, you can input or not input more command
#
# Now we define input variable like following:
# $ python convert_to_mnist_format.py target_folder test_train_or_ratio data_number
#
# target_folder: must give minimal folder path to convert data
# test_train_or_ratio: must define 'test' or 'train' about this data,
#                      if you want seperate total data to test and train automatically,
#                      you can input one integer for test ratio,
#                      e.q. if you input 2, it mean 2% data will become test data
# data_number: if you input 0 or nothing, it convert total images under each label folder,
#        e.q.
#          a. python convert_to_mnist_format.py notMNIST_small test 0
#          b. python convert_to_mnist_format.py notMNIST_small test
#          c. python convert_to_mnist_format.py notMNIST_small train 0
#          d. python convert_to_mnist_format.py notMNIST_small train
#
import zipfile
import gzip
import shutil
import numpy
import imageio
import sys
import os
import random

height = 0
width = 0
class_names = []


def get_subdir(folder):
    list_dir = None
    for root, dirs, files in os.walk(folder):
        if not dirs == []:
            list_dir = dirs
            break
    list_dir.sort()
    return list_dir


def get_labels_and_files(folder, number=0):
    # Make a list of lists of files for each label
    filelists = []
    subdir = get_subdir(folder)
    for label in range(0, len(subdir)):
        filelist = []
        filelists.append(filelist)
        dirname = os.path.join(folder, subdir[label])
        for file in os.listdir(dirname):
            if file.endswith('.png'):
                fullname = os.path.join(dirname, file)
                if os.path.getsize(fullname) > 0:
                    filelist.append(fullname)
                else:
                    print('file ' + fullname + ' is empty')
        # sort each list of files so they start off in the same order
        # regardless of how the order the OS returns them in
        filelist.sort()

    # Take the specified number of items for each label and
    # build them into an array of (label, filename) pairs
    # Since we seeded the RNG, we should get the same sample each run
    labels_and_files = []
    for label in range(0, len(subdir)):
        count = number if number > 0 else len(filelists[label])
        filelist = random.sample(filelists[label], count)
        for filename in filelist:
            labels_and_files.append((label, filename))

    return labels_and_files


def make_arrays(labels_and_files, ratio):
    global height, width
    images = []
    labels = []
    im_shape = imageio.imread(labels_and_files[0][1]).shape
    if len(im_shape) > 2:
        height, width, channels = im_shape
    else:
        height, width = im_shape
        channels = 1
    for i in range(0, len(labels_and_files)):
        # display progress, since this can take a while
        if i % 100 == 0:
            sys.stdout.write("\r%d%% complete" %
                             ((i * 100) / len(labels_and_files)))
            sys.stdout.flush()

        filename = labels_and_files[i][1]
        try:
            image = imageio.imread(filename)
            images.append(image)
            labels.append(labels_and_files[i][0])
        except OSError:
            # If this happens we won't have the requested number
            print("\nCan't read image file " + filename)

    if ratio == 'train':
        ratio = 0
    elif ratio == 'test':
        ratio = 1
    else:
        ratio = float(ratio) / 100
    count = len(images)
    train_num = int(count * (1 - ratio))
    test_num = count - train_num

    if channels > 1:
        train_imagedata = numpy.zeros(
            (train_num, height, width, channels), dtype=numpy.uint8)
        test_imagedata = numpy.zeros(
            (test_num, height, width, channels), dtype=numpy.uint8)
    else:
        train_imagedata = numpy.zeros(
            (train_num, height, width), dtype=numpy.uint8)
        test_imagedata = numpy.zeros(
            (test_num, height, width), dtype=numpy.uint8)
    train_labeldata = numpy.zeros(train_num, dtype=numpy.uint8)
    test_labeldata = numpy.zeros(test_num, dtype=numpy.uint8)

    for i in range(train_num):
        train_imagedata[i] = images[i]
        train_labeldata[i] = labels[i]

    for i in range(0, test_num):
        test_imagedata[i] = images[train_num + i]
        test_labeldata[i] = labels[train_num + i]
    print("\n")
    return train_imagedata, train_labeldata, test_imagedata, test_labeldata


def write_labeldata(labeldata, outputfile):
    header = numpy.array([0x0801, len(labeldata)], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(labeldata.tobytes())


def write_imagedata(imagedata, outputfile):
    global height, width
    header = numpy.array([0x0803, len(imagedata), height, width], dtype='>i4')
    with open(outputfile, "wb") as f:
        f.write(header.tobytes())
        f.write(imagedata.tobytes())


def convert_image_set(parameters, dst_path):
    train_label_path = dst_path + "/train-labels-idx1-ubyte"
    train_image_path = dst_path + "/train-images-idx3-ubyte"
    test_label_path = dst_path + "/t10k-labels-idx1-ubyte"
    test_image_path = dst_path + "/t10k-images-idx3-ubyte"
    labels_and_files = []

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    if len(parameters) is 2:
        labels_and_files = get_labels_and_files(parameters[0])
    elif len(parameters) is 3:
        labels_and_files = get_labels_and_files(parameters[0], int(parameters[2]))
    random.shuffle(labels_and_files)

    train_image_data, train_label_data, test_image_data, test_label_data = make_arrays(
        labels_and_files, parameters[1])

    if parameters[1] == 'train':

        write_labeldata(train_label_data, train_label_path)
        write_imagedata(train_image_data, train_image_path)
    elif parameters[1] == 'test':

        write_labeldata(test_label_data, test_label_path)
        write_imagedata(test_image_data, test_image_path)
    else:
        write_labeldata(train_label_data, train_label_path)
        write_imagedata(train_image_data, train_image_path)
        write_labeldata(test_label_data, test_label_path)
        write_imagedata(test_image_data, test_image_path)


def file_extraction_manager(mediar, file, working_dir):
    path_to_file = os.path.join(mediar, file.name)

    with zipfile.ZipFile(path_to_file, 'r') as zip_file:
        zip_file.extractall(working_dir)

    if os.path.exists(path_to_file):
        os.remove(path_to_file)


def get_last_modified_dir(mediar):
    tmp_date = 0
    tmp_position = 0
    list_dirs = os.listdir(mediar)

    for i in range(len(list_dirs)):
        if os.path.isdir(os.path.join(mediar, list_dirs[i])) is True:
            if i == 0:
                tmp_date = os.path.getmtime(os.path.join(mediar, list_dirs[i]))
                tmp_position = i
            elif i != 0:
                if tmp_date < os.path.getmtime(os.path.join(mediar, list_dirs[i])):
                    tmp_date = os.path.getmtime(os.path.join(mediar, list_dirs[i]))
                    tmp_position = i

    return os.path.join(mediar, list_dirs[tmp_position])


def gzip_all_files_in_dir(path_to_dir):
    list_dir = os.listdir(path_to_dir)

    for i in range(len(list_dir)):
        if os.path.isfile(os.path.join(path_to_dir, list_dir[i])) is True:
            with open(os.path.join(path_to_dir, list_dir[i]), 'rb') as file_opened:
                with gzip.open(os.path.join(path_to_dir, list_dir[i]) + '.gz', 'wb') as file_wr:
                    shutil.copyfileobj(file_opened, file_wr)
            os.remove(os.path.join(path_to_dir, list_dir[i]))


def set_name_classes(path):
    class_names.clear()
    list_dir = get_subdir(path)
    for label in list_dir:
        dirname = label
        class_names.append(dirname)


def load_data(dst_path):
    files = os.listdir(dst_path)
    sorted(files)
    paths = []

    for fname in files:
        paths.append(os.path.join(dst_path, fname))

    with gzip.open(paths[3], 'rb') as lbpath:
        y_train = numpy.frombuffer(lbpath.read(), numpy.uint8, offset=8)

    with gzip.open(paths[2], 'rb') as imgpath:
        x_train = numpy.frombuffer(imgpath.read(), numpy.uint8, offset=16)  .reshape(len(y_train), 28, 28)

    with gzip.open(paths[1], 'rb') as lbpath:
        y_test = numpy.frombuffer(lbpath.read(), numpy.uint8, offset=8)

    with gzip.open(paths[0], 'rb') as imgpath:
        x_test = numpy.frombuffer(imgpath.read(), numpy.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
