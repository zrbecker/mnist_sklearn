import argparse
import numpy as np
from sklearn import linear_model, svm, neighbors

from mnist_util import mnist_read

def sample_rows(xs, ys, count):
    assert(len(xs) == len(ys))
    indices = np.random.choice(range(len(xs)), count, replace=False)
    return xs[indices], ys[indices]

parser = argparse.ArgumentParser()
parser.add_argument('--train_images', type=str, required=True)
parser.add_argument('--train_labels', type=str, required=True)
parser.add_argument('--test_images', type=str, required=True)
parser.add_argument('--test_labels', type=str, required=True)
args = parser.parse_args()

print('Loading training data...', flush=True, end='')
train_images, train_rows, train_cols, train_labels = mnist_read(
        args.train_images, args.train_labels, one_hot_encoding=False)
assert(len(train_images) == len(train_labels))
print('done.')
print('Loaded {} samples with {} inputs and {} outputs.'.format(
        train_images.shape[0], train_images.shape[1],
        train_labels.shape[1] if len(train_labels.shape) > 1 else 1))

print('Training model...', flush=True, end='')
###
### CHOOSE MODEL HERE
###
model = linear_model.SGDClassifier()
# model = neighbors.KNeighborsClassifier()
# model = svm.SVC(gamma=0.001)

# Uncomment next line to use a smaller sample of the training data
# train_images, train_labels = sample_rows(train_images, train_labels, 5000)
model.fit(train_images, train_labels)
print('done.')

print('Loading testing data...', flush=True, end='')
test_images, test_rows, test_cols, test_labels = mnist_read(
        args.test_images, args.test_labels)
assert(len(test_images) == len(test_labels))
assert(train_rows == test_rows)
assert(train_cols == test_cols)
print('done.')
print('Loaded {} samples with {} inputs and {} outputs.'.format(
        test_images.shape[0], test_images.shape[1],
        test_labels.shape[1] if len(test_labels.shape) > 1 else 1))

print('Evaluating model on test data...', flush=True, end='')
# Uncomment next line to use a smaller sample of the testing data
# test_images, test_labels = sample_rows(test_images, test_labels, 1000)
score = model.score(test_images, test_labels)
print('done.')
print('Score: {}'.format(round(score, 2)))