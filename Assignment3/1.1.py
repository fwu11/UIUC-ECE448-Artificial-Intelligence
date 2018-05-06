# implement 1.1
# fwu11

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from io_tools import read_dataset

def calculate_prior(label):

    prior = np.zeros(10)
    n = len(label)
    for i in range(10):
        prior[i] = sum(label == i) / n

    return prior


def calculate_likelihood(train_data, label, k):

    foreground_likelihood = np.zeros((10, 32, 32))
    background_likelihood = np.zeros((10, 32, 32))
    for i in range(10):
        candidate = train_data[label == i]
        length = len(candidate)
        foreground_likelihood[i] = (
            np.sum(candidate, axis=0) + k) / (length + 2 * k)
        background_likelihood[i] = (
            length - np.sum(candidate, axis=0) + k) / (length + 2 * k)

    return foreground_likelihood, background_likelihood


def calculate_posterior(f1, f0, prior):

    log_posterior = np.log(prior) + np.sum(np.log(f1)) + np.sum(np.log(f0))

    return log_posterior


def test(test_data, prior, foreground_likelihood, background_likelihood):

    n = len(test_data)
    prediction = np.zeros(n)

    for i in range(n):
        posterior = np.zeros(10)
        for j in range(10):
            temp = test_data[i]
            f1 = foreground_likelihood[j][temp == 1]
            f0 = background_likelihood[j][temp == 0]
            posterior[j] = calculate_posterior(f1, f0, prior[j])
        prediction[i] = np.argmax(posterior)

    return prediction

def accuracy(result,test_label):
    
    n = len(test_label)
    test_accuracy = sum(result == test_label)/n
    
    return test_accuracy

def evaluation(result,test_label):

    confusion_matrix = np.zeros((10,10))
    for i in range(10):
        tmp = result[test_label == i]
        for j in range(10):
            confusion_matrix[i][j] = sum(tmp==j)/len(tmp)

    return confusion_matrix

def prototypical_instance(test_data,test_label,prior, foreground_likelihood, background_likelihood):

    most_prototypical = np.zeros((10,32,32))
    least_prototypical = np.zeros((10,32,32))
    max_posterior = np.zeros(10)
    min_posterior = np.zeros(10)

    for i in range(10):
        sample = test_data[test_label==i]
        maximum = float('-inf')
        minimum = float('inf')
        for j in range(len(sample)):
            temp = sample[j]
            f1 = foreground_likelihood[i][temp == 1]
            f0 = background_likelihood[i][temp == 0]
            posterior = calculate_posterior(f1, f0, prior[i])
            if posterior > maximum:
                maximum = posterior
                most_prototypical[i] = temp
            if posterior < minimum:
                minimum = posterior
                least_prototypical[i] = temp
        max_posterior[i] = maximum
        min_posterior[i] = minimum

    return max_posterior,min_posterior,most_prototypical,least_prototypical

def odd_ratios(foreground_likelihood):
    # from inspection
    # largest four pairs
    # (2,8),(4.7),(4,8),(3,7)
    pairs = [(2,8),(4,7),(4,8),(3,7)]

    for pair in pairs:
        plt.figure()
        first_class = np.log(foreground_likelihood[pair[0]])
        second_class = np.log(foreground_likelihood[pair[1]])
        odds = second_class-first_class
        
        ax1 = plt.subplot(131)
        im1 = plt.imshow(first_class, cmap="jet")
        plt.axis('off')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="20%", pad=0.2)
        plt.colorbar(im1, cax=cax)
        ax2 = plt.subplot(132)
        im2 = plt.imshow(second_class, cmap="jet")
        plt.axis('off')
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="20%", pad=0.2)
        plt.colorbar(im2, cax=cax)
        ax3 = plt.subplot(133)
        im3 = plt.imshow(odds, cmap="jet")
        plt.axis('off')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="20%", pad=0.2)
        plt.colorbar(im3, cax=cax)
        plt.show()


def show_results(confusion_matrix):
    f = open('1.1confusion_matrix.txt','w')
    for i in range(10):
        for j in range(10):
            f.write(str("%.4f" % confusion_matrix[i][j])+' ')
        f.write('\n')
    f.close()

def print_to_file(most,least):
    f1 = open('most.txt','w')
    f2 = open('least.txt','w')
    for i in range(10):
        for j in range(32):
            for k in range(32):
                f1.write(str(most[i][j][k]))
                f2.write(str(least[i][j][k]))
            f1.write('\n')
            f2.write('\n')
    f1.close()   
    f2.close()

def main():
    k = 0.1
    train_data_file = './digitdata/optdigits-orig_train.txt'
    test_data_file = './digitdata/optdigits-orig_test.txt'

    train_data = read_dataset(train_data_file)
    test_data = read_dataset(test_data_file)
    prior = calculate_prior(train_data['label'])
    foreground_likelihood, background_likelihood = calculate_likelihood(train_data['image'],train_data['label'],k)
    result = test(test_data['image'],prior,foreground_likelihood,background_likelihood)
    test_accuracy = accuracy(result,test_data['label'])
    print(test_accuracy)
    confusion_matrix = evaluation(result,test_data['label'])
    show_results(confusion_matrix)
    max_posterior,min_posterior,most_prototypical,least_prototypical = prototypical_instance(test_data['image'],test_data['label'],prior, foreground_likelihood, background_likelihood)
    odd_ratios(foreground_likelihood)
    print_to_file(most_prototypical,least_prototypical)
if __name__ == "__main__":
    main()
