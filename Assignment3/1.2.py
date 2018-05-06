# implement 1.2
# fwu11
import numpy as np
import time
from io_tools import read_dataset
import cProfile

def calculate_prior(label):

    prior = np.zeros(10)
    n = len(label)
    for i in range(10):
        prior[i] = sum(label == i) / n

    return prior


def decode_mask(mask):

    string = np.arange(len(mask.ravel())-1,-1,-1)
    power = string[mask.ravel() == 1]
    order = sum(2**power)
    
    return order


def calculate_likelihood(train_data, label, m, n, k, switch):
    # need to specify the feature_value
    feature_value = 2**(m * n)
    if switch == 'disjoint':
        horizontal = 32 // m
        vertical = 32 // n
        increment_x = m
        increment_y = n

    else:
        horizontal = 32 - m + 1
        vertical = 32 - n + 1
        increment_x = 1
        increment_y = 1
    likelihood = np.zeros((10, feature_value, vertical, horizontal))

    for i in range(10):
        candidate = train_data[label == i]
        length = len(candidate)

        for x in range(horizontal):
            for y in range(vertical):
                for idx in range(length):
                    mask = candidate[idx][y * increment_y:y * increment_y + n,x * increment_x:x * increment_x + m]
                    order = decode_mask(mask)
                    likelihood[i][order][y][x] +=1

        likelihood[i]= (likelihood[i]+k)/ (length + feature_value * k)
    return likelihood


def calculate_posterior(f,prior):

    log_posterior = np.log(prior) + sum(sum(np.log(f)))

    return log_posterior


def test(test_data, prior, likelihood, m, n, switch):
    if switch == 'disjoint':
        horizontal = 32 // m
        vertical = 32 // n
        increment_x = m
        increment_y = n

    else:
        horizontal = 32 - m + 1
        vertical = 32 - n + 1
        increment_x = 1
        increment_y = 1


    prediction = np.zeros(len(test_data))
    # 1.for each test image
    # 2.for each class
    # 3.for each feature G_ij
    for i in range(len(test_data)):
        posterior = np.zeros(10)
        for j in range(10):
            f = np.zeros((vertical,horizontal))
            for x in range(horizontal):
                for y in range(vertical):
                    mask = test_data[i][y * increment_y:y * increment_y + n,x * increment_x:x * increment_x + m]
                    order = decode_mask(mask)
                    f[y][x] = likelihood[j][order][y][x]

            posterior[j] = calculate_posterior(f,prior[j])
        prediction[i] = np.argmax(posterior)

    return prediction

def evaluation(result,test_label):
    
    n = len(test_label)
    test_accuracy = sum(result == test_label)/n
    
    return test_accuracy

    
def main():
    # n*m square of pixels
    m = 3
    n = 3
    k = 0.1
    #switch = 'disjoint'
    switch = 'overlap'
    train_data_file = './digitdata/optdigits-orig_train.txt'
    test_data_file = './digitdata/optdigits-orig_test.txt'

    train_data = read_dataset(train_data_file)
    test_data = read_dataset(test_data_file)

    time1 = time.time()
    prior = calculate_prior(train_data['label'])
    likelihood = calculate_likelihood(train_data['image'], train_data['label'], m, n, k, switch)
    time2 = time.time()
    prediction = test(test_data['image'], prior, likelihood, m, n, switch)
    time3 = time.time()
    test_accuracy = evaluation(prediction,test_data['label'])
    print('feature size of '+str(n)+' by '+str(m)+ ' with training time of ' +str(time2-time1) + ' and testing time of ' + str(time3-time2))
    print('Test accuracy of ' + str(test_accuracy))


if __name__ == "__main__":
    main()
    #cProfile.run('main()')