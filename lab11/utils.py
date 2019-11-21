import numpy as np

class LinArbPUF:
    ''' linArbPUF provides methods to simulate the behaviour of a standard
        Arbiter PUF (linear model)

        attributes:
        num_bits -- bit-length of the PUF
        delays -- runtime difference between the straight connections
            (first half) and crossed connection (second half) in every switch
        parameter -- parameter vector of the linear model (D. Lim)

        methods:
    '''

    def __init__(self, num_bits, mean=0, stdev=1):
        self.num_bits = num_bits
        # dice runtime difference between upper and lower pathes (for crossed/uncrossed state)
        self.delays = np.random.normal(mean, stdev, 2 * num_bits)
        # construct parameter vector of linear model
        param = self.delays[:num_bits] - self.delays[num_bits:]
        self.parameter = np.concatenate((self.delays[:num_bits] - self.delays[num_bits:], np.array([0])), 0)
        self.parameter[1:] += self.delays[:num_bits] + self.delays[num_bits:]

    def generate_challenge(self, numCRPs):
        challenges = np.random.randint(0, 2, [self.num_bits, numCRPs])
        return challenges

    def calc_features(self, challenges):
        # calculate feature vector of linear model
        temp = [np.prod(1 - 2 * challenges[i:, :], 0) for i in range(self.num_bits)]
        features = np.concatenate((temp, np.ones((1, challenges.shape[1]))))
        return features

    def response(self, features):
        return np.dot(self.parameter, features)

    def bin_response(self, features):
        return np.sign(self.response(features))