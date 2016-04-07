from scipy import vstack, newaxis, arange, sign, dot, array, prod, random, \
                  ones, zeros, concatenate, log, swapaxes, empty, tanh, \
                  hstack, linspace, around, mean, std, nonzero, isinf
from scipy import exp as numexp
import csv
from copy import deepcopy
import random as rd
from itertools import izip
from numpy import sum as numpy_sum
from numpy import vectorize
#import matplotlib.pyplot as plt
#from svm import *

class linearPredictor(object):

    '''linearPredictor provides methods to learn and evaluate a linear model
        based on a weight vector. Methods operate on 2D Array features, with
        the 0th dimension corresponding to different features and the first
        dimension to different samples.

        attributes:
        parameter -- the weight vector

        methods:
        calc_kernel(features1, features2) -- calculates the linear kernel
            between features1 and features2
        response(features) -- calculates the result of the predictor to
            features with respect to the current parameter
        grad(features) -- gives the gradient of the linear model
        shift_param(step) -- changes parameter of instance by step
    '''

    def __init__(self, dim, mean=0, stdev=1):
        '''creates a linearPredictor with normal distributed weights

            Keyword Arguments:
            dim -- dimension of the weight vector
            mean(optional) -- mean of the weight distribution
            stdev(optional) -- standard deviation of the weight distribution
        '''

        self.dim = dim
        self.parameter = random.normal(mean, stdev, dim)

    def calc_kernel(self, features1, features2):
        ''' calculates linear Kernel

            Keyword Attributes:
            features1, features2 -- 2D arrays of feature vectors;
                0th dim features, 1st dim samples

            returns: 2D array, with kernel[i,j] the kernel of the i-th sample
                from features1 and the j-th sample from features2
        '''
        kernel = dot(swapaxes(features2, 0, 1), features1)
        return kernel

    def response(self, features):
        ''' gives response of the predictor to the features

            Keyword Arguments:
            features -- 2D arrays of feature vectors; 0th dim features,
                1st dim samples

            returns: 1D array with response for all feature samples
        '''
        response = dot(self.parameter, features)
        return response

    def grad(self, error_fct_derivativ, features):
        '''gives gradient of the linear predictor with respect to its parameter
           based on the error function derivativ with respect to the
           predictor function

            Keyword Arguments:
            features -- 2D array of feature vectors; 0th dimension features,
                1st dimension samples
            error_fct_derivativ -- 1D array; derivativ of the error function
                with respect to the predictor function (at current vaulue);
                dimension is sample size

            returns: single element list with 1D array (dimension equiv. to
                self.parameter) as the batch gradient obtained by chainrule
                of error_fct_derivativ and gradient of the linear predictor
        '''
        return [dot(error_fct_derivativ, features.transpose())]

    def shift_param(self, step):
        ''' change parameter by amount of step

        Keyword Arguments:
        step -- single element list of 1D array wth dimension as self.parameter

        Side Effects:
        changes the instance variable parameter

        Exeptions:
        DimensionError -- dimension of step and self.parameter do not match
        '''
        step = step[0]
        if step.shape != self.parameter.shape:
            raise DimensionError
        else: self.parameter += step

class prodLinearPredictor(object):
    def __init__(self, dim, num_prod, mean=0, stdev=1):
        self.dim = dim
        self.num_prod = num_prod
        # create indivudual ArbiterPUFs
        self.indiv_linpredictor = [linearPredictor(dim, mean, stdev) for i in range(num_prod)]
        self._indiv_response = []
        self._response = []

    '''
    def getParam(self):
        delays=empty([self.numXOR, 2*self.numBits])
        parameter=empty([self.numXOR, self.numBits+1])
        for i in range(self.numXOR):
            delays[i,:]=self.indivArbiter[i].delays
            parameter[i,:]=self.indivArbiter[i].parameter
        return (delays, parameter)

    def setParam(self,delays,parameter):
        for i in range(self.numXOR):
            self.indivArbiter[i].delays=delays[i,:]
            self.indivArbiter[i].parameter=parameter[i,:]
    '''

    def response(self, features):
        self._indiv_response = empty([self.num_prod, features.shape[-1]])
        for predictor in range(self.num_prod):
            self._indiv_response[predictor, :] = dot(self.indiv_linpredictor[predictor].parameter,
                                                 features[predictor, :, :])
        self._response = prod(self._indiv_response, 0)
        return self._response

    def grad(self, error_fct_derivativ, features):
        grad = range(self.num_prod)
        for i in grad:
            grad[i] = dot(error_fct_derivativ / self._indiv_response[i, :] * self._response, swapaxes(features[i, :, :], 0, 1))
        return grad

    def shift_param(self, step):
        for predictor, indiv_step in enumerate(step):
            self.indiv_linpredictor[predictor].shift_param([indiv_step])

class bentLinearPredictor(object):
    def __init__(self, dim, num_prod, mean=0, stdev=1):
        self.dim = dim
        self.num_prod = num_prod
        # create indivudual ArbiterPUFs
        self.indiv_linpredictor = [linearPredictor(dim, mean, stdev) for i in range(num_prod)]
        self._indiv_response = []
        self._response = []

    '''
    def getParam(self):
        delays=empty([self.numXOR, 2*self.numBits])
        parameter=empty([self.numXOR, self.numBits+1])
        for i in range(self.numXOR):
            delays[i,:]=self.indivArbiter[i].delays
            parameter[i,:]=self.indivArbiter[i].parameter
        return (delays, parameter)

    def setParam(self,delays,parameter):
        for i in range(self.numXOR):
            self.indivArbiter[i].delays=delays[i,:]
            self.indivArbiter[i].parameter=parameter[i,:]
    '''

    def response(self, features):
        self._indiv_response = empty([self.num_prod, features.shape[-1]])

        for predictor in range(self.num_prod):
            self._indiv_response[predictor, :] = dot(self.indiv_linpredictor[predictor].parameter,
                                                 features[predictor, :, :])

        self._and_resp = empty([self.num_prod / 2, features.shape[-1]])
        for i in range (0, self.num_prod, 2):
            vfunc = vectorize(self.logicalAND)
            self._and_resp[i/2, :] = vfunc(self._indiv_response[i], self._indiv_response[i+1])
        self._response = prod(self._and_resp, 0)

        return self._response

    def logicalAND(self, a, b):
        if (sign(a) > 0 and sign(b) > 0):
            return (a * b)*(-1.)
        else:
            return abs(a) * abs(b)

    def grad(self, error_fct_derivativ, features):
        grad = range(self.num_prod)
        for i in grad:
            grad[i] = dot(error_fct_derivativ / self._indiv_response[i, :] * self._response, swapaxes(features[i, :, :], 0, 1))
        return grad

    def shift_param(self, step):
        for predictor, indiv_step in enumerate(step):
            self.indiv_linpredictor[predictor].shift_param([indiv_step])

class FFNeuralNet(object):
    def __init__(self, neurons, activation_fct, connectivity_matrices='all',
                 bias=False, mean=0, stdev=1):
        self.num_layer = len(neurons) - 1 #input layer not counted
        self.neurons = neurons
        self.activation_fct = activation_fct
        self._activation = range(self.num_layer)
        self._layerout = range(self.num_layer)
        self._delta = range(self.num_layer)
        self.bias = bias

        self.parameter = []
        # create random parameter (respectively weight) matrices
        for layer in range(self.num_layer):
                extra_neuron = (1 if (bias and layer > 0) else 0)
                self.parameter.append(random.normal(mean, stdev,
                              (neurons[layer + 1], neurons[layer] + extra_neuron)))
        # set parameter to zero, if there is no connection
        if connectivity_matrices != 'all':
            self.parameter *= connectivity_matrices

    def response(self, features):

        for layer, weights in enumerate(self.parameter):
            #calc the output of the neurons in the layer
            if layer == 0:
                self._layerout[0] = features
            else:
                out = self.activation_fct.calc(self._activation[layer - 1])
                if self.bias:
                    #add bias neuron with output 1
                    out = vstack((out, ones(out.shape[1])))
                self._layerout[layer] = out
            #calc the activation of the neurons in the next layer
            self._activation[layer] = dot(weights, self._layerout[layer])
        return self._activation[-1]

    def grad(self, error_fct_derivativ, features):
        '''it is important that response() is called in before, to have correct
            intermeditate resulst for the backprob'''
        grad = range(self.num_layer)
        self._delta[-1] = error_fct_derivativ
        for layer, weight in enumerate(self.parameter[1:].__reversed__()):
            # calc gradient of weights between layer
            grad[self.num_layer - 1 - layer] = dot(self._delta[-1 - layer], self._layerout[-1 - layer].transpose())
            # explicit representation of weight and delta as 2D arrays
            #if self.neurons[-1-layer] == 1:
            #    weight = weight.reshape(1, -1)
            #    self._delta[-1-layer] = self._delta[-1-layer].reshape(1, -1)
            #backprob delta
            self._delta[-2 - layer] = (self.activation_fct.grad(self._layerout[-1 - layer]) * dot(weight.transpose(), self._delta[-1 - layer]))
            #strip bias neurons
            if self.bias == True:
                self._delta[-2 - layer] = self._delta[-2 - layer][:-1, :]
        grad[0] = dot(self._delta[0], self._layerout[0].transpose())
        return grad

    def shift_param(self, step):
        for param_set, param_step in enumerate(step):
            if param_step.shape != self.parameter[param_set].shape:
                raise DimensionError
            else: self.parameter[param_set] += param_step

    def draw(self, xaxis):
        for layer in range(self.num_layer):
            f = plt.figure(layer)
            f.hold(True)
            for num, weight in enumerate(self.parameter[layer].flatten()):
                ax = plt.subplot(self.parameter[layer].shape[0], self.parameter[layer].shape[1], num + 1)
                plt.plot([xaxis], [weight] , '+k')
            f.show()

class Sigmoid(object):
    def __init__(self, scale_factor=1):
        self.scale = scale_factor

    def calc(self, input_array):
        out = 1. / (1 + numexp(-self.scale * input_array))
        return out

    def grad(self, input_array):
        out = input_array * (1. - input_array)
        return out

class Tanh(object):
    def __init__(self, scale_factor=1):
        self.scale = scale_factor

    def calc(self, input_array):
        out = tanh(-self.scale * input_array)
        return out

    def grad(self, input_array):
        out = (1. - input_array ** 2)
        return out

class LRError(object):
    def __init__(self, scale=1):
        self.sigmoid = Sigmoid(scale)

    def calc(self, targets, response):
        errors = (log(1 + numexp(-self.sigmoid.scale * targets.squeeze() * response.squeeze())))
        errors[isinf(errors)] = 1000
        return sum(errors)

    def grad(self, targets, response):
        class_prob = self.response_interpretation(response)
        error_fct_derivative = class_prob - (targets + 1) / 2
        return error_fct_derivative

    def response_interpretation(self, response):
        return self.sigmoid.calc(response)

class MSError(object):
    def calc(self, targets, response):
        error = sum((targets - response.squeeze()) ** 2) / targets.shape[0]
        return error

    def grad(self, targets, response):
        return response - targets

class MCError(object):

    def calc(self, targets, response):
        error = sum(1 - targets.squeeze() * sign(response.squeeze())) / 2
        return error

class MCC(object):

    def calc(self, targets, response):
        pass

class CRPset:
    def __init__(self, PUF, targets, challenges=''):
        self.PUF = PUF
        # if targets is an integer, so many CRPs are generated for the specified PUF, challenges specifies the type of challenge mapping
        if type(targets) == int:
            self.features = PUF.calcFeatures(PUF.generateChallenge(targets, challenges))
            (self.targets, dummy1, dummy2) = PUF.calcResponse(self.features)
        # else targets and challenges are taken as extern measurments
        else:
            self.targets = targets
            self.features = PUF.calcFeatures(challenges)

    def calcKernel(self, trainingsfeatures):
        self.kernel = self.PUF.calcKernel(trainingsfeatures, self.features)

    def saveKernel(self, fileout):
        try: self.kernel
        except:
            print 'use function calcKernel first'
        out = csv.writer(open(fileout, 'wb'), delimiter=" ")
        out.writerows(self.kernel)

class RProp(object):
    def __init__(self, dimension, initial_stepsize=1, etaminus=0.5, etaplus=1.2):
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.gradold = [ones(dim) for dim in dimension]
        self.stepold = [zeros(dim) for dim in dimension]
        self.stepsize = [ones(dim) * initial_stepsize for dim in dimension]
        self.step = [zeros(dim) for dim in dimension]

    def update_step(self, grad):

        for part_num, grad_part in enumerate(grad):
            stepindicator = sign(grad_part * self.gradold[part_num])

            self.stepsize[part_num][stepindicator > 0] *= self.etaplus
            self.stepsize[part_num][stepindicator < 0] *= self.etaminus

            self.step[part_num][stepindicator > 0] = -(self.stepsize[part_num][stepindicator > 0] * sign(grad_part[stepindicator > 0]))
            self.step[part_num][stepindicator < 0] = -self.stepold[part_num][stepindicator < 0]
            self.step[part_num][stepindicator == 0] = -self.stepsize[part_num][stepindicator == 0] * sign(grad_part[stepindicator == 0])

            self.gradold[part_num] = grad_part
            self.gradold[part_num][stepindicator < 0] = 0
            self.stepold[part_num] = self.step[part_num]

        return self.step

class GradientDescent(object):
    def __init__(self, learnrate=1):
        self.learnrate = learnrate

    def update_step(self, grad):
        step = []
        for component in grad:
            step.append(-component * self.learnrate)
        return step

class AnealingGradientDescent(object):
    def __init__(self, learnrate=1, decay=0.999):
        self.learnrate = learnrate
        self.decay = decay

    def update_step(self, grad):
        self.learnrate *= self.decay
        step = []
        for component in grad:
            step.append(-component * self.learnrate)
        return step

class SVMmodel:
    def __init__(self, PUFlearner, C=10):
        self.param = svm_parameter()
        self.param.kernel_type = PRECOMPUTED
        self.param.svm_type = C_SVC
        self.param.C = C
        self.PUFlearner = PUFlearner

    def build(self):
        self.PUFlearner.trainset.calcKernel(self.PUFlearner.trainset.features)
        numeration = arange(1, self.PUFlearner.trainset.kernel.shape[0] + 1)[:, newaxis]
        problem = svm_problem(self.PUFlearner.trainset.targets, concatenate((numeration, self.PUFlearner.trainset.kernel), 1))
        self.model = svm_model(problem, self.param)

    def predict(self, testProb):
        testProb.trainset.calcKernel(self.PUFlearner.trainset.features)
        for i in range(testProb.trainset.kernel.shape[0]):
            testProb.binResponse[i] = self.model.predict(concatenate((array([1]), testProb.trainset.kernel[i, :])))

class Trainable(object):

    def current_error(self):
        pass

    def update(self, step):
        pass

    def evaluate_lesson(self):
        pass

class BasicTrainable(Trainable):
    def __init__(self, trainset, model, errorfct):
        self.trainset = trainset
        self.model = model
        self.errorfct = errorfct

    def response(self, param={}):
        features = param['features'] if ('features' in param) else self.trainset.features
        return self.model.response(features)

    def current_error(self, param={}):
        targets = param['targets'] if ('targets' in param) else self.trainset.targets
        response = self.response(param) if ('features' in param) else self.response()
        return self.errorfct.calc(targets, response)

    def update(self, step):
        self.model.shift_param(step)
        #self.response = self.model.response(self.trainset.features)

    def evaluate_lesson(self):
        return self.current_error

    def response_interpretation(self, param={}):
        features = param['features'] if ('features' in param) else self.trainset.features
        response = param['response'] if ('response' in param) else self.response({'features':features})
        return self.errorfct.response_interpretation(response)

class Learner(Trainable):
    def __init__(self, closure_fct, trainable):
        self.lesson = trainable
        self.closure_fct = closure_fct

    ''' old version
    def getmodel(self):
        return self.lesson.model

    def setmodel(self, model):
        self.lesson.model = model

    model = property(getmodel, setmodel)
    '''

    def gettrainset(self):
        return self.lesson.trainset

    def settrainset(self, trainset):
        self.lesson.trainset = trainset

    def response(self, param={}):
        return self.lesson.response(param)

    def current_error(self, param={}):
        return self.lesson.current_error(param)

    def update(self, step):
        self.lesson.update(step)

    def response_interpretation(self, param={}):
        return self.lesson.response_interpretation(param)

    def evaluate_lesson(self):
        pass



class GradLearner(Learner):
    def __init__(self, lesson, grad_learning_strategy, closure_fct):
        self.gradstrat = grad_learning_strategy
        Learner.__init__(self, closure_fct, lesson)

    def evaluate_lesson(self):
        grad = []
        while self.closure_fct(self.lesson, grad):
            grad = self.grad()
            self.update(self.gradstrat.update_step(grad))
        return self.current_error()

    def grad(self):
        return self.lesson.model.grad(
                                      self.lesson.errorfct.grad(
                                                  self.lesson.trainset.targets,
                                                  self.lesson.response()
                                                                ),
                                      self.lesson.trainset.features
                                      )

class CrossValidation(Learner):
    def __init__(self, sampler, error_fct=False, trainable=[], trainset=[], xfold=10):
        Learner.__init__(self, [], trainable)
        self.xfold = xfold
        self.validation_set = []
        self.sublessons = []
        self.trainset = trainset
        self.sampler = sampler
        self.error_fct = error_fct
        self.val_ind = []
        self.train_ind = []
        if trainable: self.connect_trainable(trainable)

    def connect_trainable(self, trainable):
        self.lesson = trainable
        if not(self.trainset):
            self.trainset = trainable.trainset
        trainable.trainset = []
        for subpop in range(self.xfold):
            self.sublessons.append(deepcopy(trainable))
            (train_ind, val_ind) = self.sampler.sample_set()
            self.train_ind.append(train_ind)
            self.val_ind.append(val_ind)
            self.sublessons[-1].settrainset(self.trainset.sample_subset(train_ind))
            self.validation_set.append(self.trainset.sample_subset(val_ind))
        self.lesson.settrainset(self.trainset)

    def current_error(self):
        return mean(self.indiv_current_error())

    def update(self, step):
        self.lesson.update(step)
        for subpop in range(self.xfold):
            self.sublessons[subpop].lesson.update(step)

    def evaluate_lesson(self):
        self.lesson.evaluate_lesson()
        for subprob in range(self.xfold):
            self.sublessons[subprob].evaluate_lesson()
        print self.lesson.lesson.model.parameter
        print self.indiv_current_error()
        return self.current_error()

    def indiv_current_error(self):
        indiv_errors = []
        for subpop in range(self.xfold):
            learner = self.sublessons[subpop]
            val_set = self.validation_set[subpop]
            if self.error_fct:
                indiv_errors.append(self.error_fct.calc(val_set.targets,
                                learner.response({'features':val_set.features})
                                                   )
                                         / val_set.targets.shape[0])
            else:
                indiv_errors.append(learner.current_error(
                    {'targets':val_set.targets, 'features':val_set.features}))
        return indiv_errors

    def indiv_response(self, param={}):
        return [lesson.response(param) for lesson in self.sublessons]

    def indiv_response_interpretation(self, param={}):
        return [lesson.response_interpretation(param) for lesson in self.sublessons]

    def sublearner_information(self):
        '''indiv_errors_train = []
        indiv_errors = []
        for subpop in range(self.xfold):
            learner = self.sublessons[subpop]
            val_set = self.validation_set[subpop]
            indiv_errors_train.append(self.error_fct.calc(
                        learner.gettrainset().targets, learner.response()
                                                         )
                                         / learner.gettrainset().targets.shape[0])
            indiv_errors.append(self.error_fct.calc(val_set.targets,
                                learner.response({'features':val_set.features})
                                                   )
                                         / val_set.targets.shape[0])
        '''
        error_train = {}
        error_val = {}
        response_val = {}
        response_train = {}
        errors = self.indiv_current_error()

        for subpop in range(self.xfold):
            response_interprets = self.sublessons[subpop].response_interpretation(
                            {'features':self.validation_set[subpop].features})

            for list_pos, sample_ind in enumerate(self.val_ind[subpop]):
                try:
                    error_val[sample_ind].append(errors[subpop])
                except KeyError:
                    error_val[sample_ind] = [errors[subpop]]
                try:
                    response_val[sample_ind].append(
                                        response_interprets[list_pos])
                except KeyError:
                    response_val[sample_ind] = [
                                        response_interprets[list_pos]]

            response_interprets = self.sublessons[subpop].response_interpretation()
            for list_pos, sample_ind in enumerate(self.train_ind[subpop]):
                try:
                    error_train[sample_ind].append(errors[subpop])
                except KeyError:
                    error_train[sample_ind] = [errors[subpop]]
                try:
                    response_train[sample_ind].append(
                                        response_interprets[list_pos])
                except KeyError:
                    response_train[sample_ind] = [
                                        response_interprets[list_pos]]
        '''
        mean_error_val = {}
        mean_error_train = {}
        mean_response_val = {}
        mean_response_train = {}
        for key in range(self.trainset.targets.shape[0]):
            mean_error_val[key] = mean(error_val[key])
            mean_error_train[key] = mean(error_train[key])
            mean_response_val[key] = mean(response_val[key])
            mean_response_train[key] = mean(response_train[key])
        '''
        out = {'error_train': error_train, #'mean_error_train':mean_error_train,
               'error_val':error_val, #'mean_error_val':mean_error_val,
               'response_val':response_val, #'mean_response_val':mean_response_val,
               'response_train':response_train} #, 'mean_response_train': mean_response_train}

        print errors
        print mean(errors)
        return out
        #for subpop in self.lesson:
        #    print (subpop.response_interpretation({'features':self.trainset.features}))

class Closures(object):
    def __init__(self, stop_iteration=1E5000, accuracy=0.001):
        self.mc_error = MCError()
        self.ms_error = MSError()
        self.iteration_count = 0
        self.stop_grad = 0.0001
        self.accuracy = accuracy
        self.stop_iteration = stop_iteration
        self.error = 1E5000

    def reset(self):
        self.iteration_count = 0
        self.error = 1E5000

    def __call__(self, lesson, grad):
        return self.grad_performance_stop(lesson, grad)

    def mc_zero(self, lesson, grad):
        error = self.mc_error.calc(lesson.trainset.targets, lesson.response)
        print  error / lesson.trainset.targets.shape[0]
        return  error != 0

    def num_iterations(self, lesson, grad):
        self.iteration_count += 1
        if self.iteration_count % 50 == 1:
            self.error = self.mc_error.calc(lesson.trainset.targets, lesson.response())
            #lesson.model.draw(self.iteration_count)
            print self.iteration_count, self.error
        return self.iteration_count != self.stop_iteration + 1

    def grad_performance_stop(self, lesson, grad):

        self.iteration_count += 1
        if grad:
            abs_grad = [abs(grad_part) for grad_part in grad]
            total_grad = sum(sum(abs_grad).flatten())
        else:
            total_grad = self.stop_grad + 1
        train_performance = self.mc_error.calc(lesson.trainset.targets,
                                               lesson.response()
                                               ) / lesson.trainset.targets.shape[0]

        if self.iteration_count % 500 == 1:
            print self.iteration_count, total_grad, train_performance

        return ((total_grad > self.stop_grad)
                and (train_performance > self.accuracy)
                and (self.iteration_count < self.stop_iteration))

class TrainData(object):
    ''' TrainData acts as a container for traindata and will give methods for
        reading in data

        attributes:
        features -- array of features, where last dimension runs over samples
        targets -- 1D array of results for all samples
    '''
    def __init__(self, features=empty(0), targets=empty(0)):
        self._features = features
        self.targets = targets
        self.scaling = (0, 1)
        self.offset_features = False
        self.samplesize = targets.shape[0] if features.any() else 0

    @property
    def features(self):
        return self._features

    def scale_self(self):
        data_mean = mean(self._features, axis= -1)
        data_std = std(self._features, axis= -1)
        self.scaling = (data_mean, data_std)
        self._features = self.scale_same(self._features)

    def scale_same(self, features):
        scaled_features = empty(features.shape)
        for ind in range(features.shape[-1]):
            scaled_features[:, ind] = (features[:, ind] - self.scaling[0]) / self.scaling[1]
        if self.offset_features:
            scaled_features = concatenate((scaled_features, ones((1, scaled_features.shape[-1]))))
        return scaled_features

    def add_offsetfeature(self):
        self.offset_features = True
        self._features = concatenate((self._features, ones((1, self.samplesize))))

    def load(self, file):
        csvreader = csv.reader(open(file), delimiter=' ')
        filecontent = array([i for i in csvreader], dtype='d')
        self.targets = array(filecontent[:, 0])
        self._features = array(filecontent[:, 1:].transpose())

    def feature_subset(self, index):
        return TrainData(self.features[index], self.targets)

    def sample_subset(self, index):
        return TrainData((self.features.transpose()[index]).transpose(),
                                                        self.targets[index])

    def group_same_samples(self, indices=[]):
        indices = indices if indices else range(self._features.shape[-1])
        sample_groups = []
        while indices:
            existed = False
            sample = indices.pop()
            for group in sample_groups:
                if all(self._features[:, group[0]] == self._features[:, sample]):
                    group.append(sample)
                    existed = True
                    break
            if not(existed):
                sample_groups.append([sample])
        mean_value = [mean([self.targets[ind] for ind in group]) for group in sample_groups]
        return (sample_groups, mean_value)

    def stratificate(self, bin_borders, characteristic, items):
        ''' returns list of lists, each list containing items with characteristic
            between bin_borders
        '''
        bins = []
        characteristic = array(characteristic)
        low_border = bin_borders[0]
        for up_border in bin_borders[1:]:
            bins.append(list(nonzero((characteristic >= low_border) &
                                             (characteristic < up_border))[0]))
            low_border = up_border
        bins[-1] += list(nonzero(characteristic == up_border)[0])

        for strat_group in bins:
            for list_pos, item_num in enumerate(strat_group):
                strat_group[list_pos] = items[item_num]

        return bins

class SubSampling(object):

    '''SubSampling provides a generator
    '''

    def __init__(self, set_list, inv_ratio, deterministic=False):
        self.count = 0
        self.inv_ratio = inv_ratio
        self.set_list = set_list
        if deterministic: rd.seed(1234567890)
        self.sliced_set_list = [self.slider(self.slice(set))
                                                     for set in self.set_list]



    def slice(self, set):
        ''' generates self.inv_ratio lists, even randomly filled with the
            members of set
        '''
        rd.shuffle(set)
        div = around(linspace(0, len(set), min(len(set), self.inv_ratio) + 1)).astype('int')
        return [set[div[i]:div[i + 1]] for i in range(len(div) - 1)]

    def sample_set(self):
        shuffled_set = []
        for set_num, set in enumerate(self.sliced_set_list):
            try:
                shuffled_set.append(set.next())
            except StopIteration:
                resliced_set = self.slider(self.slice(self.set_list[set_num]))
                shuffled_set.append(resliced_set.next())
                self.sliced_set_list[set_num] = resliced_set
        # create two sets
        val_set_ind = []
        train_set_ind = []
        for response_category in shuffled_set:
            val_set_ind += [element for element_group in response_category.pop()
                                    for element in element_group]
            train_set_ind += [element for slices in response_category
                                      for element_group in slices
                                      for element in element_group]
        return (train_set_ind, val_set_ind)

    def slider(self, str):
        ''' gives a generator of all sequences emerged of str by sliding the
            start point
        '''
        for slidepoint in range(self.inv_ratio):
            yield str[slidepoint:] + str[:slidepoint]
