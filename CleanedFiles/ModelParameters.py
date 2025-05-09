from abc import abstractmethod


class ModelParams:
    @abstractmethod
    def setParams(self, paramsDict):
        pass

    @abstractmethod
    def getParams(self):
        pass

    def getThreshold(self):
        return self.threshold


class NeuralNetworkParams(ModelParams):
    def __init__(self):
        self.threshold = 0.5
        self.hidden_layer_sizes = (4, 5, 2)
        self.activation_fn = 'relu'
        self.solver = 'adam'
        self.alpha = 0.0001
        self.batch_size = 'auto'
        self.learning_rate = 'adaptive'
        self.learning_rate_init = 0.003
        self.max_iter = 200
        self.early_stopping = False
        self.momentum = 0.9
        self.n_iter_no_change = 100

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.hidden_layer_sizes = paramsDict.get('hidden_layer_sizes', self.hidden_layer_sizes)
        self.activation_fn = paramsDict.get('activation_fn', self.activation_fn)
        self.solver = paramsDict.get('solver', self.solver)
        self.alpha = paramsDict.get('alpha', self.alpha)
        self.batch_size = paramsDict.get('batch_size', self.batch_size)
        self.learning_rate = paramsDict.get('learning_rate', self.learning_rate)
        self.learning_rate_init = paramsDict.get('learning_rate_init', self.learning_rate_init)
        self.max_iter = paramsDict.get('max_iter', self.max_iter)
        self.early_stopping = paramsDict.get('early_stopping', self.early_stopping)
        self.momentum = paramsDict.get('momentum', self.momentum)
        self.n_iter_no_change = paramsDict.get('n_iter_no_change', self.n_iter_no_change)

    def getParams(self):
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation_fn': self.activation_fn,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'early_stopping': self.early_stopping,
            'momentum': self.momentum,
            'n_iter_no_change': self.n_iter_no_change
        }


class LogisticRegressionParams(ModelParams):
    def __init__(self):
        self.threshold = 0.35
        self.C = 5e-3
        self.max_iter = 200
        self.solver = 'saga'
        self.penalty = 'l1'
        self.tol = 1e-8
        self.class_weight = 'balanced'
        self.fit_intercept = False

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.C = paramsDict.get('C', self.C)
        self.max_iter = paramsDict.get('max_iter', self.max_iter)
        self.solver = paramsDict.get('solver', self.solver)
        self.penalty = paramsDict.get('penalty', self.penalty)
        self.tol = paramsDict.get('tol', self.tol)
        self.class_weight = paramsDict.get('class_weight', self.class_weight)
        self.fit_intercept = paramsDict.get('fit_intercept', self.fit_intercept)

    def getParams(self):
        return {
            'C': self.C,
            'max_iter': self.max_iter,
            'solver': self.solver,
            'penalty': self.penalty,
            'tol': self.tol,
            'class_weight': self.class_weight,
            'fit_intercept': self.fit_intercept
        }


class NaiveBayesParams(ModelParams):
    def __init__(self):
        self.threshold = 0.5
        self.var_smoothing = 1e-9
        self.priors = [0.3, 0.7]

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.var_smoothing = paramsDict.get('var_smoothing', self.var_smoothing)
        self.priors = paramsDict.get('priors', self.priors)

    def getParams(self):
        return {
            'var_smoothing': self.var_smoothing,
            'priors': self.priors
        }


class GradientBoostingMachineModelParams(ModelParams):
    def __init__(self):
        self.threshold = 0.6
        self.n_estimators = 100
        self.learning_rate = 0.1
        self.max_depth = 5

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.n_estimators = paramsDict.get('n_estimators', self.n_estimators)
        self.learning_rate = paramsDict.get('learning_rate', self.learning_rate)
        self.max_depth = paramsDict.get('max_depth', self.max_depth)

    def getParams(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth
        }


class XGBoostParams(ModelParams):
    def __init__(self):
        self.threshold = 0.55
        self.n_estimators = 100
        self.learning_rate = 0.1
        self.max_depth = 7

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.n_estimators = paramsDict.get('n_estimators', self.n_estimators)
        self.learning_rate = paramsDict.get('learning_rate', self.learning_rate)
        self.max_depth = paramsDict.get('max_depth', self.max_depth)

    def getParams(self):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth
        }


class CatBoostParams(ModelParams):
    def __init__(self):
        self.threshold = 0.5
        self.iterations = 500
        self.learning_rate = 0.009
        self.depth = 6
        self.l2_leaf_reg = 3
        self.subsample = 1
        self.verbose = 0

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.iterations = paramsDict.get('iterations', self.iterations)
        self.learning_rate = paramsDict.get('learning_rate', self.learning_rate)
        self.depth = paramsDict.get('depth', self.depth)
        self.l2_leaf_reg = paramsDict.get('l2_leaf_reg', self.l2_leaf_reg)
        self.subsample = paramsDict.get('subsample', self.subsample)
        self.verbose = paramsDict.get('verbose', self.verbose)

    def getParams(self):
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'subsample': self.subsample,
            'verbose': self.verbose
        }


class RandomForestParams(ModelParams):
    def __init__(self):
        self.threshold = 0.48
        self.oob_score = True
        self.n_jobs = -1
        self.n_estimators = 1000
        self.max_depth = 9
        self.min_samples_split = 9
        self.min_samples_leaf = 3
        self.random_state = 42

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.oob_score = paramsDict.get('oob_score', self.oob_score)
        self.n_jobs = paramsDict.get('n_jobs', self.n_jobs)
        self.n_estimators = paramsDict.get('n_estimators', self.n_estimators)
        self.max_depth = paramsDict.get('max_depth', self.max_depth)
        self.min_samples_split = paramsDict.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = paramsDict.get('min_samples_leaf', self.min_samples_leaf)
        self.random_state = paramsDict.get('random_state', self.random_state)

    def getParams(self):
        return {
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }


class SGDClassifierParams(ModelParams):
    def __init__(self):
        self.threshold = 0.5
        self.loss = 'log_loss'
        self.alpha = 1.65
        self.max_iter = 100000

    def setParams(self, paramsDict):
        self.threshold = paramsDict.get('threshold', self.threshold)
        self.loss = paramsDict.get('loss', self.loss)
        self.alpha = paramsDict.get('alpha', self.alpha)
        self.max_iter = paramsDict.get('max_iter', self.max_iter)

    def getParams(self):
        return {
            'loss': self.loss,
            'alpha': self.alpha,
            'max_iter': self.max_iter
        }