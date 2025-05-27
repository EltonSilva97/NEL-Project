from sklearn.base import BaseEstimator
from slim_gsgp.config.gsgp_config import *
from slim_gsgp.main_gsgp import gsgp
import os
from slim_gsgp.algorithms.GSGP.gsgp import GSGP
from slim_gsgp.config.gsgp_config import *
from slim_gsgp.utils.logger import log_settings
from slim_gsgp.utils.utils import get_terminals, validate_inputs, generate_random_uniform
from typing import Callable

class GSGPWrapper(BaseEstimator):
    """
    To make GSGP compatible with scikit-learn and to apply GridSearch
    """

    def __init__(self,
                 # Search Space parameters
                 init_depth=gsgp_pi_init["init_depth"],
                 tree_functions=list(FUNCTIONS.keys()),
                 tree_constants=[float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
                 prob_const=gsgp_pi_init["p_c"],
                 
                 # GP instance parameters
                 pop_size=gsgp_parameters["pop_size"],
                 p_xo=gsgp_parameters["p_xo"],
                 initializer=gsgp_parameters["initializer"],
                 tournament_size=2,
                 ms_lower=0,
                 ms_upper=1,
                 reconstruct=gsgp_solve_parameters["reconstruct"],
                 
                 # Solve settings
                 n_iter=gsgp_solve_parameters["n_iter"],
                 elitism=gsgp_solve_parameters["elitism"],
                 n_elites=gsgp_solve_parameters["n_elites"],
                 test_elite=gsgp_solve_parameters["test_elite"],
                 
                 # Other parameters
                 fitness_function=gsgp_solve_parameters["ffunction"],
                 minimization=True,
                 dataset_name=None,
                 log_path=None,
                 log_level=gsgp_solve_parameters["log"],
                 verbose=gsgp_solve_parameters["verbose"],
                 n_jobs=gsgp_solve_parameters["n_jobs"],
                 seed=gsgp_parameters["seed"]
                 ):
        # Initialize all parameters
        # Search Space parameters
        self.init_depth = init_depth
        self.tree_constants = tree_constants 
        self.tree_functions = tree_functions
        self.prob_const = prob_const

        # GSGP instance parameters
        self.pop_size = pop_size
        self.p_xo = p_xo
        self.initializer = initializer
        self.tournament_size = tournament_size
        self.ms_lower = ms_lower
        self.ms_upper = ms_upper
        self.reconstruct = reconstruct

        # Solve settings
        self.n_iter = n_iter
        self.elitism = elitism
        self.n_elites = n_elites
        self.test_elite = test_elite

        # Other parameters
        self.fitness_function = fitness_function
        self.minimization = minimization
        self.dataset_name = dataset_name
        self.log_path = log_path
        self.log_level = log_level
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.seed = seed

        # Initialize model-related attributes 
        self.model = None
        self.best_solution = None
        self.training_history = None
     
    def fit(self, X_train, y_train, X_test=None, y_test=None):
        self.model = gsgp(
            # Search Space
            init_depth=self.init_depth,
            tree_constants=self.tree_constants,
            tree_functions=self.tree_functions,
            prob_const=self.prob_const,
            
            # Problem Instance
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=self.dataset_name,
            fitness_function=self.fitness_function,
            minimization=self.minimization,
            
            # GSGP instance
            pop_size=self.pop_size,
            p_xo=self.p_xo,
            initializer=self.initializer,
            tournament_size=self.tournament_size,
            ms_lower=self.ms_lower,
            ms_upper=self.ms_upper,
            reconstruct=self.reconstruct,
            
            # Solve settings
            n_iter=self.n_iter,
            elitism=self.elitism,
            n_elites=self.n_elites,
            test_elite=self.test_elite,
            log_path=self.log_path,
            log_level=self.log_level,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            seed=self.seed
        )
        
        # Store results
        self.best_solution = self.model
        self.best_fitness = self.model.fitness
    
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit first")
        predictions = self.model.predict(X)
        return predictions
    
    def score(self, X, y):
        """
        Grid search uses this to evaluate the model
        Score Value higher is better.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        
        preds = self.predict(X)
        fitness_function = fitness_function_options[self.fitness_function]
        # Return negative because sklearn maximizes scores but we're minimizing error
        return -fitness_function(y, preds)