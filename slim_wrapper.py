from sklearn.base import BaseEstimator
from slim_gsgp.config.slim_config import *
from slim_gsgp.main_slim import slim
import os
from slim_gsgp.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
import tempfile
from slim_gsgp.utils.logger import log_settings
from slim_gsgp.utils.utils import get_terminals, validate_inputs, generate_random_uniform, check_slim_version
from typing import Callable

class SLIMWrapper(BaseEstimator):
    """
    To make SLIM compatible with scikit-learn and to apply GridSearch
    """

    def __init__(self,
                 # Search Space parameters
                 init_depth=slim_gsgp_pi_init ["init_depth"],
                 tree_functions=list(FUNCTIONS.keys()),
                 tree_constants=[float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
                 prob_const=slim_gsgp_pi_init ["p_c"],
                 slim_version='SLIM*SIG2',
                 
                 # GP instance parameters
                 pop_size=slim_gsgp_parameters["pop_size"],
                 #p_xo=slim_gsgp_parameters["p_xo"],
                 initializer=slim_gsgp_parameters["initializer"],
                 tournament_size=2,
                 ms_lower=0,
                 ms_upper=1,
                 p_inflate=slim_gsgp_parameters["p_inflate"],
                 copy_parent = False,
                 
                 # Solve settings
                 n_iter=slim_gsgp_solve_parameters["n_iter"],
                 elitism=slim_gsgp_solve_parameters["elitism"],
                 n_elites=slim_gsgp_solve_parameters["n_elites"],
                 test_elite=slim_gsgp_solve_parameters["test_elite"],
                 reconstruct=slim_gsgp_solve_parameters["reconstruct"],
                 
                 # Other parameters
                 fitness_function=slim_gsgp_solve_parameters["ffunction"],
                 minimization=True,
                 dataset_name=None,
                 log_path=None,
                 log_level=slim_gsgp_solve_parameters["log"],
                 verbose=slim_gsgp_solve_parameters["verbose"],
                 n_jobs=slim_gsgp_solve_parameters["n_jobs"],
                 seed=slim_gsgp_parameters["seed"]
                 ):
        # Initialize all parameters
        # Search Space parameters
        self.init_depth = init_depth
        self.tree_constants = tree_constants 
        self.tree_functions = tree_functions
        self.prob_const = prob_const
        self.slim_version = slim_version

        # SLIM instance parameters
        self.pop_size = pop_size
        #self.p_xo = p_xo
        self.initializer = initializer
        self.tournament_size = tournament_size
        self.ms_lower = ms_lower
        self.ms_upper = ms_upper
        self.p_inflate = p_inflate
        self.copy_parent = copy_parent

        # Solve settings
        self.n_iter = n_iter
        self.elitism = elitism
        self.n_elites = n_elites
        self.test_elite = test_elite
        self.reconstruct = reconstruct

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
        self.model = slim(
            # Search Space
            init_depth=self.init_depth,
            tree_constants=self.tree_constants,
            tree_functions=self.tree_functions,
            prob_const=self.prob_const,
            slim_version=self.slim_version,
                
            # Problem Instance
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            dataset_name=self.dataset_name,
            fitness_function=self.fitness_function,
            minimization=self.minimization,
                
            # SLIM instance
            pop_size=self.pop_size,
            #p_xo=self.p_xo,
            initializer=self.initializer,
            tournament_size=self.tournament_size,
            ms_lower=self.ms_lower,
            ms_upper=self.ms_upper,
            p_inflate=self.p_inflate,
            copy_parent=self.copy_parent,
                
            # Solve settings
            n_iter=self.n_iter,
            elitism=self.elitism,
            n_elites=self.n_elites,
            test_elite=self.test_elite,
            log_path=self.log_path,
            log_level=self.log_level,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            reconstruct=self.reconstruct,
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