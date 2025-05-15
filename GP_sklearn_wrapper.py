from sklearn.base import BaseEstimator
from slim_gsgp.config.gsgp_config import *
from slim_gsgp.main_gp import gp
import os
class GPWrapper(BaseEstimator):
    """
    To make GP compatible with scikit-learn and to apply GridSearch
    """

    def __init__(self,
                 #Search Space parameters
                 init_depth=gsgp_pi_init["init_depth"],
                 max_depth=5,
                 tree_functions=list(FUNCTIONS.keys()),
                 tree_constants=[float(key.replace("constant_", "").replace("_", "-")) for key in CONSTANTS],
                 prob_const=gsgp_pi_init["p_c"],
                 
                 #GP instance parameters
                 pop_size=gsgp_parameters["pop_size"],
                 p_xo=gsgp_parameters["p_xo"],
                 initializer=gsgp_parameters["initializer"],
                 tournament_size=2,
                 #Solve settings
                 n_iter=gsgp_solve_parameters["n_iter"],
                 elitism=gsgp_solve_parameters["elitism"],
                 n_elites=gsgp_solve_parameters["n_elites"],
                 #other parameters
                 fitness_function= gsgp_solve_parameters["ffunction"],
                 minimization=  True,
                 dataset_name="custom",
                 log_path="./logs",
                 log_level=gsgp_solve_parameters["log"],
                 #verbose=gsgp_solve_parameters["verbose"],
                 verbose=0,
                 n_jobs= gsgp_solve_parameters["n_jobs"],
                 seed=gsgp_parameters["seed"]
                 ):
        #Initialize all parameters
        #Search Space parameters
        self.init_depth=init_depth
        self.max_depth=max_depth
        self.tree_constants=tree_constants 
        self.tree_functions=tree_functions
        self.prob_const=prob_const

        #GP instance parameters
        self.pop_size=pop_size
        self.p_xo=p_xo
        self.initializer=initializer
        self.tournament_size=tournament_size

        # Solve settings
        self.n_iter=n_iter
        self.elitism=elitism
        self.n_elites=n_elites

        #other parameters
        self.fitness_function=fitness_function
        self.minimization=minimization
        self.dataset_name=dataset_name
        self.log_path=log_path
        self.log_level=log_level
        self.verbose=verbose
        self.n_jobs=n_jobs
        self.seed=seed

        #Initialize model-related attributes 
        self.model=None
        self.best_solution=None
        self.training_history = None
     
    
    def fit(self,X_train, y_train, X_test=None,y_test=None):

        # print("Processing the model ")
        # print(f"Shape: X={X_train.shape} and y={y_train.shape}")
        
        self.model = gp(
            # Search Space
            init_depth=self.init_depth,
            max_depth=self.max_depth,
            tree_constants=self.tree_constants,
            tree_functions=self.tree_functions,
            prob_const=self.prob_const,
            # Problem Instance
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,  # No validation during grid search fit
            y_test=y_test,
            dataset_name=self.dataset_name,
            fitness_function=self.fitness_function,
            minimization=self.minimization,
            # GP instance
            pop_size=self.pop_size,
            p_xo=self.p_xo,
            initializer=self.initializer,
            tournament_size=self.tournament_size,
            # Solve settings
            n_iter=self.n_iter,
            elitism=self.elitism,
            n_elites=self.n_elites,
            test_elite=False,  # No validation during fitting
            log_path=self.log_path,
            log_level=self.log_level,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            seed=self.seed
        )
        
        
        # Store results
        self.best_solution = self.model
        # print("Printing the model: ")
        # print(self.model)
        self.best_fitness = self.model.fitness
        # print(f"Best Fitness {self.best_fitness}")
 

    
    def predict(self,X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit first")
        preditions=self.model.predict(X)

        return preditions
    
    def score(self, X, y):
        """
        Grid search uses this to evaluate the model
        Score Value higher is better.
        """

        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        
        preds=self.predict(X)
        # print("Evaluating Score ")
        # print(f"Shape: X={X.shape} and y={y.shape}")
        
        fitness_function = fitness_function_options[self.fitness_function]
        # print(f"Score {-fitness_function(y,preds)} ")
        return -fitness_function(y,preds)
    