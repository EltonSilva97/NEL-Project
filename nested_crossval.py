import os
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
import pandas as pd
from sklearn.base import clone
import numpy as np

def run_nested_cv_for_model(X,y, estimator, param_grid, outer_cv, inner_cv, model_name="model"):
   
    if  isinstance(outer_cv, int):
        outer_cv=KFold(n_splits=outer_cv,shuffle=True, random_state=42)

    outer_scores=[]
    best_params_list=[]
    all_grid_results=[]
    summary=[]
    results_dir=f"./results/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    for i, (train_idx,test_idx) in enumerate( outer_cv.split(X)):
        print(f"\n Outer fold {i+1}")
        X_train,X_test=X[train_idx],X[test_idx]
        y_train,y_test=y[train_idx],y[test_idx]

        grid_search=GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=None, #will use the scoring of the model
            verbose=0,
            n_jobs=1
        )
        grid_search.fit(X_train,y_train)
         # Print results
        best_params=grid_search.best_params_
        best_params_list.append(best_params)
        print("\nGrid search completed!")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {grid_search.best_score_}")

        
        print(f"Training with the best hyparameters of {i+1} fold")
        best_model=clone(estimator).set_params(**best_params)
        best_model.verbose=1
        best_model.log_path=f"{results_dir}/logs_best_params_fold_{i+1}"
        best_model.fit(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
        test_score=best_model.score(X_test,y_test)
        outer_scores.append(test_score)

        print(f"Outer fold {i+1} - Best parameters: {best_params}")
        print(f"Outer fold {i+1} - Test score: {test_score:.4f}")
        
        fold_results=pd.DataFrame(grid_search.cv_results_)
        fold_results["outer_fold"]=i+1
        all_grid_results.append(fold_results)
        summary_row=best_params.copy()
        summary_row["outer_fold"]=i+1
        summary_row["grid_score"]=grid_search.best_score_
        summary_row["test_score"]=test_score
        summary.append(summary_row)
    
    general_summary = {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'best_params_by_fold': best_params_list
    }

    all_grid_results=pd.concat(all_grid_results)
    all_grid_results.to_csv(f"{results_dir}/all_folds_grid_search_results.csv")
    
    return pd.DataFrame(summary), general_summary, all_grid_results
    

# def evaluate_models_with_nested_cv(X,y, models_param_grid, outer_cv, inner_cv):
#     results=[]

#     for model_param in models_param_grid:

#         print(f"Results for model {model_param.name}")
#         grid_search=GridSearchCV(
#             estimator=model_param.estimator,
#             param_grid=model_param.parameter_grid,
#             cv=inner_cv,
#             verbose=2,
#             n_jobs=1)
#         outer_results=cross_val_score(grid_search,X,y,outer_cv,n_jobs=1)
        
#         for idx,score in enumerate(outer_results):
#             results.append({"model":model_param.name,"fold":idx+1, "score":score })
    
#         mean=np.mean(outer_results)
#         std=np.std(outer_results)
#         print(f"Results: mean={mean:.4f}, std={std:.4f}")
#     return pd.DataFrame(results)



