import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal


def plot_fitness_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot Train vs Test Fitness using matplotlib, replicating the Plotly version.
    """
    train_color = 'blue'
    test_color = 'red'

    # Determine number of rows needed (3 plots per row)
    n_rows = (n_folds + 2) // 3  # +2 to properly ceil-divide
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # keep the log-level-2 rows and drop duplicates
        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        ax = axes[fold - 1]
        generations = range(len(df_log))

        ax.plot(generations, df_log.iloc[:, 5].values, color=train_color, label='Train' if fold == 1 else "")
        ax.plot(generations, df_log.iloc[:, 8].values, color=test_color, label='Test' if fold == 1 else "")

        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Fitness')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Train vs Test Fitness ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    fig.legend(loc='upper center', ncol=2)
    plt.show()


def plot_fitness_and_size_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot Train & Test Fitness and Solution Size per fold using matplotlib.
    """
    train_color = 'blue'
    test_color = 'red'
    size_color = 'green'

    n_rows = n_folds
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=False)

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax_fitness = axes[fold - 1, 0] if n_rows > 1 else axes[0]
        ax_size = axes[fold - 1, 1] if n_rows > 1 else axes[1]

        # Plot fitness
        ax_fitness.plot(generations, df_log.iloc[:, 5].values, color=train_color, label='Train' if fold == 1 else "")
        ax_fitness.plot(generations, df_log.iloc[:, 8].values, color=test_color, label='Test' if fold == 1 else "")
        ax_fitness.set_title(f'Fold {fold} - Fitness')
        ax_fitness.set_ylabel('Fitness')

        # Plot size
        ax_size.plot(generations, df_log.iloc[:, 9].values, color=size_color, label='Size' if fold == 1 else "")
        ax_size.set_title(f'Fold {fold} - Size')
        ax_size.set_ylabel('Nodes count')

    # Set common X-labels
    for ax in axes.flatten():
        ax.set_xlabel('Generation')

    fig.suptitle(f'{model_name} Evolution - {dataset_name} dataset', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.legend(loc='upper center', ncol=3)
    plt.show()


def plot_population_diversity_overlay_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot population fitness diversity overlay for all folds on the same axes.
    """
    train_color = 'blue'

    plt.figure(figsize=(12, 5))

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax = plt.gca()
        ax.plot(generations, df_log.iloc[:, 11].values, color=train_color, label=f'Fold {fold}')

    plt.title(f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Standard Deviation')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_population_diversity_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot population fitness diversity per fold in a grid of subplots.
    """
    train_color = 'blue'
    n_rows = (n_folds + 2) // 3
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax = axes[fold - 1]
        ax.plot(generations, df_log.iloc[:, 11].values, color=train_color)
        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Fitness Std Dev')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    plt.show()


def plot_niche_entropy_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot niche entropy per fold in a grid of subplots.
    """
    train_color = 'blue'
    n_rows = math.ceil(n_folds / 3)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: not enough columns ({df_log.shape[1]})')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax = axes[fold - 1]
        ax.plot(generations, df_log.iloc[:, 10].values, color=train_color)
        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Entropy')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Niche Entropy ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    plt.show()


def plot_solution_size_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot solution size per fold in a grid of subplots.
    """
    train_color = 'blue'
    n_rows = math.ceil(n_folds / 3)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        if df_log.shape[1] <= 9:
            print(f'Fold {fold}: only {df_log.shape[1]} columns')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2]
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax = axes[fold - 1]
        ax.plot(generations, df_log.iloc[:, 9].values, color=train_color)
        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Nodes Count')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Solution Size ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    plt.show()


def plot_avg_fitness_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot average train & test fitness with std dev ribbons across folds.
    """
    gen_by_gen = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:, 12] == 2].drop_duplicates(subset=4, keep='last')
        for _, row in df.iterrows():
            g = int(row[4])
            train_val = float(row[5])
            test_val = float(row[8])
            gen_by_gen.setdefault(g, {'train': [], 'test': []})
            gen_by_gen[g]['train'].append(train_val)
            gen_by_gen[g]['test'].append(test_val)

    gens = sorted(gen_by_gen.keys())
    train_mean = np.array([np.mean(gen_by_gen[g]['train']) for g in gens])
    train_std = np.array([np.std(gen_by_gen[g]['train']) for g in gens])
    test_mean = np.array([np.mean(gen_by_gen[g]['test']) for g in gens])
    test_std = np.array([np.std(gen_by_gen[g]['test']) for g in gens])

    plt.figure(figsize=(10, 6))
    # Train ribbon
    plt.fill_between(gens, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.plot(gens, train_mean, color='blue', label='Avg Train Fitness')
    # Test ribbon
    plt.fill_between(gens, test_mean - test_std, test_mean + test_std, color='red', alpha=0.2)
    plt.plot(gens, test_mean, color='red', label='Avg Test Fitness')

    plt.title(f"{model_name} avg. fitness and Std Dev ({dataset_name}) over {n_folds} folds")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_avg_size_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot average solution size with std dev ribbon across folds.
    """
    gen_by_gen = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:, 12] == 2].drop_duplicates(subset=4, keep='last')
        for _, row in df.iterrows():
            g = int(row[4])
            size = float(row[9])
            gen_by_gen.setdefault(g, []).append(size)

    gens = sorted(gen_by_gen.keys())
    mean_vals = np.array([np.mean(gen_by_gen[g]) for g in gens])
    std_vals = np.array([np.std(gen_by_gen[g]) for g in gens])

    plt.figure(figsize=(10, 6))
    plt.fill_between(gens, mean_vals - std_vals, mean_vals + std_vals, color='green', alpha=0.2)
    plt.plot(gens, mean_vals, color='green', label='Avg Size')

    plt.title(f"{model_name} avg. node size and Std Dev ({dataset_name}) over {n_folds} folds")
    plt.xlabel('Generation')
    plt.ylabel('Node Count')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_population_semantic_diversity_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot population semantic diversity per fold in subplots.
    """
    train_color = 'red'
    n_rows = math.ceil(n_folds / 3)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: only {df_log.shape[1]} columns')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2].drop_duplicates(subset=4, keep='last')
        # Convert tensor strings to floats if necessary
        div_vals = []
        for x in df_log.iloc[:, 10].values:
            val = float(str(x).replace('tensor(', '').replace(')', ''))
            div_vals.append(val)
        generations = range(len(div_vals))

        ax = axes[fold - 1]
        ax.plot(generations, div_vals, color=train_color)
        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Semantic Diversity')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Population Semantic Diversity ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    plt.show()


def plot_population_fitness_diversity_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot population fitness diversity per fold using matplotlib.
    """
    train_color = 'orange'
    n_rows = math.ceil(n_folds / 3)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=True)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        if df_log.shape[1] <= 11:
            print(f'Fold {fold}: only {df_log.shape[1]} columns')
            continue

        df_log = df_log[df_log.iloc[:, 12] == 2].drop_duplicates(subset=4, keep='last')
        generations = range(len(df_log))

        ax = axes[fold - 1]
        ax.plot(generations, df_log.iloc[:, 11].values, color=train_color)
        ax.set_title(f'Fold {fold}')
        if (fold - 1) % n_cols == 0:
            ax.set_ylabel('Fitness Std Dev')
        ax.set_xlabel('Generation')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)', y=1.02)
    fig.tight_layout()
    plt.show()


def plot_NN_loss_logs_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot Train vs Test Loss over epochs for each fold using matplotlib.
    """
    train_color = 'blue'
    test_color = 'red'
    n_rows = math.ceil(n_folds / 3)
    n_cols = 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows), sharey=False)
    axes = axes.flatten()

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f'Missing log for fold {fold}: {path}')
            continue

        if not {'epoch', 'train_loss', 'test_loss'}.issubset(df.columns):
            print(f'Fold {fold} missing required columns.')
            continue

        epochs = df['epoch']
        train_loss = df['train_loss']
        test_loss = df['test_loss']

        ax = axes[fold - 1]
        ax.plot(epochs, train_loss, color=train_color, label='Train' if fold == 1 else "")
        ax.plot(epochs, test_loss, color=test_color, label='Test' if fold == 1 else "")
        ax.set_title(f'Fold {fold}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    # Remove empty subplots
    for idx in range(n_folds, n_rows * n_cols):
        fig.delaxes(axes[idx])

    fig.suptitle(f'{model_name} - Train vs Test Loss over Epochs ({dataset_name})', y=1.02)
    fig.tight_layout()
    fig.legend(loc='upper center', ncol=2)
    plt.show()


def plot_NN_average_loss_matplotlib(model_name, n_folds, dataset_name='sustavianfeed'):
    """
    Plot average Train & Test Loss with Std Dev ribbons across folds using matplotlib.
    """
    train_dict = {}
    test_dict = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if not {'epoch', 'train_loss', 'test_loss'}.issubset(df.columns):
            continue

        for _, row in df.iterrows():
            e = int(row['epoch'])
            train_dict.setdefault(e, []).append(float(row['train_loss']))
            test_dict.setdefault(e, []).append(float(row['test_loss']))

    epochs = sorted(train_dict.keys())
    mean_train = np.array([np.mean(train_dict[e]) for e in epochs])
    std_train = np.array([np.std(train_dict[e]) for e in epochs])
    mean_test = np.array([np.mean(test_dict[e]) for e in epochs])
    std_test = np.array([np.std(test_dict[e]) for e in epochs])

    plt.figure(figsize=(10, 6))
    # Train ribbon
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='blue', alpha=0.2)
    plt.plot(epochs, mean_train, color='blue', label='Avg Train Loss')
    # Test ribbon
    plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, color='red', alpha=0.2)
    plt.plot(epochs, mean_test, color='red', label='Avg Test Loss')

    plt.title(f"{model_name} – Average Train & Test Loss per Epoch ({dataset_name})")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_param_boxplots_matplotlib(csv_path, model_name="sustavianfeed"):
    """
    Plot boxplots for RMSE by learning rate and weight decay using matplotlib.
    """
    grid_results = pd.read_csv(csv_path)
    grid_results['rmse'] = -grid_results['mean_test_score']

    # Plot by learning rate
    plt.figure(figsize=(8, 5))
    unique_lrs = sorted(grid_results['param_optimizer__lr'].unique())
    data_by_lr = [grid_results[grid_results['param_optimizer__lr'] == lr]['rmse'] for lr in unique_lrs]
    plt.boxplot(data_by_lr, labels=unique_lrs)
    plt.title(f"Loss by Learning Rate ({model_name})")
    plt.xlabel('Learning Rate')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()

    # Plot by weight decay
    plt.figure(figsize=(8, 5))
    unique_wds = sorted(grid_results['param_optimizer__weight_decay'].unique())
    data_by_wd = [grid_results[grid_results['param_optimizer__weight_decay'] == wd]['rmse'] for wd in unique_wds]
    plt.boxplot(data_by_wd, labels=unique_wds)
    plt.title(f"Loss by Weight Decay ({model_name})")
    plt.xlabel('Weight Decay')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.show()


def stat_test_lr_matplotlib(csv_path):
    """
    Perform Kruskal-Wallis H-test between learning rate groups and print results.
    """
    df = pd.read_csv(csv_path)
    df['rmse'] = -df['mean_test_score']
    groups = df.groupby('param_optimizer__lr')['rmse'].apply(list)

    if len(groups) < 2:
        print("Not enough learning rates to compare.")
        return

    stat, p = kruskal(*groups)
    print("Kruskal-Wallis H-test between learning rates:")
    print(f"Statistic = {stat:.4f}, p-value = {p:.4g}")
    if p < 0.05:
        print("→ Significant difference between learning rates (p < 0.05)")
    else:
        print("→ No significant difference between learning rates (p ≥ 0.05)")


def plot_rmse_by_lr_weight_decay_combo_matplotlib(csv_path, model_name="NN"):
    """
    Plot boxplot of RMSE by (lr, weight_decay) combinations using matplotlib.
    """
    df = pd.read_csv(csv_path)
    df['rmse'] = -df['mean_test_score']
    df['param_combo'] = df.apply(lambda row: f"lr={row['param_optimizer__lr']}, wd={row['param_optimizer__weight_decay']}", axis=1)

    combos = df['param_combo'].unique()
    data_by_combo = [df[df['param_combo'] == combo]['rmse'] for combo in combos]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data_by_combo, labels=combos, vert=True)
    plt.title(f"Loss by (lr, weight_decay) Combination Across Folds — {model_name}")
    plt.xlabel('(lr, weight_decay)')
    plt.ylabel('Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def stat_test_param_combinations_matplotlib(csv_path):
    """
    Perform Kruskal-Wallis H-test across (lr, weight_decay) combinations and print results.
    """
    df = pd.read_csv(csv_path)
    df['rmse'] = -df['mean_test_score']
    df['param_combo'] = df.apply(lambda row: f"lr={row['param_optimizer__lr']}, wd={row['param_optimizer__weight_decay']}", axis=1)

    groups = df.groupby('param_combo')['rmse'].apply(list)
    if len(groups) < 2:
        print("Need at least two combinations to compare.")
        return

    stat, p = kruskal(*groups)
    print("Kruskal-Wallis H-test across (lr, weight_decay) combinations:")
    print(f"Statistic = {stat:.4f}, p-value = {p:.4g}")
    if p < 0.05:
        print("→ Statistically significant difference between combinations.")
    else:
        print("→ No significant difference between combinations.")
