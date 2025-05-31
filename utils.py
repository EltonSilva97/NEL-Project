import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math
from math import ceil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

def plot_fitness_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    test_color = 'red'

    # Determine number of rows needed (3 plots per row)
    n_rows = (n_folds + 2) // 3  # +2 to properly ceil-divide

    # Create subplot figure
    fig = make_subplots(rows=n_rows, cols=3, subplot_titles=[f'Fold {i}' for i in range(1, n_folds+1)])

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        # Calculate subplot position
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Add Train trace
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 5].values,
            mode='lines',
            name=f'Train',
            line=dict(color=train_color),
            showlegend=(fold == 1)  # Show legend only once
        ), row=row, col=col)

        # Add Test trace
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 8].values,
            mode='lines',
            name=f'Test',
            line=dict(color=test_color),
            showlegend=(fold == 1)
        ), row=row, col=col)

    # Update overall layout
    fig.update_layout(
        height=400 * n_rows, width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Train vs Test Fitness ({dataset_name} dataset)',
        yaxis_range=[0, None]
    )

    fig.show()



def plot_fitness_and_size_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    test_color = 'red'
    size_color = 'green'

    # Determine number of rows needed (2 plots per fold: fitness and size)
    n_rows = n_folds
    
    # Create subplot figure (2 columns per fold - fitness and size)
    fig = make_subplots(
        rows=n_rows, 
        cols=2,
        subplot_titles=[f'Fold {i} - Fitness' if j%2==0 else f'Fold {i} - Size' 
                        for i in range(1, n_folds+1) for j in range(2)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        
        # Each fold gets its own row with 2 plots
        row = fold
        
        # Add Train trace to fitness plot (column 1)
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 5].values,
            mode='lines',
            name=f'Train (Fold {fold})',
            line=dict(color=train_color),
            showlegend=(fold == 1)  # Show legend only once
        ), row=row, col=1)

        # Add Test trace to fitness plot (column 1)
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 8].values,
            mode='lines',
            name=f'Test (Fold {fold})',
            line=dict(color=test_color),
            showlegend=(fold == 1)
        ), row=row, col=1)
        
        # Add Size trace to size plot (column 2)
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 9].values,
            mode='lines',
            name=f'Size (Fold {fold})',
            line=dict(color=size_color),
            showlegend=(fold == 1)
        ), row=row, col=2)

    # Update overall layout
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        showlegend=True,
        margin=dict(t=60),
        title_text=f'{model_name} Evolution - {dataset_name} dataset',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1/n_rows,  # Adjust based on number of rows
            xanchor='center',
            x=0.5
        )
    )
    
    # Set y-axis range for fitness plots
    for i in range(1, n_folds+1):
        fig.update_yaxes(range=[0, None], row=i, col=1)

    fig.show()


def plot_population_diversity_overlay(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'

    fig = go.Figure()

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 11].values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color)
        ))

    fig.update_layout(
        height=400, width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)',
        yaxis_range=[0, None],
        xaxis_title='Generation',
        yaxis_title='Fitness Standard Deviation'
    )

    fig.show()

def plot_population_diversity(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    print("Aqui")

    n_rows = (n_folds + 2) // 3

    fig = make_subplots(rows=n_rows, cols=3, subplot_titles=[f'Fold {i}' for i in range(1, n_folds+1)])

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        
        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')
        
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 11].values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color),
            showlegend=False
        ), row=row, col=col)

    fig.update_layout(
        height=400 * n_rows, width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)',
        yaxis_range=[0, None]
    )

    fig.show()


def plot_niche_entropy_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'

    # Calculate number of rows needed (3 plots per row)
    n_rows = ceil(n_folds / 3)

    # Create subplot figure with titles per fold
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Log file not found for fold {fold} at {log_path}')
            continue

        # Check if the required column 10 (11th column) exists
        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: not enough columns ({df_log.shape[1]} columns, need at least 11)')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        # Determine subplot position
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Add Train trace for Niche entropy
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 10].values,
            mode='lines',
            name=f'Train (Fold {fold})',
            line=dict(color=train_color),
            showlegend=(fold == 1)  # Show legend only once
        ), row=row, col=col)

        # Update each subplot axis titles (optional)
        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Entropy', row=row, col=col, range=[0, None])

    # Update overall figure layout
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Niche entropy ({dataset_name} dataset)'
    )

    fig.show()


def plot_solution_size_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'

    # Calcular número de linhas necessárias (3 por linha)
    n_rows = ceil(n_folds / 3)

    # Criar figura de subplots
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Ficheiro não encontrado para o fold {fold} em {log_path}')
            continue

        if df_log.shape[1] <= 9:
            print(f'Fold {fold}: ficheiro tem apenas {df_log.shape[1]} colunas (precisa pelo menos de 10)')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        # Definir posição do subplot
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Adicionar linha da solução
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 9].values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color),
            showlegend=False
        ), row=row, col=col)

        # Eixos individuais de cada subplot
        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Nodes count', range=[0, None], row=row, col=col)

    # Layout geral
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Solution size ({dataset_name} dataset)'
        # ,yaxis_type='log'  # Descomenta se quiseres log scale global
    )

    fig.show()

def plot_avg_fitness(model_name, n_folds, dataset_name='sustavianfeed'):
    
    gen_by_gen = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:,12]==2] \
               .drop_duplicates(subset=4, keep='last')
        # iterate through the rows and collect size values by generation
        for _, row in df.iterrows():
            generation = int(row[4])     # generation
            train_fitness = float(row[5])
            test_fitness = float(row[8])  # fitness
            gen_by_gen.setdefault(generation, {'train': [], 'test': []})
            gen_by_gen[generation]['train'].append(train_fitness)
            gen_by_gen[generation]['test'].append(test_fitness)


    gens = sorted(gen_by_gen)
    train_mean = np.array([np.mean(gen_by_gen[g]['train']) for g in gens])
    train_std  = np.array([np.std (gen_by_gen[g]['train']) for g in gens])
    test_mean  = np.array([np.mean(gen_by_gen[g]['test'])  for g in gens])
    test_std   = np.array([np.std (gen_by_gen[g]['test'])  for g in gens])

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=gens, y=train_mean + train_std, line=dict(color='rgba(0,0,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=gens, y=train_mean - train_std, fill='tonexty',
                             fillcolor='rgba(0,0,180,0.2)', line=dict(color='rgba(0,0,255,0)'),
                             name='Fitness Train Std Dev'))
    fig.add_trace(go.Scatter(x=gens, y=train_mean, line=dict(color='blue'), name='Avg Train Fitness'))

    # Test ribbon and mean
    fig.add_trace(go.Scatter(x=gens, y=test_mean + test_std,
                             line=dict(color='rgba(255,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=gens, y=test_mean - test_std,
                             fill='tonexty', fillcolor='rgba(255,0,0,0.2)',
                             line=dict(color='rgba(255,0,0,0)'), name='Fitness Test Std Dev'))
    fig.add_trace(go.Scatter(x=gens, y=test_mean,
                             line=dict(color='red'), name='Avg Test Fitness'))


    fig.update_layout(
        title=f"{model_name} avg. fitness and Std Dev ({dataset_name}) over {n_folds} folds",
        xaxis_title="Generation", yaxis_title="Fitness",
        width=700, height=400
    )
    fig.show()


def plot_avg_size(model_name, n_folds, dataset_name='sustavianfeed'):
    gen_by_gen = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:,12]==2] \
               .drop_duplicates(subset=4, keep='last')

        for _, row in df.iterrows():
            generation = int(row[4]) # generation
            size = float(row[9])     # node count
            gen_by_gen.setdefault(generation, []).append(size)


    gens = sorted(gen_by_gen)
    mean = np.array([np.mean(gen_by_gen[g]) for g in gens])
    std = np.array([np.std (gen_by_gen[g]) for g in gens])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=mean + std, line=dict(color='rgba(0,180,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=gens, y=mean - std, fill='tonexty',
                             fillcolor='rgba(0,135,0,0.2)', line=dict(color='rgba(0,180,0,0)'),
                             name='Size Std Dev'))
    fig.add_trace(go.Scatter(x=gens, y=mean, line=dict(color='green'), name='Avg Size'))

    fig.update_layout(
        title=f"{model_name} avg. node size and Std Dev ({dataset_name}) over {n_folds} folds",
        xaxis_title="Generation", yaxis_title="Node count",
        width=700, height=400
    )
    fig.show()

   

def plot_population_semantic_diversity_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'red'

    # Calcular número de linhas necessárias (3 por linha)
    n_rows = ceil(n_folds / 3)

    # Criar figura de subplots
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Ficheiro não encontrado para o fold {fold} em {log_path}')
            continue

        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: ficheiro tem apenas {df_log.shape[1]} colunas (precisa pelo menos de 11)')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        # Extrair valores e remover 'tensor()'
        div_vector_log = df_log.iloc[:, 10].values
        div_vector_values = np.array([float(str(x).replace('tensor(', '').replace(')', '')) for x in div_vector_log])

        # Definir posição do subplot
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Adicionar linha da diversidade semântica
        fig.add_trace(go.Scatter(
            y=div_vector_values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color),
            showlegend=False
        ), row=row, col=col)

        # Eixos individuais de cada subplot
        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Semantic Diversity', range=[0, None], row=row, col=col)

    # Layout geral
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Semantic Diversity ({dataset_name} dataset)'
    )

    fig.show()


def plot_population_fitness_diversity_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'orange'

    # Three subplots per row
    n_rows = ceil(n_folds / 3)

    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'Ficheiro não encontrado para o fold {fold} em {log_path}')
            continue

        if df_log.shape[1] <= 11:
            print(f'Fold {fold}: ficheiro tem apenas {df_log.shape[1]} colunas (precisa pelo menos de 12)')
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df_log = df_log[df_log.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df_log = df_log.drop_duplicates(subset=4, keep='last')

        # Define subplot position
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Adicionar trace de fitness diversity
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 11].values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color),
            showlegend=False
        ), row=row, col=col)

        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Fitness Std. Dev.', range=[0, None], row=row, col=col)

    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)'
    )

    fig.show()


def plot_NN_loss_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    test_color = 'red'
    n_rows = math.ceil(n_folds / 3)

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds+1)]
    )

    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing log for fold {fold}: {path}")
            continue

        if 'epoch' not in df.columns or 'train_loss' not in df.columns or 'test_loss' not in df.columns:
            print(f"Fold {fold} missing required columns.")
            continue

        epochs = df['epoch']
        train_loss = df['train_loss']
        test_loss = df['test_loss']

        row = math.ceil(fold / 3)
        col = (fold - 1) % 3 + 1

        # Train line
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode='lines',
            line=dict(color=train_color),
            name='Train' if fold == 1 else None,
            showlegend=(fold == 1)
        ), row=row, col=col)

        # Test line
        fig.add_trace(go.Scatter(
            x=epochs, y=test_loss,
            mode='lines',
            line=dict(color=test_color),
            name='Test' if fold == 1 else None,
            showlegend=(fold == 1)
        ), row=row, col=col)

        fig.update_xaxes(title_text='Epoch', row=row, col=col)
        fig.update_yaxes(title_text='Loss', row=row, col=col)

    fig.update_layout(
        height=350 * n_rows,
        width=1000,
        title_text=f'{model_name} – Train vs Test Loss over Epochs ({dataset_name})',
        margin=dict(t=50)
    )
    fig.show()

def plot_NN_average_loss(model_name, n_folds, dataset_name='sustavianfeed'):
    train_dict = {}
    test_dict = {}

    for fold in range(1, n_folds + 1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}.'
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        if 'epoch' not in df.columns or 'train_loss' not in df.columns or 'test_loss' not in df.columns:
            continue
        
        # Iterate through df and add the respctive losses to the dict
        for _, row in df.iterrows():
            epoch = int(row['epoch'])
            train_loss = float(row['train_loss'])
            test_loss = float(row['test_loss'])
            train_dict.setdefault(epoch, []).append(train_loss)
            test_dict.setdefault(epoch, []).append(test_loss)

    # Calculate the average loss and std dev for each epoch
    epochs = sorted(train_dict.keys())
    mean_train = np.array([np.mean(train_dict[e]) for e in epochs])
    std_train = np.array([np.std(train_dict[e]) for e in epochs])
    mean_test = np.array([np.mean(test_dict[e]) for e in epochs])
    std_test = np.array([np.std(test_dict[e]) for e in epochs])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=epochs, y=mean_train + std_train,
        line=dict(color='rgba(0,0,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(
        x=epochs, y=mean_train - std_train,
        fill='tonexty',
        fillcolor='rgba(0,0,180,0.2)',
        line=dict(color='rgba(0,0,255,0)'),
        name='Train RMSE Std Dev'
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=mean_train,
        line=dict(color='blue'),
        name='Avg Train Loss'
    ))

    fig.add_trace(go.Scatter(
        x=epochs, y=mean_test + std_test,
        line=dict(color='rgba(255,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(
        x=epochs, y=mean_test - std_test,
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name='Test Loss Std Dev'
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=mean_test,
        line=dict(color='red'),
        name='Avg Test Loss'
    ))

    fig.update_layout(
        title=f"{model_name} – Average Train & Test Loss per Epoch ({dataset_name})",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        width=800,
        height=500
    )
    fig.show()

def plot_param_boxplots(csv_path, model_name="sustavianfeed"):

    # Load data
    grid_results = pd.read_csv(csv_path)
    grid_results["rmse"] = -grid_results["mean_test_score"]

    # Plot 1: by learning rate
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="param_optimizer__lr", y="rmse", data=grid_results)
    plt.title(f"Loss by Learning Rate ({model_name})")
    plt.xlabel("Learning Rate")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.show()

    # Plot 2: by weight decay
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="param_optimizer__weight_decay", y="rmse", data=grid_results)
    plt.title(f"Loss by Weight Decay ({model_name})")
    plt.xlabel("Weight Decay")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()

def stat_test_lr_(csv_path):
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

def plot_rmse_by_lr_weight_decay_combo(csv_path, model_name="NN"):
    df = pd.read_csv(csv_path)
    df["rmse"] = -df["mean_test_score"]

    # Create readable labels for each parameter combination
    df["param_combo"] = df.apply(
        lambda row: f"lr={row['param_optimizer__lr']}, wd={row['param_optimizer__weight_decay']}",
        axis=1
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="param_combo", y="rmse")
    plt.title(f"Loss by (lr, weight_decay) Combination Across Folds — {model_name}")
    plt.xlabel("(lr, weight_decay)")
    plt.ylabel("Loss")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def stat_test_param_combinations(csv_path):
    df = pd.read_csv(csv_path)
    df["rmse"] = -df["mean_test_score"]
    df["param_combo"] = df.apply(
        lambda row: f"lr={row['param_optimizer__lr']}, wd={row['param_optimizer__weight_decay']}",
        axis=1
    )

    groups = df.groupby("param_combo")["rmse"].apply(list)
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
