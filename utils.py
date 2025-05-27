import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
from math import ceil
import seaborn as sns

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

        # Calculate subplot position
        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        # Add Train trace
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 5].values,
            mode='lines',
            name=f'Train (Fold {fold})',
            line=dict(color=train_color),
            showlegend=(fold == 1)  # Show legend only once
        ), row=row, col=col)

        # Add Test trace
        fig.add_trace(go.Scatter(
            y=df_log.iloc[:, 8].values,
            mode='lines',
            name=f'Test (Fold {fold})',
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



def plot_population_semantic_diversity_logs(model_name, n_folds, dataset_name='sustavianfeed'):
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

        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: ficheiro tem apenas {df_log.shape[1]} colunas (precisa pelo menos de 11)')
            continue

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
    train_color = 'blue'

    # Calcular número de linhas necessárias (3 subplots por linha)
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

        if df_log.shape[1] <= 11:
            print(f'Fold {fold}: ficheiro tem apenas {df_log.shape[1]} colunas (precisa pelo menos de 12)')
            continue

        # Definir posição do subplot
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

        # Eixos de cada subplot
        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Fitness Std. Dev.', range=[0, None], row=row, col=col)

    # Layout geral
    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Fitness Diversity ({dataset_name} dataset)'
    )

    fig.show()
    

def plot_SLIM_fitness_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    test_color  = 'red'
    n_rows = math.ceil(n_folds / 3)

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds+1)]
    )

    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            print(f"Missing fold {fold} log: {path}")
            continue

        # keep the log-level-2 rows (per-gen best entries)
        df = df[df.iloc[:,12] == 2]
        # drop duplicate generations, keep the best row per generation
        df = df.drop_duplicates(subset=4, keep='last')

        gens  = df.iloc[:, 4]
        train = df.iloc[:, 5]
        test  = df.iloc[:, 8]

        # Define subplot's position
        row = math.ceil(fold / 3)
        col = (fold - 1) % 3 + 1

        fig.add_trace(go.Scatter(
            x=gens, y=train, mode='lines',
            line=dict(color=train_color),
            name=f'Train (Fold {fold})',
            showlegend=(fold==1)
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=gens, y=test, mode='lines',
            line=dict(color=test_color),
            name=f'Test (Fold {fold})',
            showlegend=(fold==1)
        ), row=row, col=col)

        # Subplot's axes
        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Fitness',    row=row, col=col)

    # General layout
    fig.update_layout(
        height=350 * n_rows, width=1000,
        title_text=f'{model_name} – Train vs Test Fitness ({dataset_name})',
        margin=dict(t=50)
    )
    fig.show()


def plot_SLIM_size_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    size_color = 'green'
    n_rows = math.ceil(n_folds / 3)

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds+1)]
    )

    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            print(f"Missing fold {fold} log: {path}")
            continue

        # — same filtering as before
        df = df[df.iloc[:, 12] == 2]
        df = df.drop_duplicates(subset=4, keep='last')

        gens = df.iloc[:, 4]   # generation
        size = df.iloc[:, 9]   # nodes count

        row = math.ceil(fold / 3)
        col = (fold - 1) % 3 + 1

        fig.add_trace(go.Scatter(
            x=gens,
            y=size,
            mode='lines',
            line=dict(color=size_color),
            name='Size' if fold == 1 else None,
            showlegend=(fold == 1)
        ), row=row, col=col)

        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Nodes count', row=row, col=col)

    fig.update_layout(
        height=350 * n_rows,
        width=1000,
        title_text=f'{model_name} – Solution Size over Generations ({dataset_name})',
        margin=dict(t=50)
    )
    fig.show()

def plot_SLIM_fitness_and_size_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'
    test_color = 'red'
    size_color = 'green'

    fig = make_subplots(
        rows=n_folds, cols=2,
        subplot_titles=sum(
            [[f'Fold {i} – Fitness', f'Fold {i} – Size'] for i in range(1, n_folds+1)],
            []
        )
    )

    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            print(f"[Missing fold {fold} log: {path}")
            continue

        df = df[df.iloc[:, 12] == 2]  
        df = df.drop_duplicates(subset=4, keep='last') 

        gens = df.iloc[:, 4]   # generation 
        size = df.iloc[:, 9]   # nodes count 
        test  = df.iloc[:, 8]  # test 
        train = df.iloc[:, 5]  # train

        row = fold

        fig.add_trace(go.Scatter(
            x=gens,
            y=test,
            mode='lines',
            line=dict(color=test_color),
            name='Test' if fold == 1 else None, 
            showlegend=(fold == 1) 
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=gens,
            y=train,
            mode='lines',
            line=dict(color=train_color),
            name='Train' if fold == 1 else None,
            showlegend=(fold == 1)
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=gens, 
            y=size, 
            mode='lines',
            line=dict(color=size_color),
            name='Size' if fold == 1 else None, 
            showlegend=(fold == 1) 
        ), row=row, col=2)


        fig.update_xaxes(title_text='Generation', row=row, col=1)
        fig.update_yaxes(title_text='Fitness',    row=row, col=1)
        fig.update_xaxes(title_text='Generation', row=row, col=2)
        fig.update_yaxes(title_text='Nodes count',row=row, col=2)
	 

    fig.update_layout(
        height=350 * n_folds,
        width=1000,
        title_text=f'SLIM – Solution Fitness vs Size over Generations ({dataset_name})',
        margin=dict(t=50)
    )
    fig.show()

def plot_SLIM_population_semantic_diversity_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'

    n_rows = ceil(n_folds / 3)

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'File missing for fold {fold} in {log_path}')
            continue

        if df_log.shape[1] <= 10:
            print(f'Fold {fold}: file only has {df_log.shape[1]} columns (needs at least 11)')
            continue

        df = df_log[df_log.iloc[:, 12] == 2]
        df = df.drop_duplicates(subset=4, keep='last')

        # generation vector for x-axis
        gens = df.iloc[:, 4].values

        raw = df.iloc[:, 10].astype(str)
        div_vector_values = raw.str.replace('tensor(', '').str.replace(')', '').astype(float).values

        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        fig.add_trace(go.Scatter(
            x=gens,                                      
            y=div_vector_values,
            mode='lines',
            name=f'Fold {fold}',
            line=dict(color=train_color),
            showlegend=False
        ), row=row, col=col)

        fig.update_xaxes(title_text='Generation', row=row, col=col)
        fig.update_yaxes(title_text='Semantic Diversity', range=[0, None], row=row, col=col)

    fig.update_layout(
        height=400 * n_rows,
        width=1000,
        margin=dict(t=50),
        title_text=f'{model_name} - Population Semantic Diversity ({dataset_name} dataset)'
    )

    fig.show()


def plot_SLIM_population_fitness_diversity_logs(model_name, n_folds, dataset_name='sustavianfeed'):
    train_color = 'blue'

    n_rows = ceil(n_folds / 3)

    fig = make_subplots(
        rows=n_rows, cols=3,
        subplot_titles=[f'Fold {i}' for i in range(1, n_folds + 1)]
    )

    for fold in range(1, n_folds + 1):
        log_path = f'./results/{model_name}/logs_best_params_fold_{fold}'

        try:
            df_log = pd.read_csv(log_path, header=None)
        except FileNotFoundError:
            print(f'File missing for fold {fold} in {log_path}')
            continue

        if df_log.shape[1] <= 11:
            print(f'Fold {fold}: file only has {df_log.shape[1]} columns (needs at least 12)')
            continue

        df = df_log[df_log.iloc[:, 12] == 2]
        df = df.drop_duplicates(subset=4, keep='last')

        # generation vector for x-axis
        gens = df.iloc[:, 4].values

        row = (fold - 1) // 3 + 1
        col = (fold - 1) % 3 + 1

        fig.add_trace(go.Scatter(
            x=gens,                                      
            y=df.iloc[:, 11].values,
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


def plot_SLIM_average_fitness(model_name, n_folds, dataset_name='sustavianfeed'):
    gen_by_gen = {}
    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:,12]==2] \
               .drop_duplicates(subset=4, keep='last')
        # iterate through the rows and collect size values by generation
        for _, row in df.iterrows():
            generation = int(row[4])
            train = float(row[5])
            test = float(row[8])

            gen_by_gen.setdefault(generation, {'train':[], 'test':[]})
            gen_by_gen[generation]['train'].append(train)
            gen_by_gen[generation]['test'].append(test)

    gens = sorted(gen_by_gen)
    train_mean = np.array([np.mean(gen_by_gen[g]['train']) for g in gens])
    train_std = np.array([np.std (gen_by_gen[g]['train']) for g in gens])
    test_mean = np.array([np.mean(gen_by_gen[g]['test']) for g in gens])
    test_std = np.array([np.std (gen_by_gen[g]['test']) for g in gens])

    fig = go.Figure()

    # train ribbon
    fig.add_trace(go.Scatter(x=gens, y=train_mean+train_std,
                             line=dict(color='rgba(0,0,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=gens, y=train_mean+train_std,
                             line=dict(color='rgba(0,0,255,0)'),
                             fill='tonexty',
                             fillcolor='rgba(0,0,180,0.2)',
                             name='Train Std Dev'))
    # train mean
    fig.add_trace(go.Scatter(x=gens, y=train_mean,
                             line=dict(color='blue'),
                             name='Train mean'))

    # test ribbon
    fig.add_trace(go.Scatter(x=gens, y=test_mean+test_std,
                             line=dict(color='rgba(255,0,0,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=gens, y=test_mean-test_std,
                             line=dict(color='rgba(255,0,0,0)'),
                             fill='tonexty',
                             fillcolor='rgba(255,0,0,0.2)',
                             name='Test Std Dev'))
    # test mean
    fig.add_trace(go.Scatter(x=gens, y=test_mean,
                             line=dict(color='red'),
                             name='Test mean'))

    fig.update_layout(
        title=f"SLIM avg. fitness and Std Dev ({dataset_name}) over {n_folds} folds",
        xaxis_title="Generation",
        yaxis_title="Fitness",
        width=700, height=400
    )
    fig.show()


def plot_SLIM_average_size(model_name, n_folds, dataset_name='sustavianfeed'):
    gen_by_gen = {}  

    for fold in range(1, n_folds+1):
        path = f'./results/{model_name}/logs_best_params_fold_{fold}'
        try:
            df = pd.read_csv(path, header=None)
        except FileNotFoundError:
            continue

        df = df[df.iloc[:,12]==2] \
               .drop_duplicates(subset=4, keep='last')

        for _, row in df.iterrows():
            generation = int(row[4])
            size = float(row[9])
            gen_by_gen.setdefault(generation, []).append(size)


    gens = sorted(gen_by_gen)
    mean_size  = np.array([np.mean(gen_by_gen[g]) for g in gens])
    std_size   = np.array([np.std (gen_by_gen[g]) for g in gens])

    fig = go.Figure()


    fig.add_trace(go.Scatter(
        x=gens, y=mean_size+std_size,
        line=dict(color='rgba(0,150,0,0)'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=gens, y=mean_size-std_size,
        line=dict(color='rgba(0,180,0,0)'),
        fill='tonexty',
        fillcolor='rgba(0,135,0,0.2)',
        name='Size Std Dev'
    ))

    fig.add_trace(go.Scatter(
        x=gens, y=mean_size,
        line=dict(color='green'),
        name='Mean size'
    ))

    fig.update_layout(
        title=f"SLIM avg. node size Std Dev ({dataset_name}) over {n_folds} folds",
        xaxis_title="Generation",
        yaxis_title="Node count",
        width=700, height=400
    )
    fig.show()

