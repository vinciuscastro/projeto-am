# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_outliers(df, column):
    """
    Função que plota um boxplot de uma coluna de um DataFrame
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'{column}')
    plt.show()


def plot_missing_values(df):
    """
    Função que plota um mapa de calor com os valores ausentes de um DataFrame
    """
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Valores Ausentes')
    plt.show()


def plot_count_values(df, column):
    """ 
    Função que plota a contagem de valores únicos em uma coluna de um DataFrame
    """
    plt.figure(figsize=(8,6))
    sns.countplot(x=column, data=df)
    plt.title(f'Contagem de Valores Únicos em {column}')
    plt.xlabel(column)
    plt.ylabel('Contagem')
    plt.show()
