# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()
encoder_dict = {}  # Armazena encoders para cada coluna categórica


def remove_outliers(df, columns):
    """
    Função que utiliza o método IQR para remover outliers de um DataFrame	
    """
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def encoder_train(df_train):
    """
    Função que realiza o encoding de colunas categóricas de um DataFrame de treino, 
    guardando os encoders em um dicionário para uso posterior em um DataFrame de teste
    """
    categorical_cols = df_train.select_dtypes(include=['object']).columns

    encoded_dfs = []
    for col in categorical_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoded = ohe.fit_transform(df_train[[col]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
        encoded_df.index = df_train.index  
        encoder_dict[col] = ohe  
        encoded_dfs.append(encoded_df)

    df_train = df_train.drop(columns=categorical_cols)
    df_train = pd.concat([df_train] + encoded_dfs, axis=1)
    
    return df_train


def encoder_test(df_test):
    """
    Função que realiza o encoding de colunas categóricas de um DataFrame de teste, 
    utilizando os encoders guardados em um dicionário.
    """
    categorical_cols = df_test.select_dtypes(include=['object']).columns

    encoded_dfs = []
    for col in categorical_cols:
        ohe = encoder_dict[col]
        encoded = ohe.transform(df_test[[col]])
        encoded_df = pd.DataFrame(encoded, columns=[f"{col}_{cat}" for cat in ohe.categories_[0]])
        encoded_df.index = df_test.index
        encoded_dfs.append(encoded_df)

    df_test = df_test.drop(columns=categorical_cols)
    df_test = pd.concat([df_test] + encoded_dfs, axis=1)

    return df_test
    

def fill_missing_values(df):
    """
    Função que preenche valores faltantes de um DataFrame, separando colunas categóricas e numéricas, para
    preencher com a moda e a mediana, respectivamente.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    return df


def normalize_gender(df):
    """
    Função que normaliza os valores da coluna 'SEXO' de um DataFrame para 'm' e 'f'
    """
    df['SEXO'] = df['SEXO'].str.lower()
    df['SEXO'] = df['SEXO'].replace({'masculino': 'm', 'feminino': 'f'})
    return df

