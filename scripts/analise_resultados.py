# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



def get_df_results(models, x_test, y_test):
    """
    Função que retorna um DataFrame com as métricas de avaliação de cada modelo
    """
    results = []
    for item in models:
        model = item[1]
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:,1]

        results.append([accuracy_score(y_test, y_pred),recall_score(y_test, y_pred),f1_score(y_test, y_pred), roc_auc_score(y_test, y_prob)])

    results = pd.DataFrame(results, columns=['Acurácia', 'F1 Score', 'ROC AUC', 'Recall'], index=[name for name, _ in models])
  
    return results


def plot_roc_curve(models, x_test, y_test):
    """
    Função que plota a curva ROC de cada modelo
    """
    plt.figure(figsize=(10,10))
    for item in models:
        model = item[1]
        name = item[0]
        
        y_prob = model.predict_proba(x_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label='{} (area = {:.2f})'.format(name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_confusion_matrix(models, x_test, y_test):
    """
    Função que plota a matriz de confusão de cada modelo, em um grid
    """
    num_models = len(models)
    cols = 3  
    rows = (num_models // cols) + (num_models % cols > 0)  

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # conta o número de subplots para criar o grid
    axes = axes.flatten()  

    for i, (name, model) in enumerate(models):
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=axes[i])
        axes[i].set_xlabel('Previsto')
        axes[i].set_ylabel('Real')
        axes[i].set_title(f'{name}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    