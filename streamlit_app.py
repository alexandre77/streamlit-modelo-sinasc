import streamlit as st

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import auc
#from sklearn.metrics import plot_roc_curve
from scikitplot.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split



"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""



def le_tabela(arquivo):
    df = pd.read_csv(arquivo)
    return df.values.tolist()


def grava_tabela(dados, arquivo, colunas):

    df = pd.DataFrame( dados , columns = colunas)
    df.to_csv(arquivo, index=False)


def planilha_em_dataset(database, colunas):

    df = pd.read_csv(database)

    df = df[colunas]
    
    dataset = df.to_numpy()

    return dataset


def dataset_em_planilha(database, dataset, colunas):
    df = pd.DataFrame(dataset, columns = colunas)

    df.to_csv(database, index=False)


def modelo_roc(dataset, classificador, variavel = [], selecionadas = [], titulo = "Predição de mortalidade materna (2020-2021) por MLP"):

    if len(variavel) == 0:

        # ultima coluna para a classe
        y = dataset[:, -1]
        #y = 0*(y==2) + 1*(y==4)  # trocar rotulos de 2 e 4 para 0 e 1

        if len(selecionadas) == 0:
            # demais colunas para atributos (exceto a ultima da classe)
            X = dataset[:, 0:-1]
        else:
            X = dataset[:, selecionadas]

        #X, y = X[y != 2], y[y != 2]
        n_samples, n_features = X.shape

        # Add noisy features
        #random_state = np.random.RandomState(0)
        #X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

        # #############################################################################
        # Classification and ROC analysis

        # Run classifier with cross-validation and plot ROC curves
        #cv = StratifiedKFold(n_splits=6)
        cv = StratifiedKFold(n_splits=5)
        if 'SVM' in classificador:
            #classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
            #classifier = svm.SVC(kernel='linear', probability=True)
            classifier = svm.SVC()
        elif 'MLP' in classificador:
            classifier = MLPClassifier(random_state=1, max_iter=5000)
        else:
            classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        for i, (train, test) in enumerate(cv.split(X, y)):
            classifier.fit(X[train], y[train])
            viz = plot_roc_curve(classifier, X[test], y[test],
                                 name='ROC fold {}'.format(i),
                                 alpha=0.3, lw=1, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(mean_fpr, mean_tpr, color='b',
                label=r'Média ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
               title=titulo)
        ax.legend(loc="lower right")
        #plt.show()
        #st.pyplot(fig)
        return fig


    else:
        H, W = dataset.shape

        for ii in range(W-1):

            # ultima coluna para a classe
            y = dataset[:, -1]
            #y = 0*(y==2) + 1*(y==4)  # trocar rotulos de 2 e 4 para 0 e 1

            # demais colunas para atributos (exceto a ultima da classe)
            #X = dataset[:, 0:-1]
            X = dataset[:, ii]  # seleciona um unico atributo
            X = X[:,np.newaxis]
            #print(X.shape)

            #X, y = X[y != 2], y[y != 2]
            n_samples, n_features = X.shape

            # Add noisy features
            #random_state = np.random.RandomState(0)
            #X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

            # #############################################################################
            # Classification and ROC analysis

            # Run classifier with cross-validation and plot ROC curves
            #cv = StratifiedKFold(n_splits=6)
            cv = StratifiedKFold(n_splits=5)
            if 'SVM' in classificador:
                #classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state)
                #classifier = svm.SVC(kernel='linear', probability=True)
                classifier = svm.SVC()
            elif 'MLP' in classificador:
                classifier = MLPClassifier(random_state=1, max_iter=5000)
            else:
                classifier = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            fig, ax = plt.subplots()
            for i, (train, test) in enumerate(cv.split(X, y)):
                classifier.fit(X[train], y[train])
                viz = plot_roc_curve(classifier, X[test], y[test],
                                     name='ROC fold {}'.format(i),
                                     alpha=0.3, lw=1, ax=ax)
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

            ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Média ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

            ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
                   title="Predição de mortalidade materna (2020-2021) - %s" %(variavel[ii]))
            ax.legend(loc="lower right")
            #plt.show()
            #plt.savefig('model-var/%02d-%s.png' %(ii, variavel[ii]))
            #st.pyplot(fig)
            
            #break
            
            return fig



dataset = planilha_em_dataset('database-dic-features.csv', ['consultas_ult', 'escmae2010_ult', 'estcivmae_ult', 'gestacao_ult', 'num_grav_1', 'num_grav_2', 'num_grav_n', 'idademae_ult', 'idanomal_ult', 'locnasc_ult', 'num_nasc_rep', 'num_parto_1', 'num_parto_2', 'num_filhos_peso_baixo', 'num_filhos_peso_insuf', 'num_filhos_peso_adeq', 'num_filhos_peso_macr', 'racacormae_ult', 'semagestac_ult', 'stcesparto_ult', 'sttrabpart_ult', 'tpapresent_ult', 'tpnascassi_ult', 'tprobson_ult', 'gestacao_reagrupada', 'qtdfilmort_reagrupada', 'st_alto_risco_mortalidade', 'obito'])


with st.echo(code_location='below'):
    op_clas = st.selectbox(
         'Escolha o classificador',
         ('MLP', 'SVM', 'Random Forest'))
    
    st.write('Classificador escolhido:', op_clas)
    
    lst_vars = ['consultas_ult', 'escmae2010_ult', 'estcivmae_ult', 'gestacao_ult', 'num_grav_1', 'num_grav_2', 'num_grav_n', 'idademae_ult', 'idanomal_ult', 'locnasc_ult', 'num_nasc_rep', 'num_parto_1', 'num_parto_2', 'num_filhos_peso_baixo', 'num_filhos_peso_insuf', 'num_filhos_peso_adeq', 'num_filhos_peso_macr', 'racacormae_ult', 'semagestac_ult', 'stcesparto_ult', 'sttrabpart_ult', 'tpapresent_ult', 'tpnascassi_ult', 'tprobson_ult', 'gestacao_reagrupada', 'qtdfilmort_reagrupada', 'st_alto_risco_mortalidade']
    
    def lst_vars_to_pos(variavel):
        for i, v in enumerate(lst_vars):
            if v == variavel:
                return i

    op_vars = st.multiselect(
        'Escolha as variáveis',
        lst_vars
        )

    st.write('Variáveis escolhidas:', op_vars)
        
    #op_vars_lst = list(op_vars.values())
    op_vars_lst_pos = map(lst_vars_to_pos, lst_vars)

    
    #fig = modelo_roc(dataset, variavel = ['consultas_ult', 'escmae2010_ult', 'estcivmae_ult', 'gestacao_ult', 'num_grav_1', 'num_grav_2', 'num_grav_n', 'idademae_ult', 'idanomal_ult', 'locnasc_ult', 'num_nasc_rep', 'num_parto_1', 'num_parto_2', 'num_filhos_peso_baixo', 'num_filhos_peso_insuf', 'num_filhos_peso_adeq', 'num_filhos_peso_macr', 'racacormae_ult', 'semagestac_ult', 'stcesparto_ult', 'sttrabpart_ult', 'tpapresent_ult', 'tpnascassi_ult', 'tprobson_ult', 'gestacao_reagrupada', 'qtdfilmort_reagrupada', 'st_alto_risco_mortalidade', 'obito'])
    
    #fig = modelo_roc(dataset)
    fig = modelo_roc(dataset, op_clas, selecionadas = op_vars_lst_pos, titulo = f'Predição de mortalidade materna - {op_clas}')

    st.pyplot(fig)
