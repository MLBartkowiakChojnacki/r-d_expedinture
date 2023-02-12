# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:45:05 2023

@author: krzys
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scikit_posthocs import posthoc_dunn
from pingouin import kruskal, qqplot, pairwise_tests, mwu, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import dates


def set_sns_theme() -> None:
    sns.set_theme()
    sns.set_context("paper", font_scale = 1, rc={"lines.linewidth": 2.5})


def suppres_scientific_notation() -> None:
    np.set_printoptions(suppress=True)


#load data
def load_data(path: str
              , sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep = f"{sep}")
    return df


def shapiro_wilk_test(data_frame: pd.DataFrame
                      , feature: str) -> bool:
    #h0 = rozklad zgodny z normalnym
    #h1 = rozklad rozny od normalnego
    sw = shapiro(data_frame[feature])
    if sw[1] >= 0.05:
        return sw
    else:
        return sw


def plot_qqplot(data_frame: pd.DataFrame
                , feature: str) -> qqplot:
    qqplot(data_frame[feature], dist='norm')


def kw_test(data_frame:pd.DataFrame
            , feature: str
            , groups: str) -> pd.DataFrame:
    kw = kruskal(data_frame, dv = feature, between = groups)
    return kw


def kw_post_hoc(data_frame: pd.DataFrame
                , feature: str
                , groups: str) -> pd.DataFrame:
    dunn = posthoc_dunn(data_frame, feature, groups)
    return dunn


def mann_whitney_u_test(x: pd.Series, y: pd.Series, alternative: str = None) -> pd.DataFrame:
    if alternative:
        stat = mwu(x, y, alternative)
    else:
        stat = mwu(x, y)
    return stat


def wilcoxon_test(x: pd.Series, y: pd.Series, alternative: str) -> pd.DataFrame:
    if alternative:
        stat = wilcoxon(x, y, alternative)
    else:
        stat = wilcoxon(x, y)
    return stat


def draw_histograms(df: pd.DataFrame, variables: pd.Series, n_rows: int, n_cols: int):
    fig=plt.figure(figsize=(10, 10))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=10,ax=ax)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()
    plt.show()


def draw_plots(df: pd.DataFrame, variables: pd.Series, n_rows: int, n_cols: int, group: str):
    if group == None:
        fig = plt.figure(figsize=(10, 10))
        for i, var_name in enumerate(variables):
            ax=fig.add_subplot(n_rows,n_cols,i+1)
            try:
                df_grouped.reset_index(inplace=True)
            except ValueError:
                pass
            plt.bar(df_grouped['LOCATION'], df_grouped[var_name])
            ax.set_title(var_name)
            mf = mpl.ticker.ScalarFormatter(useMathText=True)
            mf.set_powerlimits((-2,2))
            plt.gca().yaxis.set_major_formatter(mf)
        fig.tight_layout()
        plt.show()  
    else:
        for country in group:
            fig = plt.figure(figsize=(10, 10))
            for i, var_name in enumerate(variables):
                ax=fig.add_subplot(n_rows,n_cols,i+1)
                plt.bar(df[df['LOCATION'] == country]['TIME'].unique(), df[df['LOCATION'] == country][var_name])
                ax.set_title(f'Location: {country} ' + var_name)
                mf = mpl.ticker.ScalarFormatter(useMathText=True)
                mf.set_powerlimits((-2,2))
                plt.gca().yaxis.set_major_formatter(mf)
            fig.tight_layout()
            plt.show()  


if __name__ == "__main__":
    columns = "Government"
    set_sns_theme()
    suppres_scientific_notation()
    df = load_data(path = "data/expenditure_csv.csv", sep = ";")
    df = df.fillna(value = np.nan)
    df_grouped = df.groupby('LOCATION').mean().drop(columns = 'TIME')
    
    draw_histograms(df = df, variables = df.columns[3:], n_rows = 4, n_cols = 3)
    draw_plots(df = df, variables = df.columns[3:], n_rows = 4, n_cols = 3, group = df['LOCATION'].unique())
    draw_plots(df = df_grouped, variables = df_grouped[1:].columns, n_rows = 4, n_cols = 3, group = None)

    #H0: rozkład jest istotnie podobny do normalnego
    #H1: rozkład jest istotnie różny od normalnego
    sw = shapiro_wilk_test(data_frame = df[~np.isnan(df['Government'])], feature = columns)
    if sw:
        print('Brak podstaw do odrzucenia H0. Rozkład istotnie zblizony do normalnego.')
    else:
        print('Odrzucamy H0. Rozkład istotnie różny od normalnego.')
    plot_qqplot(data_frame = df, feature = columns)
    
    #H0: badane kraje nie róznia sie pod wzgledem wydatkow z sektora rzadowego na r&d
    #H1: badane kraje róznia sie pod wzgledem wydatkow z sektora rzadowego na r&d    
    kw = kw_test(data_frame = df, feature = columns, groups = 'LOCATION')
    if kw['p-unc'].values < 0.05:
        """
        parametric boolean
        If True (default), use the parametric ttest() function. 
        If False, use pingouin.wilcoxon() or pingouin.mwu() for paired 
        or unpaired samples, respectively.
        """
        # test post-hoc parami wilcoxona
        posthoc = pairwise_tests(dv = columns
                                 , between = "LOCATION"
                                 , data = df
                                 , parametric = False
                                 , padjust = "holm").round(3)

    df_pol = df[~np.isnan(df['Government']) & ~np.isnan(df['Business enterprise'])]
    df_pol = df_pol[df_pol['LOCATION'] == 'POL']
    sw_pol_gov = shapiro_wilk_test(data_frame = df_pol, feature = 'Government')
    sw_pol_bse = shapiro_wilk_test(data_frame = df_pol, feature = 'Business enterprise')
    plot_qqplot(data_frame = df_pol, feature = 'Government')
    plot_qqplot(data_frame = df_pol, feature = 'Business enterprise')
    sw = shapiro(df_pol['Business enterprise'])
    sw = shapiro(df_pol['Government'])
    
    #H0: Polska nie różni się pod kątem wydatkow z sektora prywatnego i rządowego
    #H1: Polska różni się pod katem wydatkow z sektora prywatnego i rzadowego
    pol_expedinture_mwu = mann_whitney_u_test(df_pol['Government'], df_pol['Business enterprise'])


    #H0: wydatki w Polsce sa takie same w sektorze rzadowym i prywatnym
    #H1: wydatki w Polsce są wieksze w sektorze rzadowym niz prywatnym
    pol_expedinture_mw = mann_whitney_u_test(df_pol['Government'], df_pol['Business enterprise'], alternative = 'greater')
