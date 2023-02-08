# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 13:45:05 2023

@author: krzys
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scikit_posthocs import posthoc_dunn
from pingouin import kruskal, qqplot, pairwise_tests
import seaborn as sns


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
        return True
    else:
        return False


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


if __name__ == "__main__":
    columns = "Government"
    set_sns_theme()
    suppres_scientific_notation()
    df = load_data(path = "data/expenditure_csv.csv", sep = ";")
    df = df.fillna(value = np.nan)
    sw = shapiro_wilk_test(data_frame = df[~np.isnan(df['Government'])], feature = columns)
    if sw:
        print('Brak podstaw do odrzucenia H0. Rozkład istotnie zblizony do normalnego.')
    else:
        print('Odrzucamy H0. Rozkład istotnie różny od normalnego.')
        plot_qqplot(data_frame = df, feature = columns)
        kw = kw_test(data_frame = df, feature = columns, groups = 'LOCATION')
        if kw['p-unc'].values < 0.05:
            """
            parametric boolean
            If True (default), use the parametric ttest() function. 
            If False, use pingouin.wilcoxon() or pingouin.mwu() for paired 
            or unpaired samples, respectively.
            """
    
            #dunn = kw_post_hoc(data_frame = df, feature = columns, groups = 'LOCATION')
            posthoc = pairwise_tests(dv = columns
                                     , between = "LOCATION"
                                     , data = df
                                     , parametric = False
                                     , padjust = "holm").round(3)
            significant = posthoc[posthoc["p-corr"] < 0.05]
            insignificant = posthoc[posthoc["p-corr"] >= 0.05]