import numpy as np
import pandas as pd
#import seaborn as sns
from scipy.stats import pearsonr, spearmanr


data = pd.read_csv("predictions_and_column.csv")
data['y_pred'] = 1 - data['y_pred']
corr_df = pd.Series({
    'Pearson': data.corr(method='pearson').iloc[1,0],
    'Spearman': data.corr(method='spearman').iloc[1,0]})

"""
H0: r = 0 (no relationship)
H1: r > 0 (positive relationship)

"""
pr = pearsonr(x = data['int_rate'], y = data['y_pred'], alternative = 'greater')
print(pr)
print(pr.pvalue)
print(pr.confidence_interval(0.9))
print(pr.confidence_interval(0.95))
print("-"*50)
sr = spearmanr(a = data['int_rate'], b = data['y_pred'], alternative='greater')
print(sr)
print(sr.statistic)
print(f"{sr.pvalue:.20f}")
