import pandas as pd
import numpy as  np
import plot_likert
import matplotlib.pyplot as plt

# KNN WITHOUT EXPLANATIONS
KNN_without = pd.DataFrame(
    {'Strongly disagree': {'Q1': 0.0, 'Q2': 1.0, 'Q3': 2.0},
     'Disagree': {'Q1': 4.0, 'Q2': 3.0, 'Q3': 4.0},
     'Neither agree nor disagree': {'Q1': 3.0, 'Q2': 3.0, 'Q3': 1.0},
     'Agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0},
     'Strongly agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0}}
)
print(KNN_without)

plot_likert.plot_counts(KNN_without, plot_likert.scales.agree)

# .....................................................................
# KNN WITH EXPLANATIONS
KNN_with = pd.DataFrame(
    {'Strongly disagree': {'Q1': 1.0, 'Q2': 1.0, 'Q3': 2.0},
     'Disagree': {'Q1': 4.0, 'Q2': 3.0, 'Q3': 2.0},
     'Neither agree nor disagree': {'Q1': 2.0, 'Q2': 3.0, 'Q3': 3.0},
     'Agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0},
     'Strongly agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0}}
)
print(KNN_with)

plot_likert.plot_counts(KNN_with, plot_likert.scales.agree)

# .....................................................................
# KNN+LIME RESULTS
KNN_results = pd.DataFrame(
    {'Strongly disagree': {'Q1': 2.0},
     'Disagree': {'Q1': 4.0},
     'Neither agree nor disagree': {'Q1': 1.0},
     'Agree': {'Q1': 0.0},
     'Strongly agree': {'Q1': 0.0}}
)
print(KNN_results)

plot_likert.plot_counts(KNN_results, plot_likert.scales.agree)

plt.show()