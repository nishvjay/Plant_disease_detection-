import pandas as pd
import numpy as  np
import plot_likert
import matplotlib.pyplot as plt

# CNN WITHOUT EXPLANATIONS
CNN_without = pd.DataFrame(
    {'Strongly disagree': {'Q1': 0.0, 'Q2': 1.0, 'Q3': 2.0},
     'Disagree': {'Q1': 3.0, 'Q2': 2.0, 'Q3': 3.0},
     'Neither agree nor disagree': {'Q1': 4.0, 'Q2': 4.0, 'Q3': 2.0},
     'Agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0},
     'Strongly agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0}}
)
print(CNN_without)

plot_likert.plot_counts(CNN_without, plot_likert.scales.agree);

# .....................................................................
# CNN WITH EXPLANATIONS
CNN_with = pd.DataFrame(
    {'Strongly disagree': {'Q1': 1.0, 'Q2': 2.0, 'Q3': 3.0},
     'Disagree': {'Q1': 4.0, 'Q2': 3.0, 'Q3': 3.0},
     'Neither agree nor disagree': {'Q1': 2.0, 'Q2': 2.0, 'Q3': 1.0},
     'Agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0},
     'Strongly agree': {'Q1': 0.0, 'Q2': 0.0, 'Q3': 0.0}}
)
print(CNN_with)

plot_likert.plot_counts(CNN_with, plot_likert.scales.agree)

# .....................................................................
# CNN+LIME RESULTS
CNN_results = pd.DataFrame(
    {'Strongly disagree': {'Q1': 0.0},
     'Disagree': {'Q1': 4.0},
     'Neither agree nor disagree': {'Q1': 3.0},
     'Agree': {'Q1': 0.0},
     'Strongly agree': {'Q1': 0.0}}
)
print(CNN_results)

plot_likert.plot_counts(CNN_results, plot_likert.scales.agree)

plt.show()