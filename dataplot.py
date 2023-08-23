# importing package
import matplotlib.pyplot as plt
import pandas as pd

# create data
df = pd.DataFrame([
    ['Accuracy', 0.99, 0.86, 0.96, 1.00, 0.96],
    ['Precision', 0.98, 0.79, 0.95, 0.99, 0.94],
    ['Recall', 1.00, 1.00, 0.98, 1.00, 1.00],
    ['F1-Score', 0.99, 0.88, 0.96, 1.00, 0.97]],
    columns=['Metrics', 'RF', 'SVC', 'AdaBoost', 'ET', 'QBoost'])

# view data
print(df)

# plot grouped bar chart
df.plot(x='Metrics',
        kind='bar',
        stacked=False,
        title='Comparison of Quality Metrics for ML Algorithms')

plt.show()




