import matplotlib.pyplot as plt
import numpy as np

# metrics = ("Accuracy", "Precision", "Recall", "F1-Score")
# penguin_means = {
#     'RF': (0.99, 0.98, 1.00, 0.99),
#     'SVC': (0.86, 0.79, 1.00, 0.88),
#     'AdaBoost': (0.96, 0.95, 0.98, 0.96),
#     'ET': (1.00, 0.99, 1.00, 1.00),
#     'QBoost': (0.96, 0.94, 1.00, 0.97)
# }

metrics = ("Av. Training Time(s)", "Av. Prediction Time(s)")
penguin_means = {
    'SVC': (434.51, 37.72),
    'RF': (92.44, 1.32),
    'QBoost': (24.62, 0.068),
    'AdaBoost':	(10.14, 0.25),
    'ET': (8.37, 0.50)
}


x = np.arange(len(metrics))  # the label locations
width = 0.15  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=5, rotation=90)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('')
ax.set_xlabel('Time Performance', weight='bold', fontsize=10)
# ax.set_title('Metric based comparison of Classifiers')
ax.set_title('Time Performance based comparison of Classifiers')
ax.set_xticks(x + width, metrics)
ax.legend(loc='upper left', ncols=5)
ax.set_ylim(0, 560)

plt.show()
