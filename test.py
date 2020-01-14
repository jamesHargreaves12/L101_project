import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 40, 100);
y = np.linspace(1, 4, 100);

# Actually plot the exponential values
plt.plot(x, np.e**y)
ax = plt.gca()

# Set x logaritmic
ax.set_xscale('log')

# Rewrite the y labels
y_labels = np.linspace(min(y), max(y), 4)
ax.set_yticks(np.e**y_labels)
ax.set_yticklabels(y_labels)

plt.show()
