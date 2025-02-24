import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Step 2: Prepare your data
data = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

# Step 3: Apply t-SNE with perplexity less than number of samples
tsne = TSNE(n_components=2, perplexity=2, random_state=42)
tsne_results = tsne.fit_transform(data)

# Step 4: Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c='blue', marker='o')
for i, txt in enumerate(['[0, 0, 0]', '[0, 1, 1]', '[1, 0, 1]', '[1, 1, 1]']):
    plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))
plt.title('t-SNE visualization of the data')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.show()
