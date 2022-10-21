import numpy as np
from matplotlib import colors
import matplotlib.cm as mplcm
from sklearn import manifold
import matplotlib.pyplot as plt


#  embedding可视化， 降维至二维平面
def scatter_clustering(x, y_pred, n_clusters, name, n_components=2):
    t = manifold.TSNE(n_components=n_components, init='pca', random_state=501)
    x_t = t.fit_transform(x)
    fig, axi1 = plt.subplots(1)

    # 设置不同颜色
    cm = plt.get_cmap('gist_rainbow')
    cNorm = colors.Normalize(vmin=0, vmax=n_clusters - 1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    axi1.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(n_clusters)])
    for i in range(n_clusters):
        axi1.scatter(x_t[y_pred == i, 0], x_t[y_pred == i, 1], marker='o', s=8)

    plt.title(name)
    f = plt.gcf()
    f.savefig('fig/tsne/tsne_{}.png'.format(name))
    plt.show()


def embedding_visualization():
    with open('./probe/labels_Python20.txt', 'r') as file:
        content = file.readlines()
        label = [int(line.strip()) for line in content]
        label = np.array(label)

    with open('./data/embeddings/12/test/codebert_original_Python20.txt', 'r') as file:
        content = file.readlines()
        embedding = []
        for line in content:
            arr = line.split()
            arr = [float(num) for num in arr]
            embedding.append(arr)
        embedding = np.array(embedding)

    name = "Python800-CodeBERT-Pretrained"
    scatter_clustering(embedding, label, 20, name)


# dataset = ['POJ104', 'Java250', 'Python800']
# models = ['CodeBERT-Pretrained', "CodeBERT-Finetuned", 'CodeBERTa-Pretrained', "CodeBERTa-Finetuned"]



embedding_visualization()






