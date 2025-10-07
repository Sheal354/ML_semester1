import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Загрузка данных

df = pd.read_csv('sponge.data', header=None, sep=',')

species_names = df.iloc[:, 0].tolist()
X_raw = df.iloc[:, 1:]
X_raw.columns = [f'attr_{i}' for i in range(1, 46)]

# Обработка пропусков
X_raw = X_raw.replace('?', np.nan)

# 2. Истинные метки

class_mapping = {
    'TYLEXOCLADUS_JOUBINI': 0, 'TRACHYTELEIA_STEPHENSI': 0, 'TENTORIUM_SEMISUBERITES': 0,
    'TENTORIUM_PAPILLATUS': 0, 'SUBERITES_CAMINATUS': 0, 'SPINULARIA_SPINULARIA': 0, 'AAPTOS_AAPTOS': 0,
    'SPIRASTRELLA_MINAX': 1, 'SPIRASTRELLA_CUNCTATRIX': 1, 'LAXOSUBERITES_ECTYONIMUS': 1,
    'DIPLASTRELLA_ORNATA': 1, 'DIPLASTRELLA_BISTELLATA': 1, 'CLIONA_SCHMIDTI': 1,
    'CLIONA_LABYRINTHICA': 1, 'CLIONA_CELATA': 1, 'CLIONA_CARTERI': 1, 'ALECTONA_MILLARI': 1,
    'CLIONA_VIRIDIS': 2,
    'TERPIOS_FUGAX': 3, 'SUBERITES_GIBBOSICEPS': 3, 'SUBERITES_CARNOSUS_V.TYPICUS': 3,
    'SUBERITES_CARNOSUS_V.RAMOSUS': 3, 'SUBERITES_CARNOSUS_V.INCRUSTANS': 3, 'RHIZAXINELLA_UNISETA': 3,
    'RHIZAXINELLA_PYRIFERA': 3, 'RHIZAXINELLA_ELONGATA': 3, 'RHIZAXINELLA_BISETA': 3,
    'PSEUDOSUBERITES_SULFUREUS': 3, 'PSEUDOSUBERITES_HYALINUS': 3, 'PROSUBERITES_RUGOSUS': 3,
    'PROSUBERITES_LONGISPINA': 3, 'PROSUBERITES_EPIPHYTUM': 3, 'LAXOSUBERITES_RUGOSUS': 3,
    'LAXOSUBERITES_FERRERHERNANDEZI': 3,
    'STYLOCORDYLA_BOREALIS': 4, 'OXYCORDYLA_PELLITA': 4,
    'SPHAEROTYLUS_CAPITATUS': 5, 'PROTELEIA_SOLLASI': 5, 'POLYMASTIA_UBERRIMA': 5,
    'POLYMASTIA_TENAX': 5, 'POLYMASTIA_SPINULA': 5, 'POLYMASTIA_ROBUSTA': 5,
    'POLYMASTIA_RADIOSA': 5, 'POLYMASTIA_POLYTYLOTA': 5, 'POLYMASTIA_MAMMILLARIS': 5,
    'POLYMASTIA_LITTORALIS': 5, 'POLYMASTIA_INVAGINATA': 5, 'POLYMASTIA_INFLATA': 5,
    'POLYMASTIA_HIRSUTA': 5, 'POLYMASTIA_CORTICATA': 5, 'POLYMASTIA_CONIGERA': 5,
    'POLYMASTIA_AGGLUTINARIS': 5,
    'SPHAEROTYLUS_ANTARCTICUS': 6, 'POLYMASTIA_TISSIERI': 6, 'POLYMASTIA_MARTAE': 6,
    'POLYMASTIA_INFRAPILOSA': 6, 'POLYMASTIA_GRIMALDI': 6, 'POLYMASTIA_FUSCA': 6,
    'POLYMASTIA_ECTOFIBROSA': 6,
    'WEBERELLA_VERRUCOSA': 7, 'WEBERELLA_BURSA': 7, 'RIDLEYA_OVIFORMIS': 7,
    'QUASILINA_RICHARDII': 7, 'QUASILINA_INTERMEDIA': 7, 'QUASILINA_BREVIS': 7,
    'SUBERITES_FICUS': 8, 'SUBERITES_DOMUNCULA': 8,
    'TETHYA_CITRINA': 9, 'TETHYA_AURANTIUM': 9,
    'TIMEA_UNISTELLATA': 10, 'TIMEA_STELLATA': 10, 'TIMEA_MIXTA': 10,
    'TIMEA_HALLEZI': 10, 'TIMEA_CHONDRILLOIDES': 10,
    'TRICHOSTEMA_SARSI': 11, 'TRICHOSTEMA_HEMISPHAERICUM': 11
}

true_labels = [class_mapping[sp] for sp in species_names]

# 3. Обработка признаков

numeric_features = ['attr_7', 'attr_28', 'attr_37']
for col in numeric_features:
    X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')

mode_color = X_raw['attr_39'].mode()[0]
X_raw['attr_39'] = X_raw['attr_39'].fillna(mode_color)

binary_features = [
    'attr_3', 'attr_8', 'attr_11', 'attr_12', 'attr_13', 'attr_14',
    'attr_15', 'attr_16', 'attr_17', 'attr_29', 'attr_31', 'attr_38',
    'attr_42', 'attr_43', 'attr_44'
]

for col in binary_features:
    X_raw[col] = X_raw[col].map({'NO': 0, 'SI': 1})

nominal_features = [col for col in X_raw.columns if col not in numeric_features and col not in binary_features]

# 4. OneHot + масштабирование

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('nom', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ]
)

X = preprocessor.fit_transform(X_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Кластеризация

algorithms = {
    'KMeans': KMeans(n_clusters=12, random_state=42, n_init=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=12),
    'Spectral': SpectralClustering(n_clusters=12, random_state=42, affinity='nearest_neighbors', n_neighbors=10)
}

results = {}
for name, algo in algorithms.items():
    if name == 'Spectral':
        labels = algo.fit_predict(X_scaled.astype(np.float64))
    else:
        labels = algo.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X_scaled, labels) if n_clusters >= 2 else -1
    ari = adjusted_rand_score(true_labels, labels) if n_clusters >= 2 else np.nan
    nmi = normalized_mutual_info_score(true_labels, labels) if n_clusters >= 2 else np.nan

    results[name] = {
        'n_clusters': n_clusters,
        'silhouette': sil,
        'ARI': ari,
        'NMI': nmi,
        'labels': labels
    }

# 6. Выравнивание меток

def align_labels(true_labels, pred_labels):
    unique_pred = np.unique(pred_labels)
    unique_true = np.unique(true_labels)
    n_clusters = len(unique_pred)
    n_classes = len(unique_true)

    contingency_matrix = np.zeros((n_clusters, n_classes))
    for i in range(len(true_labels)):
        contingency_matrix[pred_labels[i], true_labels[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    mapping = {old: new for old, new in zip(row_ind, col_ind)}
    aligned_labels = np.array([mapping.get(label, label) for label in pred_labels])

    return aligned_labels


# Применяем выравнивание
aligned_results = {'Истинные': true_labels}
for name in results.keys():
    aligned_results[name] = align_labels(true_labels, results[name]['labels'])

# 7. Вывод результатов

print("\nРезультаты кластеризации:")
print("-" * 60)
for name, res in results.items():
    print(f"{name}:")
    print(f"  Кластеров: {res['n_clusters']}")
    print(f"  Silhouette: {res['silhouette']:.3f}")
    if not np.isnan(res['ARI']):
        print(f"  ARI: {res['ARI']:.3f}")
        print(f"  NMI: {res['NMI']:.3f}")
    print()

# 8. Визуализация

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_tsne = tsne.fit_transform(X_scaled)

plot_labels = {
    'Истинные': aligned_results['Истинные'],
    'KMeans': aligned_results['KMeans'],
    'Agglomerative': aligned_results['Agglomerative'],
    'Spectral': aligned_results['Spectral']
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (title, labels) in enumerate(plot_labels.items()):
    ax = axes[i]
    ax.set_title(title, fontsize=12)
    ax.axis('off')

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for j, label in enumerate(unique_labels):
        mask = labels == label
        points = X_tsne[mask]
        ax.scatter(points[:, 0], points[:, 1], c=[colors[j]], s=80, edgecolors='k', linewidth=0.5, alpha=0.8)

        # Выпуклая оболочка
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=0.5, alpha=0.6)
            except:
                pass

plt.tight_layout()
plt.show()