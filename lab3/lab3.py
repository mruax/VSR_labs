import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.cluster import hierarchy
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


# ==================== ГЕНЕРАЦИЯ ДАННЫХ ====================
def generate_xps_data():
    """Генерация примерных данных РФЭС"""
    np.random.seed(42)
    samples = []

    modifier_types = ['Без модификатора', 'C-модификатор', 'N-модификатор', 'S-модификатор']
    electron_doses = [0, 1e15, 5e15, 1e16, 5e16]

    for modifier in modifier_types:
        for dose in electron_doses:
            base_noise = np.random.normal(0, 0.5)
            dose_effect = dose / 1e16

            sample = {
                'Модификатор': modifier,
                'Доза_e_см2': dose,
                'Pb_4f7': 138.2 + base_noise + dose_effect * 0.3,
                'I_3d5': 619.0 + base_noise - dose_effect * 0.4,
                'Cs_3d5': 724.8 + base_noise + dose_effect * 0.2,
                'N_1s': 400.2 + base_noise if 'N' in modifier else 399.8 + base_noise,
                'C_1s': 284.8 + base_noise + (1.0 if 'C' in modifier else 0),
                'O_1s': 531.5 + base_noise + dose_effect * 0.5,
                'Pb_I_ratio': 1.5 - dose_effect * 0.3 + np.random.normal(0, 0.1),
                'Organic_%': (15 if modifier != 'Без модификатора' else 5) - dose_effect * 3,
                'Degrad_index': dose_effect * 10 + np.random.normal(0, 1),
                'Roughness_nm': 2.5 + dose_effect * 5 + np.random.normal(0, 0.5),
                'Band_gap_eV': 1.6 - dose_effect * 0.15 + np.random.normal(0, 0.02),
            }
            samples.append(sample)

    return pd.DataFrame(samples)


# ==================== ОСНОВНОЙ АНАЛИЗ ====================
print("=" * 80)
print("АНАЛИЗ ДАННЫХ РФЭС ПЕРОВСКИТОВ")
print("=" * 80)

df = generate_xps_data()
print(f"\n✓ Загружено {df.shape[0]} образцов, {df.shape[1]} параметров")

# Сохранение таблицы
df.to_csv('outputs/xps_data.csv', index=False)
print("✓ Таблица: xps_data.csv")

# ==================== ВИЗУАЛИЗАЦИЯ 1: Распределения ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Распределение энергий связи элементов (РФЭС)', fontsize=16, fontweight='bold')

elements = ['Pb_4f7', 'I_3d5', 'Cs_3d5', 'N_1s', 'C_1s', 'O_1s']
for ax, element in zip(axes.flat, elements):
    for modifier in df['Модификатор'].unique():
        data = df[df['Модификатор'] == modifier][element]
        ax.hist(data, alpha=0.5, label=modifier, bins=8)
    ax.set_xlabel('Энергия связи (эВ)', fontsize=10)
    ax.set_ylabel('Частота', fontsize=10)
    ax.set_title(element, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fig1_distributions.png', dpi=300, bbox_inches='tight')
print("✓ График 1: fig1_distributions.png")
plt.close()

# ==================== ВИЗУАЛИЗАЦИЯ 2: Влияние дозы ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Влияние дозы электронов на свойства перовскитов', fontsize=16, fontweight='bold')

properties = [
    ('Pb_I_ratio', 'Отношение Pb/I'),
    ('Organic_%', 'Органика (%)'),
    ('Degrad_index', 'Индекс деградации'),
    ('Band_gap_eV', 'Ширина з.з. (эВ)')
]

for ax, (prop, label) in zip(axes.flat, properties):
    for modifier in df['Модификатор'].unique():
        subset = df[df['Модификатор'] == modifier]
        ax.plot(subset['Доза_e_см2'], subset[prop], marker='o', label=modifier, linewidth=2, markersize=6)
    ax.set_xlabel('Доза электронов (см⁻²)', fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

plt.tight_layout()
plt.savefig('outputs/fig2_dose_effects.png', dpi=300, bbox_inches='tight')
print("✓ График 2: fig2_dose_effects.png")
plt.close()

# ==================== КОРРЕЛЯЦИЯ ПИРСОНА ====================
print("\n" + "=" * 80)
print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 80)

corr_data = df.select_dtypes(include=[np.number])
pearson_corr = corr_data.corr(method='pearson')

# Сохранение матрицы
pearson_corr.to_csv('outputs/pearson_correlation.csv')
print("\n✓ Матрица Пирсона: pearson_correlation.csv")

# Визуализация
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Матрица корреляции Пирсона', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/fig3_pearson.png', dpi=300, bbox_inches='tight')
print("✓ График 3: fig3_pearson.png")
plt.close()


# ==================== P-VALUES ====================
def calculate_pvalues(df):
    cols = df.columns
    p_matrix = np.zeros((len(cols), len(cols)))
    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):
            if i != j:
                _, p_val = pearsonr(df[col1].dropna(), df[col2].dropna())
                p_matrix[i, j] = p_val
    return pd.DataFrame(p_matrix, columns=cols, index=cols)


pvalues = calculate_pvalues(corr_data)
pvalues.to_csv('outputs/pvalues.csv')
print("✓ P-values: pvalues.csv")

# Визуализация
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(pvalues, dtype=bool), k=1)
sns.heatmap(pvalues, mask=mask, annot=True, fmt='.3f', cmap='RdYlGn_r',
            square=True, linewidths=1, ax=ax, vmin=0, vmax=0.05)
ax.set_title('Матрица значимости (p-values)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('outputs/fig4_pvalues.png', dpi=300, bbox_inches='tight')
print("✓ График 4: fig4_pvalues.png")
plt.close()

# ==================== КЛАСТЕРНЫЙ АНАЛИЗ ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

linkage_vars = hierarchy.linkage(pearson_corr, method='ward')
dendro = hierarchy.dendrogram(linkage_vars, labels=pearson_corr.columns,
                              ax=ax1, orientation='right', leaf_font_size=10)
ax1.set_title('Дендрограмма параметров', fontsize=14, fontweight='bold')
ax1.set_xlabel('Расстояние', fontsize=12)

idx = dendro['leaves']
corr_ordered = pearson_corr.iloc[idx, idx]
sns.heatmap(corr_ordered, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax2, vmin=-1, vmax=1)
ax2.set_title('Упорядоченная матрица корреляций', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig5_cluster.png', dpi=300, bbox_inches='tight')
print("✓ График 5: fig5_cluster.png")
plt.close()

# ==================== PhiK КОРРЕЛЯЦИЯ ====================
try:
    from phik import phik_matrix, significance_matrix

    phik_corr = phik_matrix(df)
    phik_corr.to_csv('outputs/phik_correlation.csv')
    print("✓ PhiK матрица: phik_correlation.csv")

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(phik_corr, dtype=bool), k=1)
    sns.heatmap(phik_corr, mask=mask, annot=True, fmt='.2f', cmap='plasma',
                square=True, linewidths=1, ax=ax, vmin=0, vmax=1, annot_kws={'size': 8})
    ax.set_title('Матрица PhiK корреляции', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('outputs/fig6_phik.png', dpi=300, bbox_inches='tight')
    print("✓ График 6: fig6_phik.png")
    plt.close()

    # Сравнение Pearson vs PhiK
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    mask = np.triu(np.ones_like(pearson_corr, dtype=bool), k=1)
    sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, ax=axes[0], vmin=-1, vmax=1, annot_kws={'size': 7})
    axes[0].set_title('Pearson (линейные)', fontsize=14, fontweight='bold')

    common_cols = list(set(pearson_corr.columns) & set(phik_corr.columns))
    phik_aligned = phik_corr.loc[common_cols, common_cols]
    mask_phik = np.triu(np.ones_like(phik_aligned, dtype=bool), k=1)
    sns.heatmap(phik_aligned, mask=mask_phik, annot=True, fmt='.2f', cmap='viridis',
                square=True, linewidths=1, ax=axes[1], vmin=0, vmax=1, annot_kws={'size': 7})
    axes[1].set_title('PhiK (линейные + нелинейные)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/fig7_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ График 7: fig7_comparison.png")
    plt.close()

except ImportError:
    print("\n⚠ PhiK не установлен. Установите: pip install phik")

# ==================== SCATTER PLOTS ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Диаграммы рассеяния ключевых зависимостей', fontsize=16, fontweight='bold')

scatter_pairs = [
    ('Доза_e_см2', 'Degrad_index'),
    ('Pb_I_ratio', 'Degrad_index'),
    ('Organic_%', 'Band_gap_eV'),
    ('Доза_e_см2', 'Pb_4f7')
]

for ax, (x_var, y_var) in zip(axes.flat, scatter_pairs):
    for modifier in df['Модификатор'].unique():
        subset = df[df['Модификатор'] == modifier]
        ax.scatter(subset[x_var], subset[y_var], label=modifier, alpha=0.7, s=80,
                   edgecolors='black', linewidth=0.5)

    z = np.polyfit(df[x_var], df[y_var], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df[x_var].min(), df[x_var].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, label='Тренд')

    r, p_val = pearsonr(df[x_var], df[y_var])
    ax.set_xlabel(x_var, fontsize=11)
    ax.set_ylabel(y_var, fontsize=11)
    ax.set_title(f'{x_var} vs {y_var}\nr={r:.3f}, p={p_val:.4f}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/fig8_scatter.png', dpi=300, bbox_inches='tight')
print("✓ График 8: fig8_scatter.png")
plt.close()

# ==================== СИЛЬНЫЕ КОРРЕЛЯЦИИ ====================
print("\n" + "=" * 80)
print("СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7)")
print("=" * 80)

strong_corr = []
for i in range(len(pearson_corr.columns)):
    for j in range(i + 1, len(pearson_corr.columns)):
        corr_val = pearson_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            strong_corr.append({
                'Переменная 1': pearson_corr.columns[i],
                'Переменная 2': pearson_corr.columns[j],
                'r': corr_val,
                'p-value': pvalues.iloc[i, j]
            })

if strong_corr:
    strong_df = pd.DataFrame(strong_corr).sort_values('r', key=abs, ascending=False)
    print(strong_df.head(15).to_string(index=False))
    strong_df.to_csv('outputs/strong_correlations.csv', index=False)
    print("\n✓ Таблица: strong_correlations.csv")

# ==================== ИТОГ ====================
print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)
print("\n📁 Созданные файлы:")
print("   • xps_data.csv - исходные данные")
print("   • pearson_correlation.csv - матрица Пирсона")
print("   • pvalues.csv - статистическая значимость")
print("   • phik_correlation.csv - матрица PhiK")
print("   • strong_correlations.csv - сильные корреляции")
print("\n📊 Графики:")
print("   • fig1_distributions.png - распределения энергий")
print("   • fig2_dose_effects.png - влияние дозы")
print("   • fig3_pearson.png - корреляция Пирсона")
print("   • fig4_pvalues.png - значимость")
print("   • fig5_cluster.png - кластерный анализ")
print("   • fig6_phik.png - PhiK корреляция")
print("   • fig7_comparison.png - сравнение методов")
print("   • fig8_scatter.png - диаграммы рассеяния")
print("\n" + "=" * 80)
