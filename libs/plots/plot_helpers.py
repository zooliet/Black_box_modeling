
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_heatmap(data, columns):
    correlation_matrix = data[columns].corr()
    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show(block=False)

def plot_feature(data, column):
    plt.figure()
    plt.scatter(np.arange(len(data)), data[column], s=0.1, alpha=0.8)
    plt.title(f'Plot {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.grid(True)
    plt.show(block=False)

def plot_feature_distribution(data, column):
    plt.figure()
    sns.kdeplot(data[column], fill=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.grid(True)
    plt.show(block=False)

def plot_feature_to_target(data, column, target):
    plt.figure()
    plt.scatter(data[column], data[target], c='blue', s=0.2, marker='o')
    plt.title(f'Feature({column}) to Target({target})')
    plt.xlabel(column)
    plt.ylabel('target')
    plt.grid(True)
    plt.show(block=False)

def plot_feature_contribution(data, column, target, quantile=0.9, show_label=False):
    theshhold = data[target].quantile(0.9)
    data['yy'] = data[target].apply(lambda x: 1 if x > theshhold else 0)
    # data[target] = data[target] > theshhold
    # data[target] = data[target].astype(int)
    # plt.rcParams["figure.figsize"] = (12,8)
    fig,axs = plt.subplots(2,1)
    ax = sns.histplot(
        data=data, 
        x=column, 
        hue='yy', 
        multiple='stack', 
        ax=axs[0], 
        shrink=.8
    )
    if show_label:
        for bars in ax.containers:
            heights = [b.get_height() for b in bars]
            labels = [f'{h:.1f}' if h > 0.001 else '' for h in heights]
            ax.bar_label(bars, labels=labels, label_type='center')
    ax = sns.histplot(
        data=data, 
        x=column, 
        hue='yy', 
        multiple='fill', 
        ax=axs[1], 
        shrink=.8
    )
    if show_label:
        for bars in ax.containers:
            heights = [b.get_height() for b in bars]
            labels = [f'{h:.1f}' if h > 0.001 else '' for h in heights]
            ax.bar_label(bars, labels=labels, label_type='center')
    ax.set_ylabel('ratio')
    plt.tight_layout()
    plt.show(block=False)
    data.pop('yy')


def heatmap(data):
    correlation_matrix = data.corr()
    plt.figure()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.show()

def feature_graph(data, column_name):
    plt.figure()
    plt.scatter(np.arange(len(data)), data, alpha=0.8)
    plt.title(f'Distribution of {column_name} in Coordinate Space')
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.grid(True)
    plt.show()

def compare_two_column(data1, data2, data1_name, data2_name):
    plt.figure()
    plt.scatter(data1, data2, c='blue', marker='o')
    plt.title(f'Scatter Plot of {data1_name} & {data2_name}')
    plt.xlabel(f'{data1_name}')
    plt.ylabel(f'{data2_name}')
    plt.grid(True)
    plt.show()

def plot_func(df, column, target):
    count_df = df.groupby([column, target]).size().unstack(fill_value=0)
    count_df = count_df.div(count_df.sum(axis=1), axis=0)
    count_df.plot(kind='bar', stacked=True, color=['salmon', 'skyblue'], figsize=(8, 4))
    plt.xlabel(f'{column}')
    plt.ylabel('Proportion')
    plt.title(f'Proportion of Target by {column}')
    plt.legend(title='Target', loc='upper right')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for i in range(count_df.shape[0]):
        for j in range(count_df.shape[1]):
            plt.text(i, count_df.iloc[i, j] / 2 + count_df.iloc[i, :j].sum(), f"{count_df.iloc[i, j]:.2f}", ha='center', color='black')
    plt.show()



