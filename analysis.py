from matplotlib import transforms
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

def estimate_posteriors(features_df):
    bow_df = features_df.iloc[:, 2:-4]
    word_counts_df = bow_df.sum(axis=0)

    # filter out words that appear fewer than 5 times
    bow_df = bow_df.loc[:, word_counts_df >= 5]

    # calculate posteriors assuming uniform prior
    bow_df = pd.concat([features_df["label"], bow_df], axis=1)
    counts_df = bow_df.groupby(["label"]).sum()
    likelihoods_df = counts_df.divide(counts_df.sum(axis=1), axis=0)
    posteriors_df = likelihoods_df.divide(likelihoods_df.sum(axis=0))

    # calculate entropies
    posteriors = posteriors_df.to_numpy()
    log_posteriors = np.where(
        posteriors > 0,
        np.log(posteriors),
        0
    )
    entropy = (-posteriors * log_posteriors).sum(axis=0)
    entropy = pd.Series(entropy)
    entropy.index = posteriors_df.columns
    entropy = entropy.sort_values(ascending=False)
    
    return posteriors_df, entropy

def wordcloud_cutoff(data, title, cutoff=50):
    weights = {data.index[i]:data[i] for i in range(cutoff)}

    wc = WordCloud(width=1000, height=1000, background_color="white", max_font_size=80)
    wc.generate_from_frequencies(weights)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

def wordcloud_threshold(data, title, threshold=0.4):
    data = data[data > threshold].sort_values(ascending=False)
    weights = {data.index[i]:data[i] for i in range(data.shape[0])}

    wc = WordCloud(width=1000, height=1000, background_color="white", max_font_size=60)
    wc.generate_from_frequencies(weights)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()

# copied this function from matplotlib documentation website
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_image_features(features_df, title, companies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    lines = []
    for c in companies:
        x = features_df.loc[(features_df["label"] == c), "edge_detection_mean"]
        y = features_df.loc[(features_df["label"] == c), "clusters_var"]

        ax1.scatter(x, y)

        l = ax2.scatter(x.mean(), y.mean())
        color = l.get_facecolor()
        lines.append(l)
        confidence_ellipse(x, y, ax2, n_std=1, edgecolor=color)

    ax1.legend(companies)
    ax1.set_xlabel("edge detection score")
    ax1.set_ylabel("color clusters variance")

    ax2.legend(lines, companies)
    ax2.set_xlabel("edge detection score")
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    fig.suptitle(title)
