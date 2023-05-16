import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_rep_violin(ax, x, arr, label, c, alpha=0.5, edge_alpha=None, zorder=None, left=False, right=False):
    print(arr.shape, len(x))
    arr_ls = [arr[:, i] for i in range(arr.shape[1])]
    s = ax.violinplot(
        arr_ls,
        x,
        widths=3,
        showmedians=False,
        showmeans=True,
        showextrema=False,
        points=1000
    )

    # for partname in ('cbars','cmins','cmaxes', 'cmeans'):
    for partname in ('cmeans', ):
        vp = s[partname]
        vp.set_edgecolor(c)
        if edge_alpha is not None:
            vp.set_alpha(edge_alpha)
        # vp.set_linewidth(1)

    for pc in s['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(c)
        if alpha is not None:
            pc.set_alpha(alpha)

        # get the center
        m = np.mean(pc.get_paths()[0].vertices[:, 0])
        if right:
            # modify the paths to not go further left than the center
            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], m, np.inf)
        if left:
            # modify the paths to not go further left than the center
            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], -np.inf, m)

    return mpatches.Patch(
        color=s['bodies'][0].get_facecolor().flatten(),
        label=label
    )
