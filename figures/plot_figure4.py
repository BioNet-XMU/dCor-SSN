"""Plot manuscript Figure 4 from the two primary, aggregate Section C tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

methods = ("dcor", "pearson", "spearman")
names = ("dCor-SSN", "Pearson-SSN", "Spearman-SSN")
colors = ("#BF5200", "#0674A8", "#008061")
markers = ("o", "s", "^")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.2, "axes.labelsize": 9.3, "axes.titlesize": 9.8,
    "axes.linewidth": .7, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42, "savefig.facecolor": "white",
})


def grid(ax, axis):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E7E9EC", linewidth=.6)


def plot(tables, out):
    order = ("ST000355","ST003390","GC")
    cohort_names = ("Breast cancer","Type 2 diabetes","Gastric cancer")
    cohort = pd.read_csv(tables / "primary_q10.csv").set_index("cohort").loc[list(order)]
    recurrence = pd.read_csv(tables / "high_recurrence_partition.csv").query("q == 0.1 and category == 'All three'").set_index("cohort").loc[list(order)]
    assert np.allclose(recurrence["count"]/recurrence["union_count"]*100,recurrence["percent_of_union"])
    assert cohort["n_sample_pairs"].tolist()==[22791,44850,135460]
    fig=plt.figure(figsize=(7.3,2.98))
    a=fig.add_axes([.229,.249,.412,.589])
    b=fig.add_axes([.725,.249,.235,.589])
    a.text(-.48,1.08,"A",transform=a.transAxes,fontsize=11.5,weight="bold",va="bottom")
    b.text(-.15,1.08,"B",transform=b.transAxes,fontsize=11.5,weight="bold",va="bottom")
    a.set_title("Between-sample edge overlap",pad=12)
    b.set_title("Recurrent edges\nshared by all methods",pad=8)
    for i,(method,color,marker) in enumerate(zip(methods,colors,markers)):
        a.scatter(cohort[method],np.arange(3)+(i-1)*.19, color=color,marker=marker,s=37,
                  edgecolor="white",linewidth=.5,zorder=3)
    for y,value in enumerate(recurrence["percent_of_union"]):
        b.barh(y,value,height=.44,color="#566978")
        b.text(value+1.3,y,f"{value:.1f}%",va="center",fontsize=8.9)
    a.set(xlim=(.50,.92),xticks=(.5,.6,.7,.8,.9),xlabel="Mean Jaccard similarity",
          yticks=range(3),yticklabels=cohort_names)
    b.set(xlim=(0,65),xticks=(0,20,40,60),xlabel="Share of union (%)",yticks=[])
    for ax in (a,b):
        ax.set_ylim(2.50,-.50)
        ax.tick_params(axis="y",length=0)
        grid(ax,"x")
    for y in (.5,1.5):
        a.axhline(y,color="#E7E9EC",linewidth=.54,zorder=0)
        b.axhline(y,color="#E7E9EC",linewidth=.54,zorder=0)
    fig.legend([Line2D([],[],linestyle="none",marker=m,color=co,markersize=5.6) for m,co in zip(markers,colors)],
               names,frameon=False,ncol=3,loc="center",bbox_to_anchor=(.502,.071),columnspacing=1.6)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "Figure4_three_cohort_concordance.pdf")
    fig.savefig(out / "Figure4_three_cohort_concordance.png", dpi=420)
    plt.close(fig)



if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=root / "outputs/section_c/tables")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/section_c/figures")
    args = parser.parse_args()
    plot(args.tables, args.output_dir)
