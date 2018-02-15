#!/usr/bin/env python
"""Produces the graphs for my notes on graphical models."""

import os

import daft

from matplotlib import rc


def make_figure_8p1():
    """Make the intro graph."""
    pgm = daft.PGM([2.5, 2.5], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("a", r"a", 1., 1.8, observed=True)
    )
    pgm.add_node(
        daft.Node("b", r"b", 0.2, 0.2, observed=True)
    )
    pgm.add_node(
        daft.Node("c", r"c", 1.8, 0.2, observed=True)
    )
    pgm.add_edge("a", "b")
    pgm.add_edge("b", "c")
    pgm.add_edge("a", "c")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p1.pdf")


def make_figure_8p2():
    """Create a graph like figure 8.2."""
    pgm = daft.PGM([3, 4], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("x1", r"$x_1$", 1.2, 2.8, observed=True)
    )
    pgm.add_node(
        daft.Node("x2", r"$x_2$", 0.3, 2.0, observed=True)
    )
    pgm.add_node(
        daft.Node("x3", r"$x_3$", 2.2, 2.0, observed=True)
    )
    pgm.add_node(
        daft.Node("x4", r"$x_4$", 0.6, 1.0, observed=True)
    )
    pgm.add_node(
        daft.Node("x5", r"$x_5$", 1.9, 1.0, observed=True)
    )
    pgm.add_node(
        daft.Node("x6", r"$x_6$", 0.4, 0.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x7", r"$x_7$", 2.3, 0.3, observed=True)
    )
    pgm.add_edge("x1", "x4")
    pgm.add_edge("x1", "x5")
    pgm.add_edge("x2", "x4")
    pgm.add_edge("x3", "x4")
    pgm.add_edge("x3", "x5")
    pgm.add_edge("x4", "x6")
    pgm.add_edge("x4", "x7")
    pgm.add_edge("x5", "x7")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p2.pdf")


def make_figure_8p5():
    """Create a graph like figure 8.5."""
    pgm = daft.PGM([3.5, 2.5], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("s2", r"$\sigma^2$", 0.2, 0.2, fixed=True)
    )
    pgm.add_node(
        daft.Node("xn", r"$x_n$", 1.5, 1.6, fixed=True)
    )
    pgm.add_node(
        daft.Node("tn", r"$t_n$", 1.5, 0.2, observed=True)
    )
    pgm.add_node(
        daft.Node("w", r"$\boldsymbol{w}$", 2.5, 0.2)
    )
    pgm.add_node(
        daft.Node("alpha", r"$\alpha$", 2.5, 1.6, fixed=True)
    )
    pgm.add_plate(
        daft.Plate([0.8, 0.0, 1.2, 1.9],
                   label=r"$N$",
                   shift=-0.1)
    )
    pgm.add_edge("s2", "tn")
    pgm.add_edge("xn", "tn")
    pgm.add_edge("w", "tn")
    pgm.add_edge("alpha", "w")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p5.pdf")


def make_figure_8p7():
    """Create a graph like figure 8.7."""
    pgm = daft.PGM([5, 4], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("s2", r"$\sigma^2$", 1.0, 0.2, fixed=True, offset=[-9., -7.])
    )
    pgm.add_node(
        daft.Node("xn", r"$x_n$", 1.0, 2.6, fixed=True)
    )
    pgm.add_node(
        daft.Node("tn", r"$t_n$", 1.0, 1.2, observed=True)
    )
    pgm.add_node(
        daft.Node("w", r"$\boldsymbol{w}$", 3.5, 1.2)
    )
    pgm.add_node(
        daft.Node("alpha", r"$\alpha$", 3.5, 2.6, fixed=True)
    )
    pgm.add_node(
        daft.Node("t_hat", r"$\hat t$", 3.5, 0.2)
    )
    pgm.add_node(
        daft.Node("x_hat", r"$\hat x$", 4.5, 0.2, fixed=True)
    )
    pgm.add_plate(
        daft.Plate([0.2, 0.8, 1.3, 2.1], label=r"$N$", shift=-0.1)
    )
    pgm.add_edge("s2", "tn")
    pgm.add_edge("xn", "tn")
    pgm.add_edge("w", "tn")
    pgm.add_edge("alpha", "w")
    pgm.add_edge("w", "t_hat")
    pgm.add_edge("s2", "t_hat")
    pgm.add_edge("x_hat", "t_hat")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p7.pdf")


def make_figure_8p9():
    """Create a pair of graphs like figure 8.9."""
    pgm = daft.PGM([3, 2], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("x11", r"$\boldsymbol{x}_1$", 0.2, 1.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x21", r"$\boldsymbol{x}_2$", 1.8, 1.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x12", r"$\boldsymbol{x}_1$", 0.2, 0.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x22", r"$\boldsymbol{x}_2$", 1.8, 0.3, observed=True)
    )
    pgm.add_edge("x11", "x21")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p9.pdf")


def make_figure_8p11():
    """Create a graph like figure 8.11."""
    pgm = daft.PGM([3, 2], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("mu1", r"$\mu_1$", 0.2, 1.3, observed=False)
    )
    pgm.add_node(
        daft.Node("mu2", r"$\mu_2$", 1.8, 1.3, observed=False)
    )
    pgm.add_node(
        daft.Node("x1", r"$\boldsymbol{x}_1$", 0.2, 0.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x2", r"$\boldsymbol{x}_2$", 1.8, 0.3, observed=True)
    )
    pgm.add_edge("x1", "x2")
    pgm.add_edge("mu1", "x1")
    pgm.add_edge("mu2", "x2")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p11.pdf")


def make_figure_8p12():
    """Create a graph like figure 8.12."""
    pgm = daft.PGM([4, 2], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("mu1", r"$\mu_1$", 0.2, 1.3, observed=False)
    )
    pgm.add_node(
        daft.Node("mu", r"$\mu$", 2.6, 1.3, observed=False)
    )
    pgm.add_node(
        daft.Node("x1", r"$\boldsymbol{x}_1$", 0.2, 0.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x2", r"$\boldsymbol{x}_2$", 1.8, 0.3, observed=True)
    )
    pgm.add_node(
        daft.Node("x3", r"$\boldsymbol{x}_2$", 3.4, 0.3, observed=True)
    )
    pgm.add_edge("x1", "x2")
    pgm.add_edge("x2", "x3")
    pgm.add_edge("mu1", "x1")
    pgm.add_edge("mu", "x2")
    pgm.add_edge("mu", "x3")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p12.pdf")


def make_figure_8p13():
    """Create a graph like figure 8.13."""
    pgm = daft.PGM([3, 2], origin=[-0.3, -0.3])
    pgm.add_node(
        daft.Node("x1", r"$x_1$", 0.2, 1.3, observed=True)
    )
    pgm.add_node(
        daft.Node("xm", r"$x_M$", 1.8, 1.3, observed=True)
    )
    pgm.add_node(
        daft.Node("y", r"$y$", 1.0, 0.3, observed=True)
    )
    pgm.add_edge(
        "x1", "xm", directed=False, ls=":"
    )
    pgm.add_edge("x1", "y")
    pgm.add_edge("xm", "y")
    pgm.render()
    pgm.figure.savefig("figures/fig_8p13.pdf")


def main():
    """Produce the pdfs."""
    rc("font", family="serif", size=12)
    rc("text", usetex=True)
    if not os.path.exists("figures"):
        os.makedirs("figures")
    make_figure_8p1()
    make_figure_8p2()
    make_figure_8p5()
    make_figure_8p7()
    make_figure_8p9()
    make_figure_8p11()
    make_figure_8p12()
    make_figure_8p13()


if __name__ == "__main__":
    main()
