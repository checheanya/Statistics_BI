import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.weightstats import ztest
import scipy.stats as st


def demonstrate_clt(expressions):
    sample_size = len(expressions) // 5  # changable
    n_samples = 1000  # changable
    means_list = []
    for i in range(n_samples):
        sample = np.random.choice(expressions, sample_size)
        means_list.append(np.mean(sample))

    return means_list


def hist(data, gene, id):
    plt.figure()
    graph_histogram = sns.histplot(data[gene], stat="density")
    graph_histogram = graph_histogram.get_figure()
    graph_histogram.savefig(f"{gene}_type{id}_histogram.png")


def boxplot(data1, data2, gene):
    means1_gene = demonstrate_clt(data1[gene])
    means2_gene = demonstrate_clt(data2[gene])
    means_df = pd.DataFrame(data={"Means first type": means1_gene, "Means second type": means2_gene}, dtype=np.float64)
    graph_boxplot = sns.boxplot(data=means_df)
    fig3 = graph_boxplot.get_figure()
    fig3.savefig(f"{gene}_boxplot_means.png")


def plots(de_first, de_second, gene):
    # histograms for the gene expression levels distribution
    hist(de_first, gene, id=1)
    hist(de_second, gene, id=2)

    # boxplot for the means comparing two given datasets
    boxplot(de_first, de_second, gene)


def stat_ci(data, gene, alpha=0.95):
    # NK клетки
    ci = st.t.interval(alpha,
                       df=len(data[gene]) - 1,
                       loc=np.mean(data[gene]),
                       scale=st.sem(data[gene]))
    return ci


def check_intervals_intersect(first_ci, second_ci):
    first_ci = sorted(first_ci)
    second_ci = sorted(second_ci)
    if (second_ci[0] <= first_ci[1] <= second_ci[1]) or (first_ci[0] <= second_ci[1] <= first_ci[1]):
        return True
    else:
        return False


def check_dge_with_ci(first_table, second_table):
    # dge - differential gene expression
    common_genes = set(first_table.columns).intersection(set(second_table))
    ci_test_results = dict()
    for gene in common_genes:
        # if intervals intersect --> difference is not significant
        if check_intervals_intersect(stat_ci(first_table, gene), stat_ci(second_table, gene)):
            ci_test_results[gene] = False
        else:
            ci_test_results[gene] = True
    return ci_test_results


def check_dge_with_ztest(first_table, second_table):
    # dge - differential gene expression
    alpha = 0.05
    common_genes = set(first_table.columns).intersection(set(second_table))
    z_test_results = []
    p_values = []
    for gene in common_genes:
        z_stat, p_val = ztest(first_table[gene], second_table[gene])
        p_values.append(p_val)
        z_test_results.append(p_val < alpha)
    return z_test_results, p_values


if __name__ == '__main__':
    first_cell_type_expressions_path = input("Enter the path to the type-1 data: ")
    second_cell_type_expressions_path = input("Enter the path to the type-2 data: ")
    save_results_table = input("How to name the output file? ")
    # downloading the data
    first_expr = pd.read_csv(first_cell_type_expressions_path, index_col=0)
    second_expr = pd.read_csv(second_cell_type_expressions_path, index_col=0)

    # filling na with means if there are any
    if first_expr.isnull().values.any():
        first_expr.fillna(first_expr, inplace=True)
    if second_expr.isnull().values.any():
        second_expr.fillna(second_expr, inplace=True)

    # plotting
    to_plot_bool = input("Do you want to view a plot histograms and a plot for means of some genes? (y/n) ")
    if to_plot_bool == "y":
        genes = input("Enter names of the genes (a, b, c): ").split(", ")
        for gene in genes:
            plots(first_expr, second_expr, gene)
        print("Check the current folder for the plots!")

    # confidencse intervals check
    ci_test_results = check_dge_with_ci(first_expr, second_expr)

    # z-test check
    z_test_results, z_test_p_values = check_dge_with_ztest(first_expr, second_expr)

    # difference of means
    mean_diff = first_expr.mean(axis=0) - second_expr.mean(axis=0)

    results = {
        "ci_test_results": ci_test_results,
        "z_test_results": z_test_results,
        "z_test_p_values": z_test_p_values,
        "mean_diff": mean_diff}

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{save_results_table}.csv")
    print(f"Thank you for the patience, you can find your results in the {save_results_table}.csv file!")

