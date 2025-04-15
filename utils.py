import os
import re
import time
import gzip
import json
import pprint
import numpy as np
import pandas as pd
import pysubgroup as ps
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from mpi4py import MPI
from math import *
from IPython.display import display
from scipy.stats import gaussian_kde
from itertools import combinations, chain, product

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

from xgboost import XGBRegressor
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args, check_is_fitted, FLOAT_DTYPES

# directory for saving figures
FIG_BASEPATH = os.path.join("data", "fig")

# create directory for storing figures if it doesn't already exist
if not os.path.isdir(FIG_BASEPATH):
    FIG_BASEPATH = "fig"
    os.makedirs(FIG_BASEPATH)
    
FONTSIZE = 18
LINEWIDTH = 2
TICKWIDTH = 2
plt.rcParams.update(
    {
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "axes.linewidth": LINEWIDTH,
        "xtick.minor.width": TICKWIDTH,
        "xtick.major.width": TICKWIDTH,
        "ytick.minor.width": TICKWIDTH,
        "ytick.major.width": TICKWIDTH,
        "figure.facecolor": "w",
        "figure.dpi": 600,
    }
)


def write_jsonzip(data: dict, filepath: str):
    """Write json data to a zipped json file"""
    with gzip.open(filepath, "w") as fout:
        fout.write(json.dumps(data).encode("utf-8"))


def read_jsonzip(filepath: str) -> dict:
    with gzip.open(filepath, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))
    return data


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_pred.ndim == 1 and y_true.ndim == 2:
        y_pred = y_pred.reshape(-1, 1)
    
    # calculate MAPE only for non-zero actual values
    mask = y_true != 0
    y_true_non_zero = y_true[mask]
    y_pred_non_zero = y_pred[mask]
    
    return np.mean(np.abs((y_true_non_zero  - y_pred_non_zero) / y_true_non_zero)) * 100


def view_dataset():
    """
    Aggregate datasets from file and plot histograms of their
    target variables.
    """
    # read the configuration file
    config = pd.read_excel(os.path.join("data", "dataset_config.xlsx"), index_col="name")
    config = config.fillna("")
    
    # configure plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ii = 0
    ALPHA = 0.6
    BINS = 15
    TYPE_COLORS = [
        "dodgerblue",
        "tomato"
    ]

    ds = {}

    # loop over each row of the configuration file
    for dsn, row in config.iterrows():
        
        # read dataset
        print(dsn)
        df = pd.read_excel(os.path.join("data", "raw", dsn + ".xlsx"))
        t = row["target"]

        # save data to file
        print(dsn, df.values.shape, f"---> target: {t}", "\n")
        ds[dsn] = {"df": df.to_json(default_handler=str)}
        for k in list(config):
            ds[dsn][k] = row[k]
            
        # plot histogram of target variable
        std = df[t].std()
        median = df[t].median()
        ax[ii].axvline(x=median, lw=0.75, linestyle="dashed", c="k")
        ax[ii].axvline(x=median - std, lw=0.5, linestyle="dotted", c="dimgray")
        ax[ii].axvline(x=median + std, lw=0.5, linestyle="dotted", c="dimgray")
        ax[ii].hist(
            df[t], bins=BINS, color=TYPE_COLORS[ii], alpha=ALPHA, linewidth=0
        )
        xlabel = f"{t} ({row['units']})" if row["units"] else t
        ax[ii].set_xlabel(xlabel, fontsize=FONTSIZE)
        if ii in [0]:
            ax[ii].set_ylabel("Counts", fontsize=FONTSIZE)
        ax[ii].text(
            0.98, 
            0.96, 
            f"n={len(df)}", 
            fontsize=FONTSIZE - 2, 
            ha="right", 
            va="top", 
            transform=ax[ii].transAxes
        )

        ii += 1

    plt.tight_layout()
    fig.savefig(
        os.path.join(FIG_BASEPATH, "DatasetHistograms.png"), bbox_inches="tight"
    )
    plt.show()

    # save datsets to file
    write_jsonzip(ds, os.path.join("data", "datasets.json.gz"))


def get_rule_extraction(
    ds: dict, depth: int = 3, weight_factor: int = 1.0, bins: int = 5, result_size: int = 5
):
    """
    Extraction the rule within the dataset.
    """
    # save extract rules
    rules = {k: {} for k, _ in ds.items()}
    
    # loop over each dataset
    for di, dsn in enumerate(list(ds)):

        # get dataset
        print(f"Extracting rules {di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        # employ subgroup discovery algorithm
        target = ps.NumericTarget(t)
        searchspace = ps.create_selectors(df, ignore=[t], nbins=bins)
        task = ps.SubgroupDiscoveryTask (
            df,
            target,
            searchspace,
            result_set_size=result_size,
            depth=depth,
            qf = ps.StandardQFNumeric(weight_factor)
        )

        result = ps.BeamSearch().execute(task).to_dataframe()
        rules[dsn] = {"rules": result.to_json(default_handler=str)}
        
    return rules


def view_all_rules(rules: dict):
    """View the information of each rule"""
    for dsn in rules:
        print(dsn)
        rules_df = pd.DataFrame(json.loads(rules[dsn]['rules']))
        display(rules_df)


def convert_to_rule(input_str):
    pattern = r'(\w+): \[(\d+\.\d+):(\d+\.\d+)\['
    match = re.match(pattern, input_str)
    if match:
        feature = match.group(1)
        lower_bound = match.group(2)
        upper_bound = match.group(3)
        return f"{lower_bound} <= {feature} < {upper_bound}"
    return None


def parse_rule(rule):
    # define regular expressions for comparison operators
    operator_pattern = r'(>=|<=|==|>|<)'
    
    # match feature names
    feature_pattern = r'([^ \t\n\r><= \'"]+)'

    # find all operators in the rule
    operators = re.findall(operator_pattern, rule)
    if not operators:
        return None, []

    # split rule string by operator
    parts = re.split(operator_pattern, rule)
    parts = [part.strip() for part in parts if part.strip()]

    if len(parts) == 3:
        rule_feature = parts[0]
        operator = parts[1]
        value = parts[2]
        return None, None, rule_feature, operator, value
    else:
        value1 = parts[0]
        operator1 = parts[1]
        rule_feature = parts[2]
        operator2 = parts[3]
        value2 = parts[4]
        return value1, operator1, rule_feature, operator2, value2
        

def view_rule_extraction():
    """View all rules of a given dataset"""
    
    # import the datasets and rules
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    rules = read_jsonzip(os.path.join("data", "rules.json.gz"))
    
    # configure overall plot
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 2)
    
    # loop over each dataset
    for di, dsn in enumerate(list(ds)):
        
        # configure plot
        ax1 = fig.add_subplot(gs[di, 0])
        gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[di, 1], height_ratios=[1, 1])
        ax2_1 = fig.add_subplot(gs_b[0])
        ax2_2 = fig.add_subplot(gs_b[1])
    
        # get dataset
        print(f"{di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))

        t = ds[dsn]["target"]

        # get top rule
        rule = json.loads(rules[dsn]['rules'])['subgroup']['0']
        print("rule: ", rule)
        
        # check if there are comparison operators in the rules
        operators = ['<', '>', '<=', '>=', '==']
        if not any(op in rule for op in operators):
            rule = convert_to_rule(rule)
                
        # parse rule
        value1, operator1, rule_feature, operator2, value2 = parse_rule(rule)

        # generate dynamic Labels
        if value1 is None and operator1 is None:
            label_A = f"{rule_feature} = {value2}"
            if operator2 == '==':
                label_B = f"{rule_feature} = 1"
        else:
            label_A = f"{value1} {operator1} {rule_feature} {operator2} {value2}"
            label_B = "other"
        
        # data split according rulebased rule data splitting
        if value1 == None and operator1 == None:
            class_A = df[df[rule_feature] == eval(value2)]
            class_B = df[df[rule_feature] != eval(value2)]
        else:
            class_A = df[(df[rule_feature] >= eval(value1)) & (df[rule_feature] < eval(value2))]
            class_B = df[~((df[rule_feature] >= eval(value1)) & (df[rule_feature] < eval(value2)))]
    
        # plot scatter of target variable
        if di == 0:
            s = 150
        else:
            s = 50
        ax1.scatter(
            class_A[rule_feature], class_A[t], label=label_A, color='#CC554C', alpha=0.4, s=s, edgecolors='black'
        )
        ax1.scatter(
            class_B[rule_feature], class_B[t], label=label_B, color='#666666', alpha=0.4, s=s, edgecolors='black'
        )
        
        # nested diagram
        if di == 1:
            ax1.axvline(x=5.7847418, color='black', linestyle=(0, (3, 3)), linewidth=3)
            ax1.axvline(x=5.8648785, color='black', linestyle=(0, (3, 3)), linewidth=3)
            inset_axes = fig.add_axes([0.31, 0.205, 0.18, 0.18])
            inset_axes.scatter(class_A[rule_feature], class_A[t], color='#CC554C', alpha=0.3, s=70, edgecolors='black')
            inset_axes.scatter(class_B[rule_feature], class_B[t], color='#666666', alpha=0.3, s=70, edgecolors='black')
            inset_axes.axvline(x=5.7847418, color='black', linestyle=(0, (5, 5)), linewidth=3)
            inset_axes.axvline(x=5.8648785, color='black', linestyle=(0, (5, 5)), linewidth=3)
            inset_axes.set_xlim([5.5, 6.2])
            inset_axes.set_ylim([df[t].min(), df[t].max()])
            inset_axes.set_xticks([5.5, 5.8, 6.1])
            inset_axes.set_yticks([0, 200, 400, 600])
            font_properties = {'fontsize': 14, 'fontweight': 'bold'}
            inset_axes.set_xticklabels([5.5, 5.8, 6.1], fontdict=font_properties)
            inset_axes.set_yticklabels([0, 200, 400, 600], fontdict=font_properties)

        ax1.legend()
        ax1.set_xlabel(rule_feature)
        ax1.set_ylabel(f"{t} ({ds[dsn]['units']})")
        if di == 0:
            ax1.set_ylim(0, 8000)
        else:
            ax1.set_ylim(0, 780)
        ax1.text(
            -0.13, 
            1.05, 
            '(a)', 
            transform=ax1.transAxes, 
            fontsize=20
        )
    
        # unified container distribution range
        bins = np.linspace(df[t].min(), df[t].max(), 30)
        
        # plot histogram of target variable
        ax2_1.hist(
            class_B[t], bins=bins, color='#666666', edgecolor='black', alpha=0.8, label=label_B
        )
        x_vals = np.linspace(df[t].min(), df[t].max(), len(df[t]))
        kde_b = gaussian_kde(class_B[t])
        ax2_1.plot(
            x_vals, kde_b(x_vals) * len(class_B) * (bins[1] - bins[0]), color='#333333', linestyle='-', linewidth=3
        )
        ax2_1.set_ylabel('Counts')
        ax2_1.legend()
        ax2_1.text(-0.13, 1.10, '(b)', transform=ax2_1.transAxes, fontsize=20)
        ax2_1.set_xticks([])
        
        ax2_2.hist(
            class_A[t], bins=bins, color='#CC554C', edgecolor='black', alpha=0.8, label=label_A
        )
        kde_a = gaussian_kde(class_A[t])
        ax2_2.plot(
            x_vals, kde_a(x_vals) * len(class_A) * (bins[1] - bins[0]), color='#D83327', linestyle='-', linewidth=3
        )
        ax2_2.set_xlabel(f"{t} ({ds[dsn]['units']})")
        ax2_2.set_ylabel('Counts')
        if di == 0:
            ax2_2.set_ylim(0, 13)
        else:
            ax2_2.set_ylim(0, 35)
        ax2_2.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_BASEPATH, "RuleExtraction.png"))
    plt.show()


def within_group_tests(X, y, groups, cv, scoring, pipeline):
    group_scores = {}
    for group_value in np.unique(groups):
        group_indices = np.where(groups == group_value)[0]
        X_group = X.iloc[group_indices]
        y_group = y.iloc[group_indices]
        cv_results = cross_validate(pipeline, X_group, y_group, cv=cv, scoring=scoring)
        avg_scores = {
            metric: float(-np.mean(cv_results['test_' + metric])) 
            if metric != 'r2' 
            else float(np.mean(cv_results['test_' + metric])) 
            for metric in scoring
        }
        avg_scores = {metric: round(score, 4) for metric, score in avg_scores.items()}
        group_scores[f'Group {group_value}'] = avg_scores
    return group_scores


def cross_group_test(X, y, groups, scoring, pipeline):
    scores = {metric: [] for metric in scoring}
    g1_indices = np.where(groups)[0]
    g2_indices = np.where(~groups)[0]
    
    # subgroup1 training, subgroup2 test
    pipeline.fit(X.iloc[g1_indices], y.iloc[g1_indices])
    y_pred = pipeline.predict(X.iloc[g2_indices])
    # scores['r2'].append(r2_score(y.iloc[g2_indices], y_pred))
    scores['rmse'].append(float(rmse(y.iloc[g2_indices], y_pred)))
    # scores['mae'].append(float(mean_absolute_error(y.iloc[g2_indices], y_pred)))
    scores['mape'].append(float(mape(y.iloc[g2_indices], y_pred)))

    # subgroup2 training, subgroup1 test
    pipeline.fit(X.iloc[g2_indices], y.iloc[g2_indices])
    y_pred = pipeline.predict(X.iloc[g1_indices])
    # scores['r2'].append(r2_score(y.iloc[g1_indices], y_pred))
    scores['rmse'].append(float(rmse(y.iloc[g1_indices], y_pred)))
    # scores['mae'].append(float(mean_absolute_error(y.iloc[g1_indices], y_pred)))
    scores['mape'].append(float(mape(y.iloc[g1_indices], y_pred)))

    scores = {metric: [round(score, 4) for score in scores_list] for metric, scores_list in scores.items()}
    
    return {metric: scores[metric][0] for metric in scoring}, {metric: scores[metric][1] for metric in scoring}


def rule_validation(models):
    """
    no subgrouping, within-subgroup, and cross-subgroup
    """
    
    # import the data and rule
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    rules = read_jsonzip(os.path.join("data", "rules.json.gz"))
    split_types = ["rule", "random"]
    
    ss = {dsn: {model_name: {} for model_name in models} for dsn in ds}

    # custom scorer
    scoring = {
        # 'r2': 'r2', 
        'rmse': make_scorer(rmse, greater_is_better=False), 
        # 'mae': 'neg_mean_absolute_error', 
        'mape': make_scorer(mape, greater_is_better=False)
    }

    # k-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # loop over each dataset
    for di, dsn in enumerate(list(ds)):
        
        # get dataset
        print(f"{di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]
        X = df.drop([t], axis=1)
        y = df[t]

        # get top rule
        rule = json.loads(rules[dsn]['rules'])['subgroup']['0']
        print("rule: ", rule)

        # check if there are comparison operators in the rules
        operators = ['<', '>', '<=', '>=', '==']
        if not any(op in rule for op in operators):
            rule = convert_to_rule(rule)
                
        # parse rule
        value1, operator1, rule_feature, operator2, value2 = parse_rule(rule)

        # generate dynamic Labels
        if value1 is None and operator1 is None:
            is_rule_satisfied = df[rule_feature] == eval(value2)
        else:
            value1 = 5.7847418
            value2 = 5.8648785
            is_rule_satisfied = (df[rule_feature] >= value1) & (df[rule_feature] < value2)
        print(is_rule_satisfied.sum())
        
        for model_name, model in models.items():
            # initialization result dictionary
            ss[dsn][model_name] = {
                "overall": {},
                "split": {
                    stype: {
                        "within_group": {},
                        "cross_group": {}
                    } for stype in split_types
                }
            }

            scaler = StandardScaler()
            
            if di == 0 and model_name == "MLP":
                model.max_iter = 4000
            if di == 1 and model_name == "RF":
                model.random_state = 1
            if (di == 1 and model_name == 'MLP') or (di == 1 and model_name == 'SVR(linear)'):
                scaler = MinMaxScaler()

            pipeline = make_pipeline(scaler, model) 
                
            # 1.Cross validation of the entire dataset
            overall_cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)
            overall_scores = {
                metric: -np.mean(overall_cv_results['test_' + metric]) 
                if metric != 'r2' 
                else np.mean(overall_cv_results['test_' + metric]) 
                for metric in scoring
            }
            ss[dsn][model_name]['overall'] = {metric: round(float(score), 4) for metric, score in overall_scores.items()}
            
            # 2.within-subgroup cross-validation of rule subgroups
            within_group_cv_results = within_group_tests(X, y, is_rule_satisfied, kf, scoring, pipeline)
            ss[dsn][model_name]['split']['rule']['within_group'] = {
                group: {metric: round(score, 4) for metric, score in scores.items()}
                for group, scores in within_group_cv_results.items()
            }
        
            # 3.cross-subgroup evaluation of rule subgroups
            g1_train_g2_test_scores, g2_train_g1_test_scores = cross_group_test(X, y, is_rule_satisfied, scoring, pipeline)
    
            ss[dsn][model_name]['split']['rule']['cross_group'] = {
                'Group1 Train, Group2 Test': {metric: round(score, 4) for metric, score in g1_train_g2_test_scores.items()},
                'Group2 Train, Group1 Test': {metric: round(score, 4) for metric, score in g2_train_g1_test_scores.items()}
            }
    
            # 4.within-subgroup cross-validation of random subgroups
            np.random.seed(42)
            random_groups_list = [np.random.permutation(is_rule_satisfied) for _ in range(10)]
            cv_scores_list = []
            
            for random_groups in random_groups_list:
                group_scores = within_group_tests(X, y, random_groups, kf, scoring, pipeline)
                cv_scores_list.append(group_scores)
                
            df_random = pd.json_normalize(cv_scores_list)
            avg_values = df_random.mean().to_dict()
            ss[dsn][model_name]['split']['random']['within_group'] = {
                'Group False': {
                    'rmse': round(avg_values['Group False.rmse'], 4),
                    'mape': round(avg_values['Group False.mape'], 4)
                },
                'Group True': {
                    'rmse': round(avg_values['Group True.rmse'], 4),
                    'mape': round(avg_values['Group True.mape'], 4)
                }
            }
        
            # 5.cross-subgroup evaluation of random subgroups
            cross_group_scores_list = []
            for random_groups in random_groups_list:
                group1_train_group2_test_scores, group2_train_group1_test_scores = cross_group_test(X, y, random_groups, scoring, pipeline)
                cross_group_scores_list.append({
                    'group1_train_group2_test': group1_train_group2_test_scores,
                    'group2_train_group1_test': group2_train_group1_test_scores
                })
            
            df_cross = pd.json_normalize(cross_group_scores_list)
            avg_values2 = df_cross.mean().to_dict()

            ss[dsn][model_name]['split']['random']['cross_group'] = {
                'Group1 Train, Group2 Test': {
                    'rmse': round(avg_values2['group1_train_group2_test.rmse'], 4),
                    'mape': round(avg_values2['group1_train_group2_test.mape'], 4)
                },
                'Group2 Train, Group1 Test': {
                    'rmse': round(avg_values2['group2_train_group1_test.rmse'], 4),
                    'mape': round(avg_values2['group2_train_group1_test.mape'], 4)
                }
            }

            print(model_name, ": ", end="\t")
            pprint.pprint(ss[dsn][model_name], compact=True)
        
    return ss


def calculate_weighted(group1, group2, group1_size, group2_size):
    total_size = group1_size + group2_size
    return [(group1_size * group1[i] + group2_size * group2[i]) / total_size for i in range(len(group1))]
    

def create_2x16_matrix(data):
    matrix = np.zeros((2, 16))
    for i, block in enumerate(data):
        start_col = i * 2
        matrix[0:2, start_col:start_col + 2] = block
    return matrix


def plot_heatmap(di, ax, data, title, metric, models, is_rule=False):
    """plot heatmaps for rule subgroups and random subgroups"""
    matrix = create_2x16_matrix(data)

    # color range
    if metric == 'mape':
        vmin = 0
        vmax = 350
    elif metric == 'rmse':
        if di == 0:
            vmin = 0
            vmax = 3300
        elif di == 1:
            vmin = 0
            vmax = 500
        
    img = ax.imshow(matrix, cmap='Blues', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title, loc='center', y=1.43 if is_rule else -0.53)
    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(2))
    xticklabels = []
    yticklabels = ['G1', 'G2']
    if not is_rule:
        for model in models:
            xticklabels.extend(['G1', 'G2'])
    ax.set_xticklabels(xticklabels, ha='center')
    ax.set_yticklabels(yticklabels)

    # add numerical values
    for i, block in enumerate(data):
        start_col = i * 2
        for row in range(2):
            for col in range(2):
                value = matrix[row, start_col + col]
                if metric == 'mape':
                    color = 'white' if value > 275 and is_rule else 'black'
                    fontsize = 14
                elif metric == 'rmse' and di == 0:
                    color = 'white' if value > 2000 and is_rule else 'black'
                    fontsize = 12
                else:
                    color = 'white' if value > 400 and is_rule else 'black'
                    fontsize = 13
                ax.annotate(f'{value:.2f}', xy=(start_col + col, row), ha='center', va='center', color=color, fontsize=fontsize)

    # Label algorithm name and braces (only for rule subgroups)
    if is_rule:
        for i, model in enumerate(models):
            center_x = i * 2 + 0.5
            ax.text(center_x, -0.8, r'$\{$', ha='center', va='center', rotation=-90, fontsize=22)
            ax.text(center_x, -1.05, model, ha='center', va='center', fontsize=16)

    # bold border
    for i in range(len(data)):
        start_col = i * 2
        ax.plot([start_col - 0.5, start_col + 2 - 0.5], [2 - 0.5, 2 - 0.5], color='black', linewidth=3)
        ax.plot([start_col - 0.5, start_col + 2 - 0.5], [-0.5, -0.5], color='black', linewidth=3)
        ax.plot([start_col - 0.5, start_col - 0.5], [-0.5, 2 - 0.5], color='black', linewidth=3)
        ax.plot([start_col + 2 - 0.5, start_col + 2 - 0.5], [-0.5, 2 - 0.5], color='black', linewidth=3)

    return img


def calculate_SDSS(values):
    """calculate Subgroup Discovery Similarity Score (SDSS)"""
    
    diagonal_mean = (values[0][0][0] + values[0][1][1]) / 2
    off_diagonal_mean = (values[0][0][1] + values[0][1][0]) / 2
    similarity_score = diagonal_mean / off_diagonal_mean
    return similarity_score


def view_rule_validation(metric: str):
    """
    Create bar charts and heatmap grids to display the results of 
    nosubgrouping , rule subgroup, and random subgroup for each dataset. 
    """
    
    ds = read_jsonzip(os.path.join("data", "datasets.json.gz"))
    rules = read_jsonzip(os.path.join("data", "rules.json.gz"))
    splits = read_jsonzip(os.path.join("data", "splits.json.gz"))
    split_types = ["rule", "random"]

    # configure overall plot
    # fig = plt.figure(figsize=(20, 13))
    # gs = gridspec.GridSpec(2, 2, width_ratios=[1.6, 3.4])
    fig = plt.figure(figsize=(17, 13))
    gs = gridspec.GridSpec(2, 1, hspace=0.6)
    
    # loop over each dataset
    for di, dsn in enumerate(list(ds)):
        
        # configure plot
        # ax1 = fig.add_subplot(gs[di, 0])
        ax2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[di], hspace=-0.2)
        ax_rule = fig.add_subplot(ax2[0])
        ax_random = fig.add_subplot(ax2[1])
    
        # get dataset
        print(f"{di+1}: {dsn}")
        df = pd.DataFrame(json.loads(ds[dsn]["df"]))
        t = ds[dsn]["target"]

        # get models name
        models = list(splits[dsn].keys())

        nosubgrouping_list = []
        g1_list = []
        g2_list = []
        rule_all = []
        random_all = []
        
        for model in models:
            
            # get MAPE with no subgrouping
            nosubgrouping = splits[dsn][model]["overall"][metric]
            
            # get the MAPE of two rule subgroups
            rule_within_group = splits[dsn][model]["split"]["rule"]["within_group"]
            group1 = rule_within_group.get("Group False", {}).get(metric)
            group2 = rule_within_group.get("Group True", {}).get(metric)

            # get the MAPE of within-subgroup and cross-subgroup for rule subgrouping
            rule_within_group = splits[dsn][model]['split']['rule']['within_group']
            rule_cross_group = splits[dsn][model]['split']['rule']['cross_group']
            rule_g1_within = rule_within_group.get('Group True', {}).get(metric, 0)
            rule_g2_within = rule_within_group.get('Group False', {}).get(metric, 0)
            rule_g1_cross = rule_cross_group.get('Group1 Train, Group2 Test', {}).get(metric, 0)
            rule_g2_cross = rule_cross_group.get('Group2 Train, Group1 Test', {}).get(metric, 0)

            # get the MAPE of within-subgroup and cross-subgroup for random subgrouping
            random_within_group = splits[dsn][model]['split']['random']['within_group']
            random_cross_group = splits[dsn][model]['split']['random']['cross_group']
            random_g1_within = random_within_group.get('Group True', {}).get(metric, 0)
            random_g2_within = random_within_group.get('Group False', {}).get(metric, 0)
            random_g1_cross = random_cross_group.get('Group1 Train, Group2 Test', {}).get(metric, 0)
            random_g2_cross = random_cross_group.get('Group2 Train, Group1 Test', {}).get(metric, 0)

            # gather data
            nosubgrouping_list.append(nosubgrouping)
            g1_list.append(group1)
            g2_list.append(group2)
            rule_all.append([
                [rule_g1_within, rule_g1_cross],
                [rule_g2_cross, rule_g2_within]
            ])
            random_all.append([
                [random_g1_within, random_g1_cross],
                [random_g2_cross, random_g2_within]
            ])

        if di == 0:
            g1_size = 72
            g2_size = 90
        else:
            g1_size = 890
            g2_size = 223
        group_weighted_list = calculate_weighted(g1_list, g2_list, g1_size, g2_size)

        Rule_SDSS_list = []
        Random_SDSS_list = []
        for i in range(len(rule_all)):
            Rule_SDSS_list.append(calculate_SDSS([rule_all[i]]))
            Random_SDSS_list.append(calculate_SDSS([random_all[i]]))
        
        for i, (Rule_SDSS, Random_SDSS) in enumerate(zip(Rule_SDSS_list, Random_SDSS_list)):
            print(f"{models[i]:<12} - Rule SDSS: {Rule_SDSS * 100:>6.2f}%,\t\t Random SDSS: {Random_SDSS * 100:>6.2f}%")

        rule_mean = np.mean(Rule_SDSS_list) * 100
        rule_std = np.std(Rule_SDSS_list) * 100
        random_mean = np.mean(Random_SDSS_list) * 100
        random_std = np.std(Random_SDSS_list) * 100
        print(f"{'Average':<12} - Rule SDSS: {rule_mean:.2f}% ± {rule_std:.2f}%,\t Random SDSS: {random_mean:.2f}% ± {random_std:.2f}%")

        # a：MAPE bar chart
        # x = np.arange(len(models))
        # width = 0.45
        # rects1 = ax1.bar(x - width / 2, nosubgrouping_list, width, label='All', edgecolor='black')
        # rects2 = ax1.bar(x + width / 2, group_weighted_list, width, label='Subgroup', edgecolor='black')
        # ax1.set_xticks(x)
        # ax1.set_xticklabels(models, rotation=45)
        # ax1.set_ylabel(f"{metric.upper()} ({t})")
        # if metric == 'mape':
        #     ax1.set_ylim(0, 100)
        #     ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        # ax1.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        # ax1.text(-0.15, 1.13, '(a)', transform=ax1.transAxes, fontsize=24)
        
        # b: heat map (rule subgroup and random subgroup)
        img_rule = plot_heatmap(di, ax_rule, rule_all, "Rule Subgroup", metric, models, is_rule=True)
        ax_rule.text(-0.10, 1.8, '(a)', transform=ax_rule.transAxes, fontsize=24)
        img_random = plot_heatmap(di, ax_random, random_all, "Random Subgroup", metric, models)
        
        cbar_ax_rule = fig.add_axes([0.33, 0.95, 0.37, 0.006])  # [left, bottom, width, height]
        cbar_rule = fig.colorbar(img_rule, cax=cbar_ax_rule, orientation='horizontal', shrink=0.11)
        cbar_ax_random = fig.add_axes([0.33, 0.48, 0.37, 0.006])  # [left, bottom, width, height]
        cbar_random = fig.colorbar(img_random, cax=cbar_ax_random, orientation='horizontal', shrink=0.11)
        if metric == 'mape':
            cbar_rule.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
            cbar_random.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.0f}%'))
        cbar_rule.ax.xaxis.set_ticks_position('top')
        cbar_random.ax.xaxis.set_ticks_position('top')
        cbar_rule.ax.tick_params(axis='x', labelsize=15)
        cbar_random.ax.tick_params(axis='x', labelsize=15)
        
        # remove all scale lines
        ax_rule.tick_params(left=False, bottom=False)
        ax_random.tick_params(left=False, bottom=False)
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_BASEPATH, f"RuleValidation({metric}).png"), bbox_inches="tight")
    plt.show()
    
    
class NonlinearTransAndDesc(TransformerMixin):
    @_deprecate_positional_args
    def __init__(self, n=2, nonlineartrans_list=None):
        self.n = n
        self.nonlineartrans_list = nonlineartrans_list
        
        # Get the MPI communicator and the rank of the current process
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    
    @staticmethod
    def _combinations(X, N, nonlineartrans_list):
        
        # note: When certain features are negative, certain nonlinear transformations
        # (such as sqrt (x), lnx) cannot be performed, and the root of negative numbers
        # cannot be opened when it is 0

        # total number of nonlinear features after the transformation of all features
        n_transformed_features = 0
        # list of the number of transformations for each feature
        transformed_features_num = []
        # index of the first column after each feature transformation
        groups = []
        
        for col in range(X.shape[1]):
            nontrans_num = 0
            # check if negative powers are included
            for i in range(1, 10):
                if f'x^-{i}' in nonlineartrans_list and (X.iloc[:, col] == 0).any():
                    nontrans_num += 1
            if 'sqrt(x)' in nonlineartrans_list:
                if (X.iloc[:, col] < 0).any():
                    nontrans_num += 1
            if 'lnx' in nonlineartrans_list:
                if (X.iloc[:, col] <= 0).any():
                    nontrans_num += 1
                    
            transformed_features = len(nonlineartrans_list) - nontrans_num
            transformed_features_num.append(transformed_features)
            n_transformed_features += transformed_features
            groups.append(n_transformed_features - transformed_features)

        # obtain the column number combination of N-dimensional descriptors
        descriptor_combinations = list(combinations(groups, N))

        # number of feature columns to be multiplied
        feature_combinations = []
        for combination in descriptor_combinations:
            group_features = []
            for group in combination:
                # less than n of nonlinear transformation in the features
                # the number of features corresponding to the nonlinear transformation in the current group
                index_num = groups.index(group)
                num_features = transformed_features_num[index_num]
                
                # create a sublist based on the number of features
                group_features.append([j for j in range(group, group + num_features)])
                
            feature_combinations.extend(list(product(*group_features)))
            
        return n_transformed_features, len(feature_combinations), feature_combinations

    
    def featuretransformation(self, X):
        
        X_transformed = pd.DataFrame()
        col_num = X.shape[1] 

        for i in range(0, col_num):
            col = X.iloc[:, i]
            
            if (col != 0).all():
                if 'x^-3' in self.nonlineartrans_list:
                    X_transformed.insert(X_transformed.shape[1], col.name + '_rec3', np.power(col.astype(float), -3))
                if 'x^-2' in self.nonlineartrans_list:
                    X_transformed.insert(X_transformed.shape[1], col.name + '_rec2', np.power(col.astype(float), -2))
                if 'x^-1' in self.nonlineartrans_list:
                    X_transformed.insert(X_transformed.shape[1], col.name + '_rec', np.power(col.astype(float), -1))
                    
            if 'x' in self.nonlineartrans_list:
                X_transformed.insert(X_transformed.shape[1], col.name, np.power(col.astype(float), 1)) 
                
            if 'sqrt(x)' in self.nonlineartrans_list:
                if (col >= 0).all():
                    X_transformed.insert(X_transformed.shape[1], col.name + '_sqrt', np.sqrt(abs(col.astype(float))))
                    
            if 'x^2' in self.nonlineartrans_list:
                X_transformed.insert(X_transformed.shape[1], col.name + '_square', np.power(col.astype(float), 2))
                
            if 'x^3' in self.nonlineartrans_list:
                X_transformed.insert(X_transformed.shape[1], col.name + '_cubic', np.power(col.astype(float), 3))
                
            if 'e^x' in self.nonlineartrans_list:
                X_transformed.insert(X_transformed.shape[1], col.name + '_exp', np.exp(col.astype(float)))
                
            if 'lnx' in self.nonlineartrans_list:
                if (col > 0).all():
                    X_transformed.insert(X_transformed.shape[1], col.name + '_ln', np.log(abs(col.astype(float))))

        return X_transformed

    
    def fit(self, X, y=None):
        n_samples, n_features = check_array(X, accept_sparse=True).shape
        
        n_transformed_features, descriptor_count, feature_combinations = self._combinations(
            X, 
            self.n,
            self.nonlineartrans_list
        )

        # sample number
        self.n_samples_ = n_samples
        
        # input features number
        self.n_input_features_ = n_features
        
        # nonlinear features number
        self.n_transformed_features = n_transformed_features
        
        # constructed descriptors number
        self.n_output_features_ = descriptor_count 
        
        # feature combinations
        self.feature_combinations = feature_combinations
        
        # target property
        self.y = y
        
        return self

    
    def transform(self, X):
        X_trans = self.featuretransformation(X)
        check_is_fitted(self)
        X_des = pd.DataFrame()

        # Distribute the column combinations across different processes
        col_combinations = self.feature_combinations
        
        # MPI parallelization: local column index combination
        local_col_combinations = np.array_split(col_combinations, self.size)[self.rank]

        """
        Group the nonlinear transformed features.
        Obtain column index numbers for each possible feature multiplication based on the dimensionality of the input descriptor
        """
        for i in local_col_combinations:
            col_indexes = list(i)
            col_names = X_trans.columns[col_indexes]
            new_col_name = '*'.join(map(str, col_names))

            col_values = X_trans.iloc[:, col_indexes].to_numpy()
            new_col = pd.Series(np.prod(col_values, axis=1), name=new_col_name, index=X.index)
            X_des = pd.concat([X_des, new_col], axis=1)

        # Merge the columns generated by different processes
        all_X_des = self.comm.gather(X_des, root=0)
        if self.rank == 0:
            X_des = pd.concat(all_X_des, axis=1)
            data = pd.concat([X_des, self.y], axis=1)
            
            return data


def descriptors_check(X_des):
    
    # loop through each column
    for column in X_des.columns:
        
        # Check if a column contains only inf, -inf, and NaN values
        if ((X_des[column] == np.inf) | (X_des[column] == -np.inf) | X_des[column].isnull()).all():
            # If the condition is true, drop the column
            X_des.drop([column], axis=1, inplace=True)
            continue
            
        # Check if a column contains only two of the following cases: 
        # positive infinity and negative infinity；
        # positive infinity and NaN;
        # negative infinity and NaN
        if (((X_des[column] == np.inf) | (X_des[column] == -np.inf)).any() and
            ((X_des[column] == np.inf) | X_des[column].isnull()).any() and
            ((X_des[column] == -np.inf) | X_des[column].isnull()).any()):
            # drop the column
            X_des.drop([column], axis=1, inplace=True)
            continue
            
        # If it reaches here, the column contains only one type of these values or none of them
        # Check if there are positive infinity values. If so, replace them with the maximum value
        # of the column excluding inf values
        if (X_des[column] == np.inf).any():
            max_value = X_des[column][X_des[column] != np.inf].max()
            X_des[column].replace(np.inf, max_value, inplace=True)
            
        # Check if there are negative infinity values. If so, replace them with the minimum value
        # of the column excluding -inf values.
        if (X_des[column] == -np.inf).any():
            min_value = X_des[column][X_des[column] != -np.inf].min()
            X_des[column].replace(-np.inf, min_value, inplace=True)
            
        # Use the isna() or isnull() method to check if there are NaN or null values
        if X_des[column].isnull().any():
            X_des[column].fillna(X_des[column].mean(), inplace=True)

    return X_des


def descriptors_filter(X, y):
    correlation = X.corrwith(y)
    sorted_selected = correlation.abs().sort_values(ascending=False)
    top = sorted_selected.head(15).index
    top_descriptors = top.tolist()
    top_coefficients = [correlation[desc] for desc in top]

    return top_descriptors, top_coefficients


def regression_prediction(X, y, force_descriptor=None):
    """linear regression"""
    
    # descriptor outlier check
    X = descriptors_check(X)
    
    # Pearson correlation screening
    top_descriptors, top_coefficients = descriptors_filter(X, y)
    X_top = X[top_descriptors]

    out = pd.DataFrame(columns=['Eq', 'w', 'b', 'r', 'R2', 'RMSE', 'MAE', 'MAPE'])
    r_max = 0
    best_out = []
    col_indices = X_top.shape[1]

    for i in range(0, col_indices):
        des_name = X_top.columns[i]
        col = X_top.iloc[:, [i]]
        
        # standardscaler
        min_max_scaler = StandardScaler()
        X_minMax = min_max_scaler.fit_transform(col)

        # regression
        model = LinearRegression()
        model.fit(X_minMax, y) 
        y_pred = model.predict(X_minMax)
        
        w = round(float(model.coef_), 2)
        b = round(float(model.intercept_), 2)
        r = np.corrcoef(y, y_pred.flatten())[0, 1]
        R2 = round(r2_score(y, y_pred), 2)
        RMSE = round(rmse(y, y_pred), 2)
        MAE = round(mean_absolute_error(y, y_pred), 2)
        MAPE = round(mape(y, y_pred), 2)
        
        out.loc[i] = [des_name, w, b, r, R2, RMSE, MAE, MAPE]

        # ξ = 1
        if force_descriptor and des_name == force_descriptor:
            r_max = r
            best_out = [des_name, w, b, r, R2, RMSE, MAE, MAPE, y_pred]
        # ξ = 4
        elif r > r_max:
            r_max = r
            best_out = [des_name, w, b, r, R2, RMSE, MAE, MAPE, y_pred]
    
    # descriptors sort
    out.sort_values(by="r", inplace=True, ascending=False)
    display(out)

    y = y.to_frame()
    y = y.rename(columns={y.columns[0]: 'y'}).reset_index(drop=True)
    y_pred = pd.DataFrame(best_out[8], columns=['y_pred'])  

    return best_out[1], best_out[2], best_out[3], y, y_pred


# sigmoid membership function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# tanh membership function
def tanh(x):
    return np.tanh(x)


# rectangular membership function
def rectangular_membership(x, a, b):
    return 1 if a <= x < b else 0


# bell membership function
def bell_membership(x, a, b, c):
    return 1 / (1 + abs((x - c) / a) ** (2 * b))


# gaussian membership function
def gaussian_membership(x, c, sigma):
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


# sigmoid membership function
def sigmoid_membership(x, c, a):
    return 1 / (1 + math.exp(-a * (x - c)))