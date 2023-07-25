import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def load_big_greiner():
    grbgen = pd.read_excel('grbgen.xlsx')
    big_table = pd.read_csv('GRBimpu_update.csv')
    big_table['SN_asso'] = big_table['SN'].notna()
    big_table['z'] = 10 ** big_table['z'] - 1
    big_table.rename(columns={'log.T_ai.': "T_ai", 'log.L_a.': "L_a", 'log_SSFR': 'SSFR', 'log_Mass': 'Mass'},
                     inplace=True)
    big_table = big_table.loc[big_table['T90'].notna()]

    grbgen.loc[grbgen['GRB'] == '081022', 'GRB'] = '081022B'
    grbgen.loc[grbgen['GRB'] == '080802', 'GRB'] = '080802B'
    grbgen.loc[grbgen['GRB'] == '040912X', 'GRB'] = '040912BX'
    grbgen.loc[grbgen['GRB'] == '020409', 'GRB'] = '020409B'
    grbgen.loc[grbgen['GRB'] == '020317X', 'GRB'] = '020317BX'
    grbgen.loc[grbgen['GRB'] == '020305', 'GRB'] = '020305B'
    grbgen.loc[grbgen['GRB'] == '011212X', 'GRB'] = '011212BX'
    grbgen.loc[grbgen['GRB'] == '010629B', 'GRB'] = '010629A'
    grbgen.loc[grbgen['GRB'] == '010220', 'GRB'] = '010220C'
    grbgen.loc[grbgen['GRB'] == '010213X', 'GRB'] = '010213BX'
    grbgen.loc[grbgen['GRB'] == '010126', 'GRB'] = '010126B'
    grbgen.loc[grbgen['GRB'] == '010119S', 'GRB'] = '010119BS'
    grbgen.loc[grbgen['GRB'] == '010104', 'GRB'] = '010104B'
    grbgen.loc[grbgen['GRB'] == '001219', 'GRB'] = '001219C'
    grbgen.loc[grbgen['GRB'] == '001212', 'GRB'] = '001212B'
    grbgen.loc[grbgen['GRB'] == '001011', 'GRB'] = '001011C'
    grbgen.loc[grbgen['GRB'] == '000801', 'GRB'] = '000801B'
    grbgen.loc[grbgen['GRB'] == '000623', 'GRB'] = '000623B'
    grbgen.loc[grbgen['GRB'] == '000620', 'GRB'] = '000620B'
    grbgen.loc[grbgen['GRB'] == '000604', 'GRB'] = '000604B'
    grbgen.loc[grbgen['GRB'] == '000529', 'GRB'] = '000529B'
    grbgen.loc[grbgen['GRB'] == '000424', 'GRB'] = '000424B'
    grbgen.loc[grbgen['GRB'] == '000418', 'GRB'] = '000418B'
    grbgen.loc[grbgen['GRB'] == '000424', 'GRB'] = '000424B'
    grbgen.loc[grbgen['GRB'] == '000408', 'GRB'] = '000408B'
    grbgen.loc[grbgen['GRB'] == '000301C', 'GRB'] = '000301D'
    grbgen.loc[grbgen['GRB'] == '000126', 'GRB'] = '000126D'
    grbgen.loc[grbgen['GRB'] == '991216', 'GRB'] = '991216B'
    grbgen.loc[grbgen['GRB'] == '991105', 'GRB'] = '991105C'
    grbgen.loc[grbgen['GRB'] == '990806', 'GRB'] = '990806B'
    grbgen.loc[grbgen['GRB'] == '990712', 'GRB'] = '990712B'
    grbgen.loc[grbgen['GRB'] == '990627', 'GRB'] = '990627B'
    grbgen.loc[grbgen['GRB'] == '990527', 'GRB'] = '990527C'
    grbgen.loc[grbgen['GRB'] == '990308', 'GRB'] = '990308C'
    grbgen.loc[grbgen['GRB'] == '980706', 'GRB'] = '980706B'
    grbgen.loc[grbgen['GRB'] == '980519', 'GRB'] = '980519B'
    grbgen.loc[grbgen['GRB'] == '980425', 'GRB'] = '980425B'
    grbgen.loc[grbgen['GRB'] == '971214', 'GRB'] = '971214B'
    grbgen.loc[grbgen['GRB'] == '970616', 'GRB'] = '970616B'
    grbgen.loc[grbgen['GRB'] == '020418B', 'GRB'] = '020418A'

    for i, row in big_table.iterrows():
        grb_name = str(row['GRB'])
        if len(grb_name) < 6:
            big_table.loc[i, 'GRB'] = grb_name.zfill(6)
        if grb_name[-1] == 'S':
            big_table.loc[i, 'GRB'] = grb_name[:-1]

    grb_class = -1 * np.ones(len(big_table))
    for i, grb in grbgen[122:].iterrows():
        if grb['GRB'].endswith('S'):
            real_name = grb['GRB'][:-1]
        elif grb['GRB'].endswith('X'):
            real_name = grb['GRB'][:-1]
        else:
            real_name = grb['GRB']
        match_index = np.where(big_table['GRB'] == real_name)[0]
        if match_index.size == 0:
            real_name = real_name + "A"
            match_index = np.where(big_table['GRB'] == real_name)[0]
            if match_index.size == 0:
                pass
                # print(real_name)
            else:
                if grb['GRB'].endswith('S'):
                    grb_class[match_index] = 1
                elif grb['GRB'].endswith('L'):
                    grb_class[match_index] = 0
                else:
                    grb_class[match_index] = 0
        else:
            if grb['GRB'].endswith('S'):
                grb_class[match_index] = 1
            elif grb['GRB'].endswith('X'):
                grb_class[match_index] = 0
            else:
                grb_class[match_index] = 0

    big_table['class'] = grb_class
    big_table.loc[big_table['GRB']=='090426', 'class'] = 0.  # 090426
    big_table.loc[big_table['GRB']=='060505A', 'class'] = 1.  # 060505A
    big_table.loc[big_table['GRB']=='060614A', 'class'] = 1.  # 060614
    return grbgen, big_table


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    '''

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in (cf / np.c_[np.sum(cf,axis=1),np.sum(cf,axis=1)]).flatten()]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    res = sns.heatmap(cf / np.sum(cf, axis=1)[np.newaxis].T, vmin=0, vmax=1, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)
    for _, spine in res.spines.items():
        spine.set_visible(True)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.tight_layout()