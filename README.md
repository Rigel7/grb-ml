Analysis code used in paper:<br />
Identifying the physical origin of gamma-ray bursts with supervised machine learning<br />
by Jia-wei Luo et al.

grbgen.xlsx: GRB data taken from https://www.mpe.mpg.de/~jcg/grbgen.html<br />
GRBimpu_update.csv: GRB Big Table data up to 160509A<br />
grb_ml_utils.py: File containing utility functions<br />
trained_all.json: XGBoost model file trained with all data and features<br />
greiner_bigtable.ipynb: Main analysis code<br />
compare_f1.ipynb: Repeated trials to measure the average F1 scores and uncertainties of different feature subgroups<br />
unclassified_grb.ipynb: Classify currently unclassified GRBs<br />
new_intermingled_grbs.ipynb: Classification of two newly discovered intermingled GRBs of GRB 200826A and GRB 211211A<br />
classify_grbs.ipynb: Use the trained model to predict GRBs<br />

Note that the Big Table data included here in only a subset of what we used in our study, thus one cannot 100% reproduce results in our paper with only these code. The full Big Table will be released with another paper. In the meantime, the readers can use the model trained with all the data and features to predict the physical origin of any new GRBs.