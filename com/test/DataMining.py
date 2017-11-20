import numpy as np
import pandas as pd
from scipy.stats import mode

from numpy.distutils.misc_util import is_string
from sklearn import preprocessing


def main():
    dataframe = extractData()#pd.DataFrame([[1,"v",2],[1,2,3]],columns=["name1","name2","name3"])
    # 'Binarizer',
    preprocessing.Binarizer().fit_transform()
    # 'FunctionTransformer',
    # 'Imputer',
    # 'KernelCenterer',
    # 'LabelBinarizer',
    # 'LabelEncoder',
    # 'MultiLabelBinarizer',
    # 'MinMaxScaler',
    # 'MaxAbsScaler',
    # 'QuantileTransformer',
    # 'Normalizer',
    # 'OneHotEncoder',
    # 'RobustScaler',
    # 'StandardScaler',
    # 'add_dummy_feature',
    # 'PolynomialFeatures',
    # 'binarize',
    # 'normalize',
    # 'scale',
    # 'robust_scale',
    # 'maxabs_scale',
    # 'minmax_scale',
    # 'label_binarize',
    # 'quantile_transform',

def is_number(series):
    return series.dtype in (np.int64,int, float)


def is_string(series):
    return  series.dtype in (object, str)

def describe_number(series) :

    desc = series.describe()
    desc = pd.Series([series.name], index=["column"]).append(desc)
    # 众数
    desc = desc.append(pd.Series(mode(series).mode, index=["median"]))
    # 极差
    desc = desc.append(pd.Series([series.ptp()], index=["ptp"]))
    # 变异系数
    desc = desc.append(pd.Series([series.std() / series.mean()], index=["cv"]))
    # 缺失比例
    desc = desc.append(pd.Series([1 - (series.count() / series.index.size)], index=["Nan_percent"]))

    return desc


def describe_str(series):
    # 各值占比
    desc = series.describe()
    count_percent = series.value_counts(sort=True) / series.count()
    # count
    series = pd.Series([count_percent.count()], index=["count:"]).append(count_percent)
    return series


def extractData():
    data = pd.read_csv("c:/data/train.csv");
    return data


if __name__ == '__main__':
    main()
