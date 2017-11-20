
import pandas as pd

df_data = pd.read_csv("C:\\data\\train.csv")

df_data=df_data.drop(["Name","PassengerId","Ticket","Cabin"],axis=1)
sex_unique = df_data.Sex.unique()
sex_mapping = dict(zip(sex_unique,range(0,len(sex_unique))))
df_data["Sex"]=df_data["Sex"].map(sex_mapping)
df_data["Age"] = df_data["Age"].fillna(value=df_data["Age"].median())
df_data =  pd.get_dummies(df_data,columns=["Embarked","Pclass"],dummy_na=True)

# 特征标准化
from sklearn import preprocessing

train_Y=df_data["Survived"].values
train_X=df_data.ix[:,1:].values
scaler = preprocessing.StandardScaler
scaler(train_X)

# 特征选择

# 训练测试分割
import sklearn.model_selection as select
x_train, x_test, y_train, y_test = select.train_test_split(train_X,train_Y,test_size=0.25)

# -----------------------------------------------------------------------------------------------------------------------
# 逻辑回归
# penalty='l2' 正则化
# dual=False
# tol=1e-4
# C=1.0,
# fit_intercept=True
# intercept_scaling=1
# class_weight=None, 分类权重
# random_state=None
# solver='liblinear' 损失函数优化算法
# max_iter=100, 只对newton-cg, sag and lbfgs solvers.有效
# multi_class='ovr'
# verbose=0
# warm_start=False
# n_jobs=1
# LogisticRegression和LogisticRegressionCV的主要区别是LogisticRegressionCV使用了交叉验证来选择正则化系数C。
# 而LogisticRegression需要自己每次指定一个正则化系数。
# 除了交叉验证，以及选择正则化系数C以外， LogisticRegression和LogisticRegressionCV的使用方法基本相同。

# penalty,在调参时如果我们主要的目的只是为了解决过拟合，一般penalty选择L2正则化就够了。
# 但是如果选择L2正则化发现还是过拟合，即预测效果差的时候，就可以考虑L1正则化。
# 另外，如果模型的特征非常多，我们希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化。

# solver,penalty参数的选择会影响我们损失函数优化算法的选择。即参数solver的选择.
# 如果是L2正则化，那么4种可选的算法{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}都可以选择。
# 但是如果penalty是L1正则化的话，就只能选择‘liblinear’了。这是因为L1正则化的损失函数不是连续可导的
            # a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
            # b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
            # c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
            # d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择

# class_weight参数用于标示分类模型中各种类型的权重，可以不输入。
# 　　　　如果class_weight选择balanced，那么类库会根据训练样本量来计算权重。某种类型样本量越多，则权重越低，样本量越少，则权重越高。
# 　　　　那么class_weight有什么作用呢？在分类模型中，我们经常会遇到两类问题：
# 　　　　第一种是误分类的代价很高。比如我们宁愿将合法用户分类为非法用户，我们可以适当提高非法用户的权重。
# 　　　　第二种是样本是高度失衡的，比如我们有合法用户和非法用户的二元样本数据10000条，里面合法用户有9995条，非法用户只有5条，如果我们不考虑权重，则我们可以将所有的测试集都预测为合法用户，这样预测准确率理论上有99.95%，但是却没有任何意义。这时，我们可以选择balanced，让类库自动提高非法用户样本的权重。
# 　　　　提高了某种分类的权重，相比不考虑权重，会有更多的样本分类划分到高权重的类别，从而可以解决上面两类问题。
# 　　　　当然，对于第二种样本失衡的情况，我们还可以考虑用下一节讲到的样本权重参数： sample_weight，而不使用class_weight。sample_weight在下一节讲。
#
# sample_weight
# 　　　　上一节我们提到了样本不失衡的问题，由于样本不平衡，导致样本不是总体样本的无偏估计，从而可能导致我们的模型预测能力下降。遇到这种情况，我们可以通过调节样本权重来尝试解决这个问题。调节样本权重的方法有两种，第一种是在class_weight使用balanced。第二种是在调用fit函数时，通过sample_weight来自己调节每个样本权重。
# 　　　　在scikit-learn做逻辑回归时，如果上面两种方法都用到了，那么样本的真正权重是class_weight*sample_weight.
#
# 　　　　还有些参数比如正则化参数C（交叉验证就是 Cs），迭代次数max_iter

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

lr = LogisticRegression()
lrcv = LogisticRegressionCV()
lr.fit(x_train,y_train)
lrcv.fit(x_train,y_train)
print(lr.score(x_test,y_test))
print(lrcv.score(x_test,y_test))


# -----------------------------------------------------------------------------------------------------------------------
# # # 随机森林
#  n_estimators=10,
#  criterion="gini",
#  max_depth=None,
#  min_samples_split=2,
#  min_samples_leaf=1,
#  min_weight_fraction_leaf=0.,
#  max_features="auto",
#  max_leaf_nodes=None,
#  min_impurity_decrease=0.,
#  min_impurity_split=None,
#  bootstrap=True,
#  oob_score=False,
#  n_jobs=1,
#  random_state=None,
#  verbose=0,
#  warm_start=False,
#  class_weight=None):

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(x_train,y_train)
print(rfc.score(x_test,y_test))


# -------------------------------------------------------------------------------------------
# GBDT
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
rfc = GradientBoostingClassifier(n_estimators=500)
rfc.fit(x_train,y_train)
print(rfc.score(x_test,y_test))


# -------------------------------------------------------------------------------------------
# # xgboost
import xgboost as xgboost
gbdt = xgboost.XGBClassifier()
gbdt.fit(x_train,y_train)
print(gbdt.score(x_test,y_test))