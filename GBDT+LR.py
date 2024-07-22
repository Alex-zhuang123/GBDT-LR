import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time,datetime
import variable_bin_methods as varbin_meth
import variable_encode as var_encode
import toad
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import bayes_opt as bo
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc,confusion_matrix,recall_score,precision_score,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import  OneHotEncoder
from imblearn.over_sampling import BorderlineSMOTE
import missingno as msno
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
import pickle
import warnings
warnings.filterwarnings("ignore") ##忽略警告

######子函数
##删除缺失值多的样本
def del_na(df,colname_1,rate):
    ##df: dataframe
    ##colname_1: 列名list
    ##rate:缺失值比例，大于rate变量删除
    na_cols = df[colname_1].isna().sum().sort_values(ascending=False)/float(df.shape[0])
    na_del = na_cols[na_cols >= rate]
    df = df.drop(na_del.index, axis=1)
    return df,na_del

##目标变量映射字典
def target_mapping(lst):
    ##Late (31-120 days)、Default、Charged Off映射为1，坏样本
    ##Late (16-30 days)、In Grace Period映射为2,不确定样本
    ##Current、Fully Paid映射为0，好样本
    mapping = {}
    for elem in lst:
        if elem in ["Charged Off", "Late (31-120 days)" ]:
            mapping[elem] = 1
        elif elem in ['In Grace Period','Late (16-30 days)',]:
            mapping[elem] = 2
        elif elem in ['Current','Fully Paid']:
            mapping[elem] = 0
        else:
            mapping[elem] = 3
    return mapping

# 删除常量
def constant_del(df, cols):
    dele_list = []
    for col in cols:
        # remove repeat value counts
        uniq_vals = list(df[col].unique())
        if pd.isnull(uniq_vals).any():
            if len( uniq_vals ) == 2:
                dele_list.append(col)
                print (" {} 变量只有一种取值,该变量被删除".format(col))
        elif len(df[col].unique()) == 1:
            dele_list.append(col)
            print (" {} 变量只有一种取值,该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df,dele_list

##删除长尾数据
def tail_del(df,cols,rate):
    dele_list = []
    len_1 = df.shape[0]
    for col in cols:
        if len(df[col].unique()) < 5:
            if df[col].value_counts().max()/len_1 >= rate:
                dele_list.append(col)
                print (" {} 变量分布不均衡,该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df,dele_list

##时间格式转化
def trans_format(time_string, from_format, to_format='%Y.%m.%d'):
    ##from_format:原字符串的时间格式
    ##param to_format:转化后的时间格式
    if pd.isnull(time_string):
        return np.nan
    else:
        time_struct = time.strptime(time_string, from_format)
        times = time.strftime(to_format, time_struct)
        times = datetime.datetime.strptime(times, to_format)
        return times
##离散变量与连续变量区分
def category_continue_separation(df,feature_names):
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names:
        feature_names.remove('target')
    ##先判断类型，如果是int或float就直接作为连续变量
    numerical_var = list(df[feature_names].select_dtypes(include=['int','float','int32','float32','int64','float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var,numerical_var

##变量选择
##iv筛选
def iv_selection_func(bin_data, data_params, iv_low=0.02, iv_up=5, label='target'):
    # 简单看一下IV，太小的不要
    selected_features = []
    for k, v in data_params.items():
        if iv_low <= v < iv_up and k in bin_data.columns:
            selected_features.append(k+'_woe')
        else:
            print('{0} 变量的IV值为 {1}，小于阈值删除'.format(k, v))
    selected_features.append(label)
    return bin_data[selected_features]

# 确定最优树的颗数
def xgb_cv(param, x, y, num_boost_round=10000):
    dtrain = xgb.DMatrix(x, label=y)
    cv_res = xgb.cv(param, dtrain, num_boost_round=num_boost_round, early_stopping_rounds=30)
    num_boost_round = cv_res.shape[0]
    return num_boost_round

def train_xgb(params, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, early_stopping_rounds=30, verbose_eval=50):
    """
    训练xgb模型
    """
    dtrain = xgb.DMatrix(x_train, label=y_train)
    if x_test is None:
        num_boost_round = xgb_cv(params, x_train, y_train)
        early_stopping_rounds = None
        eval_sets = ()
    else:
        dtest = xgb.DMatrix(x_test, label=y_test)
        eval_sets = [(dtest, 'test')]
    model = xgb.train(params, dtrain, num_boost_round, evals=eval_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
    return model

def xgboost_bayesian_optimization(params_space, x_train, y_train, x_test=None, y_test=None, num_boost_round=10000, nfold=5, init_points=2, n_iter=5, verbose_eval=0, early_stopping_rounds=30):
    """
    贝叶斯调参, 确定其他参数
    """
    # 设置需要调节的参数及效果评价指标
    def xgboost_cv_for_bo(eta, gamma, max_depth, min_child_weight,
                          subsample, colsample_bytree):
        params = {
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': eta,
            'gamma': gamma,
            'max_depth': int(max_depth),
            'min_child_weight': int(min_child_weight),
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'seed': 1
        }
        if x_test is None:
            dtrain = xgb.DMatrix(x_train, label=y_train)
            xgb_cross = xgb.cv(params,
                               dtrain,
                               nfold=nfold,
                               metrics='auc',
                               early_stopping_rounds=early_stopping_rounds,
                               num_boost_round=num_boost_round)
            test_auc = xgb_cross['test-auc-mean'].iloc[-1]
        else:
            clf_obj = train_xgb(params, x_train, y_train, x_test, y_test, num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval)
            test_auc = roc_auc_score(y_test, clf_obj.predict(xgb.DMatrix(x_test)))
        return test_auc

    # 指定需要调节参数的取值范围
    xgb_bo_obj = bo.BayesianOptimization(xgboost_cv_for_bo, params_space, random_state=1)
    xgb_bo_obj.maximize(init_points=init_points, n_iter=n_iter)
    best_params = xgb_bo_obj.max['params']
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    best_params['eval_metric'] = 'auc'
    best_params['booster'] = 'gbtree'
    best_params['objective'] = 'binary:logistic'
    best_params['seed'] = 1
    return best_params

if __name__ == '__main__':
    path = 'E:\GBDT+LR'
    data_path = os.path.join(path, 'data')
    file_name = 'LoanStats_2019Q1.csv'
    #########读取数据####################################################
    df_1 = pd.read_csv(os.path.join(data_path, file_name), header=1, sep=',', low_memory=False)
    #    df_1.columns

    ########好坏样本定义##################################################
    ##做标签状态映射
    list(df_1["loan_status"].unique())
    ###查看不同标签的样本分布
    df_1.groupby(["loan_status"])[['int_rate']].count()

    df_1.rename(columns={'loan_status': 'target'}, inplace=True)
    df_1 = df_1.loc[~(df_1.target.isnull()),]
    df_1["target"] = df_1["target"].map(target_mapping(df_1["target"].unique()))
    df_1.target.unique()
    df_1 = df_1.loc[df_1.target <= 1,]
    ##样本不均衡非常严重
    print(sum(df_1.target == 0) / df_1.target.sum())

    #################数据清洗与预处理#####################################
    ##1.删除贷后数据
    var_del = ['collection_recovery_fee', 'initial_list_status', 'last_credit_pull_d', 'last_pymnt_amnt',
               'last_pymnt_d', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 'recoveries', 'total_pymnt',
               'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp', 'settlement_percentage']
    df_1 = df_1.drop(var_del, axis=1)

    ##2.删除LC公司信用评估的结果,利率也是LC公司的结果，且利率越高风险越大，也是数据泄露的变量
    var_del_1 = ['grade', 'sub_grade', 'int_rate']
    df_1 = df_1.drop(var_del_1, axis=1)
    #    df_1.isnull().any()

    ##3.查看缺失值情况
    ##看一下数据缺失情况
    desc_list = df_1.describe()
    ##缺失值绘图
    var_list = list(df_1.columns)
    for i in range(1, 4):
        start = (i - 1) * 40
        stop = i * 40
        plt.figure(figsize=(10, 6))
        msno.bar(df_1[var_list[start:stop]], labels=True, fontsize=10)
        plt.xticks(rotation=30)

    ##删除缺失值比率超过40%的变量
    df_1, na_del = del_na(df_1, list(df_1.columns), rate=0.9)
    len(na_del)
    ##删除行全为缺失值
    df_1.dropna(axis=0, how='all', inplace=True)
    df_1.shape

    ###4.删除除缺失值外，只有一种状态的变量
    cols_name = list(df_1.columns)
    cols_name.remove('target')
    df_1, dele_list = constant_del(df_1, cols_name)

    ##5.删除长尾数据
    cols_name_1 = list(df_1.columns)
    cols_name_1.remove('target')
    df_1, dele_list = tail_del(df_1, cols_name_1, rate=0.9)

    ##6.删除一些明显无关的变量
    ##emp_title工作岗级，可以做一个等级划分，这里直接删除。离散程度较大删除，
    ##zip_code邮编信息，离散程度太大
    ##title与purpose一致，直接删除
    len(df_1.emp_title.unique())
    var_del_2 = ['emp_title', 'zip_code', 'title']
    df_1 = df_1.drop(var_del_2, axis=1)

    ##7.数据格式规范化
    df_1.shape
    ##设置全部显示列信息
    pd.set_option('display.max_columns', None)
    df_1.head(5)
    np.unique(df_1.dtypes)

    ##revol_util数据格式规约
    df_1['revol_util'] = df_1['revol_util'].str.replace('%', '').astype('float')

    ##8.日期变量处理
    ##'sec_app_earliest_cr_line'
    var_date = ['issue_d', 'earliest_cr_line', 'sec_app_earliest_cr_line']
    ##时间格式转化
    df_1['issue_d'] = df_1['issue_d'].apply(trans_format, args=('%b-%Y', '%Y-%m',))
    df_1['earliest_cr_line'] = df_1['earliest_cr_line'].apply(trans_format, args=('%b-%Y', '%Y-%m',))
    df_1['sec_app_earliest_cr_line'] = df_1['sec_app_earliest_cr_line'].apply(trans_format, args=('%b-%Y', '%Y-%m',))
    #################特征工程#####################################
    ####尝试做一点特征工程
    ##将时间差值转为月份
    df_1['mth_interval'] = df_1['issue_d'] - df_1['earliest_cr_line']
    df_1['sec_mth_interval'] = df_1['issue_d'] - df_1['sec_app_earliest_cr_line']

    df_1['mth_interval'] = df_1['mth_interval'].apply(lambda x: round(x.days / 30, 0))
    df_1['sec_mth_interval'] = df_1['sec_mth_interval'].apply(lambda x: round(x.days / 30, 0))
    df_1['issue_m'] = df_1['issue_d'].apply(lambda x: x.month)
    ##删除原始日期变量
    df_1 = df_1.drop(var_date, axis=1)

    ##年还款总额占年收入百分比
    index_1 = df_1.annual_inc == 0
    if sum(index_1) > 0:
        df_1.loc[index_1, 'annual_inc'] = 10
    df_1['pay_in_rate'] = df_1.installment * 12 / df_1.annual_inc
    index_s1 = (df_1['pay_in_rate'] >= 1) & (df_1['pay_in_rate'] < 2)
    if sum(index_s1) > 0:
        df_1.loc[index_s1, 'pay_in_rate'] = 1
    index_s2 = df_1['pay_in_rate'] >= 2
    if sum(index_s2) > 0:
        df_1.loc[index_s2, 'pay_in_rate'] = 2
        ##信用借款账户数与总的账户数比
    df_1['credit_open_rate'] = df_1.open_acc / df_1.total_acc
    ##周转余额与所有账户余额比
    df_1['revol_total_rate'] = df_1.revol_bal / df_1.tot_cur_bal
    ##欠款总额和本次借款比
    df_1['coll_loan_rate'] = df_1.tot_coll_amt / df_1.installment
    index_s3 = df_1['coll_loan_rate'] >= 1
    if sum(index_s3) > 0:
        df_1.loc[index_s3, 'coll_loan_rate'] = 1
    ##银行卡状态较好的个数与总银行卡数的比
    df_1['good_bankcard_rate'] = df_1.num_bc_sats / df_1.num_bc_tl
    ##余额大于零的循环账户数与所有循环账户数的比
    df_1['good_rev_accts_rate'] = df_1.num_rev_tl_bal_gt_0 / df_1.num_rev_accts

    #################变量分箱####################################
    ##离散变量与连续变量区分
    categorical_var, numerical_var = category_continue_separation(df_1, list(df_1.columns))
    for s in set(numerical_var):
        if len(df_1[s].unique()) <= 10:
            print('变量' + s + '可能取值' + str(len(df_1[s].unique())))
            categorical_var.append(s)
            numerical_var.remove(s)
            ##同时将后加的数值变量转为字符串
            index_1 = df_1[s].isnull()
            if sum(index_1) > 0:
                df_1.loc[~index_1, s] = df_1.loc[~index_1, s].astype('str')
            else:
                df_1[s] = df_1[s].astype('str')

    ##划分测试集与训练集25
    data_train, data_test = train_test_split(df_1, test_size=0.2, stratify=df_1.target, random_state=25)
    sum(data_train.target == 0) / data_train.target.sum()
    sum(data_test.target == 0) / data_test.target.sum()

    ###连续变量分箱
    dict_cont_bin = {}
    for i in numerical_var:
        dict_cont_bin[i], gain_value_save, gain_rate_save = varbin_meth.cont_var_bin(data_train[i], data_train.target,
                                                                                     method=2, mmin=4, mmax=12,
                                                                                     bin_rate=0.01, stop_limit=0.05,
                                                                                     bin_min_num=20)
    ###离散变量分箱
    dict_disc_bin = {}
    del_key = []
    for i in categorical_var:
        dict_disc_bin[i], gain_value_save, gain_rate_save, del_key_1 = varbin_meth.disc_var_bin(data_train[i],
                                                                                                data_train.target,
                                                                                                method=2, mmin=4,
                                                                                                mmax=10,
                                                                                                stop_limit=0.05,
                                                                                                bin_min_num=20)
        if len(del_key_1) > 0:
            del_key.extend(del_key_1)
    ###删除分箱数只有1个的变量
    if len(del_key) > 0:
        for j in del_key:
            del dict_disc_bin[j]

    ##训练数据分箱
    ##连续变量分箱映射
    df_cont_bin_train = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_train = pd.concat(
            [df_cont_bin_train, varbin_meth.cont_var_bin_map(data_train[i], dict_cont_bin[i])], axis=1)
    ##离散变量分箱映射
    #    ss = data_train[list( dict_disc_bin.keys())]
    df_disc_bin_train = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_train = pd.concat(
            [df_disc_bin_train, varbin_meth.disc_var_bin_map(data_train[i], dict_disc_bin[i])], axis=1)

    ##测试数据分箱
    ##连续变量分箱映射
    df_cont_bin_test = pd.DataFrame()
    for i in dict_cont_bin.keys():
        df_cont_bin_test = pd.concat([df_cont_bin_test, varbin_meth.cont_var_bin_map(data_test[i], dict_cont_bin[i])],
                                     axis=1)
    ##离散变量分箱映射
    #    ss = data_test[list( dict_disc_bin.keys())]
    df_disc_bin_test = pd.DataFrame()
    for i in dict_disc_bin.keys():
        df_disc_bin_test = pd.concat([df_disc_bin_test, varbin_meth.disc_var_bin_map(data_test[i], dict_disc_bin[i])],
                                     axis=1)

    ###组成分箱后的训练集与测试集
    df_disc_bin_train['target'] = data_train.target
    data_train_bin = pd.concat([df_cont_bin_train, df_disc_bin_train], axis=1)
    df_disc_bin_test['target'] = data_test.target
    data_test_bin = pd.concat([df_cont_bin_test, df_disc_bin_test], axis=1)

    data_train_bin.reset_index(inplace=True, drop=True)
    data_test_bin.reset_index(inplace=True, drop=True)

    var_all_bin = list(data_train_bin.columns)
    var_all_bin.remove('target')

    ###WOE编码
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values, var_woe_name = var_encode.woe_encode(data_train_bin, data_path,
                                                                                     var_all_bin, data_train_bin.target,
                                                                                     'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(data_test_bin, data_path, var_all_bin, data_test_bin.target,
                                                      'dict_woe_map', flag='test')

    ####取出训练数据与测试数据
    x_train = df_train_woe[var_woe_name]
    y_train = data_train_bin.target
    x_test = df_test_woe[var_woe_name]
    y_test = data_test_bin.target


    ###xgboost模型
    # 贝叶斯调参
    params_test = {'eta': (0.05, 0.2),
                   'gamma': (0.005, 0.05),
                   'max_depth': (3, 5),
                   'min_child_weight': (0, 3),
                   'subsample': (0.9, 1.0),
                   'colsample_bytree': (0.9, 1.0)}
    optimal_params = xgboost_bayesian_optimization(params_test, x_train, y_train,x_test,y_test, init_points=5,
                                                   n_iter=8)

    print("贝叶斯调参最优参数: ", optimal_params)

    ##训练xgboost模型
    final_xgb_model = train_xgb(optimal_params, x_train, y_train,x_test,y_test)
    auc_score = roc_auc_score(y_test, final_xgb_model.predict(xgb.DMatrix(x_test)))
    print("贝叶斯调参模型AUC: ", auc_score)

    ###得到树的映射结果
    train_new_feature = final_xgb_model.predict(xgb.DMatrix(x_train),pred_leaf=True)
    test_new_feature = final_xgb_model.predict(xgb.DMatrix(x_test),pred_leaf=True)

    ##进行One-hot编码
    enc = OneHotEncoder(dtype='int').fit(train_new_feature)
    df_train = pd.DataFrame(enc.transform(train_new_feature).toarray())
    df_test = pd.DataFrame(enc.transform(test_new_feature).toarray())

    ##合并得到新的数据集
    x_train = x_train.join(df_train)
    x_test = x_test.join(df_test)


#######################################特征筛选#####################################
    train = x_train.join(y_train)
    train.columns = train.columns.astype(str)
    # 利用toad库筛选特征
    quality_data=toad.quality(train,'target',iv_only=True)
    train_selected, dropped = toad.selection.select(train,target = 'target', iv = 0.06, corr = 0.6, return_drop=True)

    print("基于toad筛选得到%s个特征: \n" % len(train_selected.columns), train_selected.columns)

    x = train_selected.drop("target", axis=1)
    # 带L1惩罚项的逻辑回归作为基模型的特征选择
    LR = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
    sf = SelectFromModel(LR)
    x_new = sf.fit_transform(x, train.target)

    selected_cols = x.columns[sf.get_support()].tolist()
    print("基于L1范数筛选得到%s个特征: \n" % len(selected_cols), selected_cols)

    # GBDT作为基模型的特征选择
    x = x[selected_cols]
    sf = SelectFromModel(GradientBoostingClassifier())
    x_new = sf.fit_transform(x, train.target)
    selected_cols = x.columns[sf.get_support()].tolist()
    print("基于树模型筛选得到%s个特征: \n" % len(selected_cols), selected_cols)

    ##样本不均衡非常严重
    print(sum(y_train == 0) / y_train.sum())

##########################################样本不均衡处理######################################
    var_name = list(selected_cols)

    # ####尝试一点样本生成#####
    # ##随机抽取一些样本
    # df_1_normal = df_1[df_1.target == 0]
    # df_1_normal.reset_index(drop=True, inplace=True)
    # index_1 = np.random.randint(low=0, high=df_1_normal.shape[0] - 1, size=20000)
    # index_1 = np.unique(index_1)
    # df_temp = df_1_normal.loc[index_1]
    # index_2 = [x for x in range(df_1_normal.shape[0]) if x not in index_1]
    # df_temp_other = df_1_normal.loc[index_2]
    # df_temp = pd.concat([df_temp, df_1[df_1.target == 1]], axis=0, ignore_index=True)
    # ##用随机抽取的样本做样本生成
    # sm_sample_1 = BorderlineSMOTE(random_state=10, sampling_strategy=1, k_neighbors=5,kind="borderline-1")
    # x_train, y_train = sm_sample_1.fit_resample(df_temp[var_name], df_temp.target)
    #
    #
    # ##合并数据
    # x_train = np.vstack([x_train, np.array(df_temp_other[var_name])])
    # y_train = np.hstack([y_train, np.array(df_temp_other.target)])

    # print(sum(y_train == 0) / sum(y_train))
    x_train = train[var_name]
    y_train = train.target


    test = x_test.join(y_test)
    test.columns = test.columns.astype(str)
    x_test = test[var_name]
    y_test = test.target


#####################################logistic模型#######################################
    ##设置待优化的超参数
    lr_param = {'C': [0.001,0.005,0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1},  {1: 2, 0: 1}, {1: 3, 0: 1}, {1: 5, 0: 1}]}
    ##初始化网格搜索
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga',max_iter=10000),
        param_grid=lr_param, cv=8, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lr_gsearch.fit(x_train, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    ##用最优参数，初始化logistic模型
    LR_model = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])
    ##训练logistic模型
#    LR_model = LogisticRegression(C=0.01, penalty='l2', solver='saga',
#                                    class_weight={1: 3, 0: 1})
    LR_model_fit = LR_model.fit(x_train, y_train)


    # 保存模型
    with open('my_model.pkl', 'wb') as file:
        pickle.dump(LR_model_fit, file)

#############################模型评估##########################################
    # 载入模型
    with open('my_model.pkl', 'rb') as file:
        LR_model_fit = pickle.load(file)

    ##模型评估
    y_pred = LR_model_fit.predict(x_test)
    ##计算混淆矩阵与recall、precision
    cnf_matrix = confusion_matrix(y_test, y_pred)
    recall_value = recall_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    print(cnf_matrix)
    print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                                                                             precision_value))

    ##给出概率预测结果
    y_score_test = LR_model_fit.predict_proba(x_test)[:, 1]
    ##计算AR。gini等
    fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    ar = 2 * roc_auc - 1
    print('test set:  model AR is {0},and ks is {1},auc={2}'.format(ar,
                                                                    ks, roc_auc))

    ####ks曲线
    plt.figure(figsize=(10, 6))
    fontsize_1 = 12
    plt.plot(np.linspace(0, 1, len(tpr)), tpr, '--', color='black')
    plt.plot(np.linspace(0, 1, len(tpr)), fpr, ':', color='black')
    plt.plot(np.linspace(0, 1, len(tpr)), tpr - fpr, '-', color='grey')
    plt.grid()
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.xlabel('概率分组', fontsize=fontsize_1)
    plt.ylabel('累积占比%', fontsize=fontsize_1)

    ####ROC曲线
    plt.figure(figsize=(10, 6))
    lw = 2
    fontsize_1 = 16
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.xlabel('FPR', fontsize=fontsize_1)
    plt.ylabel('TPR', fontsize=fontsize_1)
    plt.title('ROC', fontsize=fontsize_1)
    plt.legend(loc="lower right", fontsize=fontsize_1)