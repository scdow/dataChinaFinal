# encoding: utf-8
"""
@Project Name : dataChinaFinal
@File Name    : last_xgb.py  
@Programmer   : LJ
@Start Date   : 2022/11/30 0:14:03
@brief        ：
"""
from sklearn.model_selection import KFold
from lightgbm import log_evaluation, early_stopping
# import lightgbm as lgb
import xgboost as xgb
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def del_PART_ORDER(PART_ORDER):
    """
    读取处理数据
    :return:
    """

    part_order = pd.read_csv(PART_ORDER).fillna(0)


    part_order['MONTH'] = pd.to_datetime(part_order['MONTH']).dt.strftime('%Y-%m-%d')
    part_order['MONTH_INT'] = part_order.apply(month_int_time, axis=1)
    # 去重
    part_order = part_order.drop_duplicates(subset=['PART_NO', 'MONTH_INT'], keep='last')
    # 排序
    part_order = part_order.sort_values(by='MONTH_INT')

    # print(part_order)

    return part_order


def del_PART_LIST(PART_LIST):

    part_list = pd.read_csv(PART_LIST, encoding="GBK")
    part_list['PIDT'] = pd.to_datetime(part_list['PIDT']).dt.strftime('%Y-%m-%d')
    ONEHOT_COLUMNS = ['CAR_CLASS', 'REPAIR_TYPE', 'TYPE_CODE', 'CONSTRUCT_NAME']

    DROP = ['KPDS', 'PIDT']
    for column in ONEHOT_COLUMNS:
        part_list = hot_coding_dispose(part_list, column)

    part_list = part_list.drop(columns=DROP)
    # 去重
    part_list = part_list.drop_duplicates(['PART_NO'],keep='last')

    return part_list

def del_TESTDATA_ID(TESTDATA_ID):
    part_predict_list = []
    with open(TESTDATA_ID, 'r') as f:
        for part in f.readlines():
            part_predict_list.append(part[:-1])

    return part_predict_list




def month_int_time(a):
    """
    从2015-1-1开始当前为第sum个月
    :param a:
    :return:
    """
    month = int(a['MONTH'].split('-')[1]) - 1
    year = (int(a['MONTH'].split('-')[0]) - 2015 ) * 12
    sum = year + month
    return sum

# def hot_encode():


def hot_coding_dispose(data, column):
    """
    使用热编码完成字符串转化
    :param data: 数据集
    :param column: 需要编码和删除的列名
    :return: 合并之后的集合
    """
    # 使用热门编码转化字符串
    code = pd.DataFrame({column: data[column]})
    code_DataFrame = pd.get_dummies(code)
    # merge，改用concat
    data = pd.concat([data, code_DataFrame], axis=1)
    # 删除列
    data = data.drop(columns=column)

    return data


def del_data(part_order, part_list, part_predict_list):
    """
    特征处理
    :param part_order:
    :param part_list:
    :param part_predict_list:
    :return:
    """
    part_train_x={}
    part_train_y1={}
    part_train_y3={}
    # 预测使用
    part_predict_x_lst={}
    part_len = len(part_predict_list)
    FEATURES = ['DM01','REPAIR_CNT','REPAIR_AMOUNT','RUN_DIST','MONTH_INT']
    break_time = 23

    # 合并特征
    part_order_list = pd.merge(part_order, part_list, how='left', left_on='PART_NO', right_on='PART_NO')
    del_feature = ['PART_NO', 'MONTH']
    features_all = [i for i in part_order_list.columns if i not in del_feature]

    for j in range(part_len):
        if (j % 100 == 0):
            print('\r', "数据处理进度：{}%".format(j*100/part_len), end="", flush=True)
        order_part = part_order_list[part_order_list['PART_NO'] == part_predict_list[j]]  # 依次取出零件历史销售数据
        for i in range(23, 81, 10):  ### （23,33,43,53,63，73）
            if i == break_time:
                ### 取i-23<=X<=i的内容
                train_x = order_part[(order_part['MONTH_INT'] <= i) & (order_part['MONTH_INT'] >= i - break_time)][FEATURES]
                ### train_x变为numpy并转置，reshape(1,-1)转化为一行
                train_x = train_x.to_numpy().T.reshape(1, -1)
                ### 获取第i+1个订单值 （未来一个月）
                train_y1 = order_part[(order_part['MONTH_INT'] == i + 1)]['DM01']
                ### 获取i+1,i+2,i+3的订单值的和 （未来三个月）
                train_y3 = np.array([order_part[(order_part['MONTH_INT'] == i + 1) | (order_part['MONTH_INT'] == i + 2) | (order_part['MONTH_INT'] == i + 3)]['DM01'].sum()])
            else:
                ### 取i-23<=X<=i的内容
                train_x_tmp = order_part[(order_part['MONTH_INT'] <= i) & (order_part['MONTH_INT'] >= i - break_time)][FEATURES]
                train_x_tmp = train_x_tmp.to_numpy().T.reshape(1, -1)
                train_y1_tmp = order_part[(order_part['MONTH_INT'] == i + 1)]['DM01']
                train_y3_tmp = np.array([order_part[((order_part['MONTH_INT'] == i + 1) | (order_part['MONTH_INT'] == i + 2) | (order_part['MONTH_INT'] == i + 3))]['DM01'].sum()])
                # 将某个零件的数据合并在一起
                train_x = np.append(train_x, train_x_tmp, axis=0)
                train_y1 = np.append(train_y1, train_y1_tmp, axis=0)
                train_y3 = np.append(train_y3, train_y3_tmp, axis=0)

        # 将所有零件的数据合并在一起
        part_train_x[part_predict_list[j]] = train_x
        part_train_y1[part_predict_list[j]] = train_y1  ###未来一个月
        part_train_y3[part_predict_list[j]] = train_y3  ###未来三个月

        #    最后一个输出数据
        i = 83
        train_x_lst = order_part[(order_part['MONTH_INT'] <= i) & (order_part['MONTH_INT'] >= i - break_time)][FEATURES]
        train_x_lst = train_x_lst.to_numpy().T.reshape(1, -1)
        part_predict_x_lst[part_predict_list[j]] = train_x_lst

    # 模型所用数据

    for part in part_predict_list:
        if part == part_predict_list[0]:
            train_x = part_train_x[part]
            train_y1 = part_train_y1[part]
            train_y3 = part_train_y3[part]
            predict_x_lst = part_predict_x_lst[part]
        else:
            train_x = np.concatenate((train_x, part_train_x[part]))
            train_y1 = np.concatenate((train_y1, part_train_y1[part]))
            train_y3 = np.concatenate((train_y3, part_train_y3[part]))
            predict_x_lst = np.concatenate((predict_x_lst, part_predict_x_lst[part]))

    return train_x, train_y1, train_y3, predict_x_lst, features_all

def format_model_data(part_train_x, part_predict_x_lst):

    return pd.DataFrame(part_train_x), pd.DataFrame(part_predict_x_lst)

def mape_evaluate(preds, train_data):
    """
    评估方法
    :param preds:
    :param train_data:
    :return:
    """
    mape = 0.0
    train_y = train_data.get_label()
    for i in range(0, len(preds)):
        tmp = abs(float(int(train_y[i]) - int(preds[i]))) / (max(float(train_y[i]), float(preds[i])) + 1)
        mape += tmp
    mape /= len(preds)
    return "mape", mape



def train_model_xgb(train_x, train_y, test, features):
    """
    训练预测lgb
    :param train_x:
    :param train_y:
    :param test:
    :param features:
    :return:
    """

    params = {
        'booster': 'gbtree',
        'min_child_weight': 10.0,
        'learning_rate': 0.02,
        'objective': 'reg:squarederror',
        # 'eval_metric': "mape",
        'max_depth': 7,
        'max_delta_step': 1.8,
        'colsample_bytree': 0.4,
        'subsample': 0.8,
        'eta': 0.025,
        'gamma': 0.65,
        'nthread': -1,
        'seed': 111,
    }



    folds = KFold(n_splits=5, shuffle=True, random_state=111)
    test_pred_prob = np.zeros((test.shape[0]))

    feature_importance_df = pd.DataFrame()
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
        print("fold {}".format(fold_ + 1))
        trn_data = xgb.DMatrix(train_x.iloc[trn_idx], label=train_y[trn_idx])
        val_data = xgb.DMatrix(train_x.iloc[val_idx], label=train_y[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        clf = xgb.train(params,
                        trn_data,
                        num_boost_round=1000,
                        evals=watchlist,
                        custom_metric=mape_evaluate,
                        verbose_eval=200,)
                        # early_stopping_rounds=200)
        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["Feature"] = features
        # fold_importance_df["importance"] = clf.feature_importance()
        # fold_importance_df["fold"] = fold_ + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        test_pred_prob += clf.predict(xgb.DMatrix(test), ntree_limit=clf.best_iteration) / folds.n_splits

    return test_pred_prob, feature_importance_df

def file_write(part_list, predict_y1, predict_y3, file):
    """
    写入文件
    :param result:
    :param file:
    :return:
    """
    if os.path.exists(file):
        os.remove(file)

    with open(file, 'a', encoding="UTF-8", ) as f:
        for i in range(len(part_list)):
            f.write("{}:{} {}\n".format(part_list[i], round(predict_y1[i]), round(predict_y3[i])))



if __name__ == '__main__':

    PART_ORDER = "resources\PART_ORDER.csv"
    PART_LIST = "resources\PART_LIST.csv"
    TESTDATA_ID = "resources/testdata_id.txt"


    part_order = del_PART_ORDER(PART_ORDER)
    part_list = del_PART_LIST(PART_LIST)
    part_predict_list = del_TESTDATA_ID(TESTDATA_ID)

    part_train_x, part_train_y1, part_train_y3, part_predict_x_lst, features_all = del_data(part_order, part_list, part_predict_list)

    part_train_x, part_predict_x_lst = format_model_data(part_train_x, part_predict_x_lst)

    predict_y1 = train_model_xgb(part_train_x, part_train_y1, part_predict_x_lst, features_all)
    predict_y3 = train_model_xgb(part_train_x, part_train_y3, part_predict_x_lst, features_all)

    # file_write(part_predict_list, predict_y1[0], predict_y3[0], "res/predict_xgb_去重.txt")
