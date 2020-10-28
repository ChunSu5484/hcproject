import pandas as pd
import numpy as np

import gc
import os

from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score


def main(debug=False):
    num_rows = 30000 if debug else None

    df = preprocess_train_test(path, num_rows=num_rows)
    print('application df shape:', df.shape)

    bur_df = preprocess_bureau(path, num_rows=num_rows)
    df = pd.merge(df, bur_df, on='SK_ID_CURR', how='left')
    print('bureau df shape:', bur_df.shape)

    prev_df = preprocess_previous_applications(path, num_rows=num_rows)
    print('previous_application df shape:', prev_df.shape)
    df = pd.merge(df, prev_df, on='SK_ID_CURR', how='left')

    pos_df = preprocess_pos_cash(path, num_rows=num_rows)
    print('pos df shape:', pos_df.shape)
    df = pd.merge(df, pos_df, on='SK_ID_CURR', how='left')

    ins_df = preprocess_installment_payments(path, num_rows=num_rows)
    print('installment df shape:', ins_df.shape)
    df = pd.merge(df, ins_df, on='SK_ID_CURR', how='left')

    ccb_df = preprocess_credit_card(path, num_rows=num_rows)
    print('ccb df shape:', ccb_df.shape)
    df = pd.merge(df, ccb_df, on='SK_ID_CURR', how='left')

    print('The shape of total DataFrame:', df.shape)

    # kfold_lgbm(df)


def kfold_lgbm(df):
    # TARGET을 기준으로 df을 각각 train, test 데이터프레임으로 나눔
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]

    # 교차검증을 수행하기 위해 KFold모델 사용
    folds = KFold(n_splits=10, shuffle=True, random_state=55)

    # 결과값을 저장하기 위해 array 생성
    # 교차검증에서 검증세트의 최적 예측값을 저장하기 위한 변수
    oof_preds = np.zeros(train_df.shape[0])

    # submit(제출)할 예측값을 저장하기 위한 변수
    sub_preds = np.zeros(test_df.shape[0])

    # 데이터프레임이 LGBMClassifier 모델에 적용될 수 있도록 데이터프레임의 컬럼이름을 재설정
    train_df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in train_df.columns]
    test_df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in test_df.columns]

    # 모델 성능 평가시 TARGET, SK_ID_CURR 변수를 제외
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR']]

    # 교차검증을 진행하기 위해 KFold모델의 split옵션을 사용하여 train 데이터프레임을 훈련세트와 검증세트로 나눔
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # 모델 생성
        clf = LGBMClassifier(
            objective='binary',
            n_estimators=5000,
            n_jobs=-1,
            silent=-1,
            verbose=-1,
            random_state=55
        )

        # 모델에 데이터 적용 및 평가
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
                eval_metric='auc', verbose=200, early_stopping_rounds=200)

        # 교차검증에서 검증세트의 최적 예측값의 index를 저장
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]

        # 제출할 예측값을 저장
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        # 교차검증에서 검증세트의 최적 예측값의 AUC값을 출력
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))


# 카테고리형 변수들을 수치형으로 변경하는 함수 생성
def cat_encoding(df):
    le = LabelEncoder()
    original_columns = list(df.columns)

    for col in df:

        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
            elif len(list(df[col].unique())) > 2:
                df = pd.get_dummies(df, columns=[col], dummy_na=False)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def preprocess_train_test(path, num_rows=None):
    train_df = pd.read_csv(path + 'application_train.csv', nrows=num_rows)
    test_df = pd.read_csv(path + 'application_test.csv', nrows=num_rows)

    # data cleaning
    # code_gender 컬럼의 XNA(알수없음) 값을 포함한 행을 제외(4개)
    train_df = train_df[train_df['CODE_GENDER'] != 'XNA']
    # name_family_status 컬럼의 UNKNOWN(알수없음) 값을 포함한 행을 제외(2개)
    train_df = train_df[train_df['NAME_FAMILY_STATUS'] != 'UNKNOWN']

    # AMT_INCOME_TOTAL 컬럼의 특이값(이상치)를 제외
    train_df = train_df[train_df['AMT_INCOME_TOTAL'] != 117000000]

    # DAYS_EMPLOYED 컬럼의 이상치를 nan 값으로 변경
    train_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # CNT_CHILDREN(자녀 수)가 특이하게 높은 행을(2개) 제외
    train_df = train_df[train_df['CNT_CHILDREN'] != 19]

    # 차가 있는데 연식이 쓰여있지 않은 행(5개) 제외
    train_df = train_df.drop(train_df[(train_df.FLAG_OWN_CAR == 'Y') & (train_df.OWN_CAR_AGE.isnull())].index)

    # test데이터에서도 마찬가지로 365243을 nan 값으로 변경
    test_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # test 데이터에서도 자녀수가 20인 행에 대해 nan값으로 변경
    test_df['CNT_CHILDREN'].replace(20, np.nan, inplace=True)

    #  train, test 데이터 프레임을 병합
    df = train_df.append(test_df)

    # 카테고리 변수를 수치형 변수로 변환
    df, df_new_columns = cat_encoding(df)

    return df


def preprocess_bureau(path, num_rows=None):
    bb_df = pd.read_csv(path + 'bureau_balance.csv', nrows=num_rows)
    bur_df = pd.read_csv(path + 'bureau.csv', nrows=num_rows)

    # bb 데이터 프레임은 하나의 BUREAU ID에 대해 여러개의 행이 있으므로 이를 그룹화할 필요가 있음.
    # 수치형 피쳐들과 카테고리형 피쳐들을 수치형으로 변환하며 생긴 피쳐들에 대해 agg 기준 설정 및 그룹화.

    # bureau_balance 파일에서 가져온 df의 category변수들을 수치형으로 변경
    bb_df, bb_new_columns = cat_encoding(bb_df)

    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'mean', 'median', 'sum', 'size']}

    for col in bb_new_columns:
        bb_aggregations[col] = ['mean', 'sum']

    bb_agg = bb_df.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    # 컬럼 구분 및 이름 변경
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])

    # bureau.csv 파일에서 가져온 df에 SK_ID_BUREAU기준으로 join
    bur_df = bur_df.join(bb_agg, how='left', on='SK_ID_BUREAU')

    # 더이상 필요하지 않은 SK_ID_BUREAU 컬럼 drop
    bur_df.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    # bur_df 또한 category변수들을 수치형으로 변환
    bur_df, bur_new_columns = cat_encoding(bur_df)

    # 피쳐와 피쳐를 결합하여 새로운 피쳐 생성
    # 실제 낸 날짜 - 예정 날짜
    bur_df['DAYS_INADVANCE'] = bur_df['DAYS_ENDDATE_FACT'] - bur_df['DAYS_CREDIT_ENDDATE']
    # 신용거래 기간(거래완료 예정날짜 - 신청날짜)
    bur_df['CREDIT_DURATION'] = bur_df['DAYS_CREDIT_ENDDATE'] - bur_df['DAYS_CREDIT']
    # 부채 비율
    bur_df['DEBT_PRECENTAGE'] = bur_df['AMT_CREDIT_SUM_DEBT'] / bur_df['AMT_CREDIT_SUM']
    # 거래 총액 - 총 부채액(갚은 돈(?))
    bur_df['AMT_CREDIT_DIFF'] = bur_df['AMT_CREDIT_SUM'] - bur_df['AMT_CREDIT_SUM_DEBT']
    # 월별 갚는 비율
    bur_df['MONTH_CREDIT_REPAY_RATIO'] = bur_df['AMT_ANNUITY'] / bur_df['AMT_CREDIT_SUM']
    # 현재 거래 진행 이후 업데이트 됐을 때까지의 날짜
    bur_df['DAYS_CREDIT_NEW'] = bur_df['DAYS_CREDIT_UPDATE'] - bur_df['DAYS_CREDIT']
    # 총 거래액에서 갚지못한 돈의 최대값 비율
    bur_df['AMT_CREDIT_OVERDUE_RATIO'] = bur_df['AMT_CREDIT_MAX_OVERDUE'] / bur_df['AMT_CREDIT_SUM']

    # 새로운 피쳐들을 생성하는 과정에서 만들어진 inf 값을 가진 행을 제외시킴. (소수이기 때문에 제외)
    bur_df = bur_df[bur_df['AMT_CREDIT_OVERDUE_RATIO'] != np.inf]
    bur_df = bur_df[bur_df['DEBT_PRECENTAGE'] != np.inf]
    bur_df = bur_df[bur_df['DEBT_PRECENTAGE'] != -np.inf]
    bur_df = bur_df[bur_df['MONTH_CREDIT_REPAY_RATIO'] != np.inf]

    # category 변수들로 인해 새롭게 생성된 수치형 변수들에 대해 agg 기준 설정
    cat_aggregations = {}
    values = ['min', 'max', 'mean', 'median', 'sum', 'size']

    # bb_df에서 새롭게 생성된 변수들
    for cat in bb_new_columns: cat_aggregations[cat + "_MEAN"] = ['mean']
    for cat in bb_new_columns: cat_aggregations[cat + "_SUM"] = ['mean', 'sum', 'max', 'min']

    for value in values:
        cat_aggregations["MONTHS_BALANCE_" + value.upper()] = ['mean']

    # bur_df에서 새롭게 생성된 변수들
    for cat in bur_new_columns: cat_aggregations[cat] = ['mean', 'sum']

    # 수치형 변수들에 대한 agg 기준 설정
    # category형 변수들의 컬럼이름을 리스트로 저장
    categories = []
    for key, value in cat_aggregations.items():
        categories.append(key)

    num_aggregations = {}

    # category형 변수들의 컬럼이름을 제외한 컬럼들에 대해 agg 기준 설정
    num_columns = [ _ for _ in bur_df.columns if _ not in categories]
    num_columns.remove('SK_ID_CURR')
    for col in num_columns:
        num_aggregations[col] = ['min', 'max', 'mean', 'median', 'sum', 'size']

    # 위에서 생성한 agg 설정에 대해 SK_ID_CURR을 기준으로 groupby
    bureau_agg = bur_df.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    # 변수명 정리
    bureau_agg.columns = pd.Index(['BUR_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # bur 데이터프레임에서 CREDIT_ACTIVE 피쳐가 각각 Active, closed인 행들만을 추출하는 새로운 데이터프레임을 생성하고,
    # 수치형 agg 기준으로만 그룹화 하여 기존의 bur 데이터프레임에 join 시켜 새로운 피쳐들을 생성.

    # bur_df에서 CREDIT_ACTIVE가 각각 Active, closed인 행들만을 추출한 새로운 df들 생성
    active = bur_df[bur_df['CREDIT_ACTIVE_Active'] == 1]
    closed = bur_df[bur_df['CREDIT_ACTIVE_Closed'] == 1]

    # 수치형 agg 설정값에 대해 SK_ID_CURR을 기준으로 groupby
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)

    # 변수명 정리
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])

    # bureau_agg에 join
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del bb_df; gc.collect()
    return bureau_agg


def preprocess_previous_applications(path, num_rows=None):
    prev_df = pd.read_csv(path + 'previous_application.csv', nrows=num_rows)

    # Data Cleaning
    # 소수의 결측값을 갖는 피쳐에 대해 해당 결측값을 포함하는 행을 제외시킴
    prev_df = prev_df.dropna(subset=['PRODUCT_COMBINATION'])
    prev_df = prev_df.dropna(subset=['AMT_CREDIT'])

    # 피쳐의 특징에 따라 결측값을 채워줌
    prev_df['NAME_TYPE_SUITE'].fillna("XNA", inplace=True)
    prev_df['NFLAG_INSURED_ON_APPROVAL'].fillna(0, inplace=True)
    prev_df['AMT_DOWN_PAYMENT'].fillna(0, inplace=True)
    prev_df['AMT_ANNUITY'].fillna(0, inplace=True)
    prev_df['AMT_GOODS_PRICE'].fillna(0, inplace=True)

    # 소수의 이상치를 갖는 피쳐에 대해 해당 이상치를 갖는 행을 제외시킴
    prev_df = prev_df.drop([191373, 251875, 287370, 627069, 648826, 772107, 1346611, 1521472])

    # 다수의 이상치를 갖는 피쳐에 대해 이상치를 갖는 행의 이상치를 결측치로 바꿈
    prev_df['DAYS_FIRST_DRAWING'].replace({365243: np.nan}, inplace=True)
    prev_df['DAYS_FIRST_DUE'].replace({365243: np.nan}, inplace=True)
    prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace({365243: np.nan}, inplace=True)
    prev_df['DAYS_LAST_DUE'].replace({365243: np.nan}, inplace=True)
    prev_df['DAYS_TERMINATION'].replace({365243: np.nan}, inplace=True)

    # Create New Features
    prev_df['DAYS_FIRST_OVERDUE'] = prev_df['DAYS_FIRST_DUE'] - prev_df['DAYS_FIRST_DRAWING']

    prev_df['DAYS_PAYMENT_PERIOD'] = prev_df['DAYS_LAST_DUE'] - prev_df['DAYS_FIRST_DUE']

    prev_df['DAYS_EXPECT_LAST_DUE_GAP'] = prev_df['DAYS_TERMINATION'] - prev_df['DAYS_LAST_DUE']

    prev_df['AMT_CREDIT_DIFF'] = prev_df['AMT_APPLICATION'] - prev_df['AMT_CREDIT']
    prev_df['AMT_CREDIT_RATIO'] = prev_df['AMT_CREDIT'] / prev_df['AMT_APPLICATION']

    prev_df['AMT_DOWN_PAYMENT_DIFF'] = prev_df['AMT_APPLICATION'] - prev_df['AMT_DOWN_PAYMENT']
    prev_df['AMT_DOWN_PAYMENT_RATIO'] = prev_df['AMT_DOWN_PAYMENT'] / prev_df['AMT_APPLICATION']

    # 새로운 피쳐를 생성할 때 생긴 inf값 처리
    prev_df['AMT_CREDIT_RATIO'].replace(to_replace=np.inf, value=0, inplace=True)
    prev_df['AMT_CREDIT_RATIO'].replace(to_replace=-np.inf, value=0, inplace=True)

    prev_df, prev_new_columns = cat_encoding(prev_df)

    # NAME_CONTRACT_STATUS 컬럼이 approved와 refused인 행들만을 포함하는 새로운 데이터 프레임을 생성
    approved = prev_df[prev_df['NAME_CONTRACT_STATUS_Approved'] == 1]
    refused = prev_df[prev_df['NAME_CONTRACT_STATUS_Refused'] == 1]

    # 카테고리형 피쳐들과 수치형 피쳐들에 대해 각각 groupby시 다른 agg 조건 설정

    # 기존의 카테고리형 피쳐들에 대한 agg 조건 설정
    cat_aggregations = {}

    for col in prev_new_columns:
        cat_aggregations[col] = ['mean', 'sum']

    # 기존에 있던 수치형 피쳐들을 리스트로 저장
    prev_columns = [_ for _ in prev_df.columns if _ not in prev_new_columns]

    # SK_ID_PREV와 SK_ID_CURR 피쳐를 agg 리스트에서 제외
    prev_columns.remove('SK_ID_PREV')
    prev_columns.remove('SK_ID_CURR')


    # 수치형 피쳐들에 대한 agg 조건 설정
    num_aggregations = {}

    for col in prev_columns:
        num_aggregations[col] = ['min', 'max', 'mean', 'median', 'sum', 'size']

    # 위에서 설정한 조건에 따라 SK_ID_PREV를 기준으로 그룹화
    prev_agg = prev_df.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg({**cat_aggregations, **num_aggregations})

    # 컬럼 구분 및 이름 변경
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # SK_ID_PREV를 기준으로 그룹화한 데이터프레임을 SK_ID_CURR, agg 조건은 mean으로 다시 한 번 그룹화
    prev_agg = prev_agg.groupby(['SK_ID_CURR']).agg('mean')

    # approved와 refused로 생성된 데이터 프레임에 대해서도 수치형 피쳐들에 대해서만 agg기준 적용 및 피쳐 구분
    # 이후 기존의 prev 데이터프레임에 join

    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    return prev_agg


def preprocess_pos_cash(path, num_rows=None):
    pos_cash = pd.read_csv(path + 'POS_CASH_balance.csv', nrows=num_rows)

    # create new features
    pos_cash['DPD_MIN_DPDDEF'] = pos_cash['SK_DPD'] - pos_cash['SK_DPD_DEF']
    pos_cash['MONTHS_PLUS_CNT'] = pos_cash['MONTHS_BALANCE'] + pos_cash['CNT_INSTALMENT']
    pos_cash['MONTHS_PLUS_CNTFUT'] = pos_cash['MONTHS_BALANCE'] + pos_cash['CNT_INSTALMENT_FUTURE']

    pos_cash_df, pos_new_columns = cat_encoding(pos_cash)

    # aggregations
    # 카테고리형 피쳐들을 수치형 피쳐들로 변환하는 과정에서 생성된 피쳐들에 대한 agg 조건 설정
    pos_aggregations = {}

    for col in pos_new_columns:
        pos_aggregations[col] = ['mean']
    # 기존 수치형 피쳐들만을 포함하는 리스트 생성
    pos_columns = [c for c in pos_cash_df.columns if c not in pos_new_columns]
    pos_columns.remove('SK_ID_PREV')
    pos_columns.remove('SK_ID_CURR')

    # 기존 수치형 피쳐들에 대한 agg 조건 설정
    for col in pos_columns:
        pos_aggregations[col] = ['min', 'max', 'mean', 'median', 'sum', 'size']

    pos_cash1 = pos_cash_df.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(pos_aggregations)

    # sk_id_prev, months_balance를 기준으로 정렬
    pos_cash2 = pos_cash.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'])
    pos_cash2 = pos_cash2.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg({'MONTHS_BALANCE': ['max'], 'CNT_INSTALMENT': ['last']})
    pos_cash2 = pos_cash2.drop('MONTHS_BALANCE', axis=1)

    # 위에서 구한 최종 할부 개월수를 pos_cash 데이터 프레임에 병합
    pos_agg = pos_cash1.merge(pos_cash2, on=['SK_ID_PREV', 'SK_ID_CURR'])

    # 컬럼 구분 및 이름 변경
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # SK_ID_PREV를 기준으로 그룹화한 데이터프레임을 SK_ID_CURR, agg 조건은 mean으로 다시 한 번 그룹화
    pos_agg = pos_agg.groupby(['SK_ID_CURR']).agg('mean')

    # 소수인 결측값을 포함하는 행을 제외
    pos_agg = pos_agg.dropna(axis=0)

    return pos_agg


def preprocess_installment_payments(path, num_rows=None):
    ins = pd.read_csv(path + 'installments_payments.csv', nrows=num_rows)

    # Add feature
    ins['DAYS_INADVANCE'] = (ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT'])
    ins['AMT_GAP'] = (ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT'])

    # agregation
    aggregation = {'NUM_INSTALMENT_NUMBER': ['max', 'mean'], 'DAYS_INADVANCE': ['mean'],
                   'NUM_INSTALMENT_VERSION': ['max', 'min'],
                   'AMT_INSTALMENT': ['mean'], 'AMT_PAYMENT': ['mean'], 'AMT_GAP': ['max', 'min', 'mean']}

    ins = ins.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(aggregation)

    # 컬럼 구분 및 이름 변경
    ins.columns = pd.Index([e[0] + "_" + e[1].upper() for e in ins.columns.tolist()])

    # SK_ID_CURR를 기준으로 aggregation
    ins = ins.groupby('SK_ID_CURR').agg(['mean'])
    # 컬럼 구분 및 이름 변경
    ins.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins.columns.tolist()])

    return ins


def preprocess_credit_card(path, num_rows=None):
    ccb = pd.read_csv(path + 'credit_card_balance.csv', nrows=num_rows)

    # Add features
    ccb['AMT_BALANCE_RATIO'] = ccb['AMT_BALANCE'] / ccb['AMT_CREDIT_LIMIT_ACTUAL']

    ccb['ONCE_DRAWINGS_ATM_CURRENT'] = ccb['AMT_DRAWINGS_ATM_CURRENT'] / ccb['CNT_DRAWINGS_ATM_CURRENT']
    ccb['ONCE_DRAWINGS_CURRENT'] = ccb['AMT_DRAWINGS_CURRENT'] / ccb['CNT_DRAWINGS_CURRENT']
    ccb['ONCE_DRAWINGS_OTHER_CURRENT'] = ccb['AMT_DRAWINGS_OTHER_CURRENT'] / ccb['CNT_DRAWINGS_OTHER_CURRENT']
    ccb['ONCE_DRAWINGS_POS_CURRENT'] = ccb['AMT_DRAWINGS_POS_CURRENT'] / ccb['CNT_DRAWINGS_POS_CURRENT']

    ccb['AMT_DRAWINGS_ATM_CURRENT_RATIO'] = ccb['AMT_DRAWINGS_ATM_CURRENT'] / ccb['AMT_DRAWINGS_CURRENT']
    ccb['AMT_DRAWINGS_OTHER_CURRENT_RATIO'] = ccb['AMT_DRAWINGS_OTHER_CURRENT'] / ccb['AMT_DRAWINGS_CURRENT']
    ccb['AMT_DRAWINGS_POS_CURRENT_RATIO'] = ccb['AMT_DRAWINGS_POS_CURRENT'] / ccb['AMT_DRAWINGS_CURRENT']

    ccb['CNT_DRAWINGS_ATM_CURRENT_RATIO'] = ccb['CNT_DRAWINGS_ATM_CURRENT'] / ccb['CNT_DRAWINGS_CURRENT']
    ccb['CNT_DRAWINGS_OTHER_CURRENT_RATIO'] = ccb['CNT_DRAWINGS_OTHER_CURRENT'] / ccb['CNT_DRAWINGS_CURRENT']
    ccb['CNT_DRAWINGS_POS_CURRENT_RATIO'] = ccb['CNT_DRAWINGS_POS_CURRENT'] / ccb['CNT_DRAWINGS_CURRENT']

    ccb['AMT_RECIVABLE_DIFF'] = ccb['AMT_RECIVABLE'] - ccb['AMT_TOTAL_RECEIVABLE']
    ccb['SK_DPD_LOW_LOAN'] = ccb['SK_DPD'] - ccb['SK_DPD_DEF']

    # category encoding
    ccb, ccb_new_columns = cat_encoding(ccb)

    # aggregation
    ccb_aggregations = {}

    for col in ccb_new_columns:
        ccb_aggregations[col] = ['mean']

    ccb_columns = [c for c in ccb.columns if c not in ccb_new_columns]
    ccb_columns.remove('SK_ID_PREV')
    ccb_columns.remove('SK_ID_CURR')

    for col in ccb_columns:
        ccb_aggregations[col] = ['min', 'max', 'mean', 'median', 'sum', 'size']

    ccb = ccb.groupby(['SK_ID_PREV', 'SK_ID_CURR']).agg(ccb_aggregations)
    ccb.columns = pd.Index([e[0] + "_" + e[1].upper() for e in ccb.columns.tolist()])

    ccb = ccb.groupby(['SK_ID_CURR']).agg('mean')

    # modified columns name
    ccb.columns = pd.Index(['CCB_' + e[0] + "_" + e[1].upper() for e in ccb.columns.tolist()])

    return ccb

path = 'data/'

if __name__ == '__main__':
    main(debug=False)
