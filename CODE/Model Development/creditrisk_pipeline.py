import numpy as np
import pandas as pd
import pickle as pkl
import yaml
import shap
import os
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, PowerTransformer, RobustScaler, PolynomialFeatures
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from feature_transformer import SimpleImputer, WOEEncoder, SeasonalityExtractor, CategoryMerger, FicoCombiner, StringFormatter, OutlierExtractor, Debug
from sklearn.feature_selection import SelectFromModel,  RFE
from sklearn.svm import LinearSVC
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from sklearn.pipeline import Pipeline
from focal_loss import BinaryFocalLoss
from scipy.stats import pearsonr
    

class CreditRiskModel:
    
    def __init__(self, model_config_path):
        
        self.config = yaml.safe_load(open(model_config_path))
        self.catboost_params = self.config['catboost']['bayesian_search_tune2']
        self.estimator = self.explanation_model = CatBoostClassifier(**self.catboost_params)

        self.model = StackingClassifier(
            cv=5, 
            stack_method = 'auto',
            estimators=[
                ('catboost1', self.estimator),
                ('catboost2', self.estimator),
                ('catboost3', self.estimator),
                ('catboost4', self.estimator),
                ('catboost5', self.estimator)
            ],
            final_estimator=LogisticRegression(C=1.0, class_weight=None,
                                              dual=False,
                                              fit_intercept=True,
                                              intercept_scaling=1,
                                              l1_ratio=None,
                                              max_iter=1000,
                                              multi_class='auto',
                                              n_jobs=None, penalty='l2',
                                              random_state=4718,
                                              solver='lbfgs',
                                              tol=0.0001, verbose=0,
                                              warm_start=False),
            n_jobs=-1, 
            passthrough=True, 
            verbose=0                               
        )
        
        self.focal_loss = BinaryFocalLoss(gamma=3)
        self.final_model = GridSearchCV(self.model, {}, cv=5, scoring=self.focal_loss, n_jobs=-1, refit='roc_auc')
        

    def fit(self, X_train, y_train, numeric_features = None, categorical_features = None, high_cardinal_categorical_features = None, date_features = None):

        if numeric_features == None:
            numeric_features = self.config['input_features']['numeric'] 
        if categorical_features == None:
            categorical_features =  self.config['input_features']['categorical'] 
        if high_cardinal_categorical_features == None:
            high_cardinal_categorical_features = self.config['input_features']['high_categorical'] 
        if date_features == None:
            date_features = self.config['input_features']['date'] 
                
        numeric_transformer = Pipeline(
            steps = [
                ('Average Fico', FicoCombiner(weights = [0.5, 0.5])),
                ('Format Strings', StringFormatter()),
                ("Mean Impute", SimpleImputer(strategy="mean")), 
                # ("Standardize", StandardScaler())
                # ("Robust Scaler", RobustScaler())
            ]
        )
        categorical_transformer = Pipeline(
            steps = [
                ('Merge Categories', CategoryMerger()),
                ("Mode Impute", SimpleImputer(strategy="most_frequent")),
                ("Dummy Encode", OneHotEncoder(handle_unknown='ignore', drop = 'first')),
                # ("Robust Scaler", RobustScaler()),
            ]
        )
        high_categorical_transformer = Pipeline(
            steps = [
                ("Mode Imputing", SimpleImputer(strategy="most_frequent")),
                ("Weight-of-Evidence Encode", WOEEncoder(handle_unknown='ignore'))
            ]
        )
        datetime_transformer = Pipeline(
            steps = [
                ("Extract Seasonality", SeasonalityExtractor(expand_date_column = "last_credit_pull_d", time_features = ["day"], expand_days = False)),
                ("Mode Impute", SimpleImputer(strategy="most_frequent")),
            ]     
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("Num", numeric_transformer, numeric_features),
                ("Cat", categorical_transformer, categorical_features),
                ("Hi_Cat", high_categorical_transformer, high_cardinal_categorical_features),
                ("Date", datetime_transformer, date_features)
            ],
            remainder='drop'
        )
        steps = [
            ('preprocessor', preprocessor),
            # ("postprocess_debug", Debug()),
            ("Robust Scaler", RobustScaler()),
            ('power_transformer', PowerTransformer()),
            ('oversampler', ADASYN(sampling_strategy = 0.8)), 
            ('undersampler', RandomUnderSampler(sampling_strategy = 0.9999)),       
            ('feature_selection', SelectFromModel(LinearSVC(C=5, penalty="l1", dual = False))), ## the smaller C the fewer features selected. Don't select feature.
            # ("pre_model_fitting_debug", Debug()),
            # ('feature_selection', SelectFromModel(RandomForestClassifier())), 
            # ('model', self.estimator)
            ('model', self.estimator)
        ]
        pipe = imbpipeline(steps=steps).fit(X_train, y_train)
        return pipe        
        
    def load_data_from_csv(self, earliest_issue_date = '2015-01-01'): 
        data = pd.read_csv(self.config["lc_data_path"])
        data = data[data['term'] == ' 36 months']
        data['issue_d'] = pd.to_datetime(data['issue_d'])
        data = data.sort_values(by = 'issue_d')
        data = data[data['issue_d'] > earliest_issue_date]
        data = self.set_default(data)
        return data

    def set_default(self, data):   
        ''' Remove "Current". Get only fully paid and charged off data '''
        data = data.loc[data['loan_status'].isin(['Fully Paid', 'Charged Off', \
                                                  'Late (31–120 days)', 'Default', \
                                                  'Does not meet the credit policy. Status:Charged Off'])]
        data['loan_status'] = data['loan_status'].replace(to_replace = ['Late (31-120 days)', \
                                        'Default', 'Does not meet the credit policy. Status:Charged Off'], value = 'Charged Off')
        data = data.loc[data['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        data['default'] = (data['loan_status'] == 'Charged Off').apply(np.uint8)
        print(data['loan_status'].value_counts(dropna=False))
        return data
    
    def split_data(self, data, test_size = 0.1, use_custom_set = False, sample_train = True):
        label = self.config['label_col_name']
        if not use_custom_set:
            data = data.sample(frac = 1)
            X, y = data.drop(columns=label), data[label]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=23)
        else:
            test_id = np.load(self.config["benchmark_testset_id"])
            test_idx = data["id"].isin(test_id)
            X, y = data.drop(columns=label), data[label]
            test, train = data[test_idx], data[~test_idx]
            train.drop('id', axis = 1, inplace = True)
            test.drop('id', axis = 1, inplace = True)
            X_train, y_train = train.drop(label, axis = 1), train[label]
            X_test, y_test = test.drop(label, axis = 1), test[label]
        if sample_train:
            X_train, y_train = X_train.iloc[-120000:], y_train.iloc[-120000:]
        return X_train, X_test, y_train, y_test, X, y


#     def drop_lookahead_variables(self, data):
#         '''
#         Do not call prior to training, since columns already dropped in ML pipeline
#         '''        
#         unknown_prior_to_funding = ['funded_amnt', 'funded_amnt_inv', 'url', "recoveries", "collection_recovery_fee",
#                        "otal_rec_late_fee", "total_rec_int","total_rec_prncp", "out_prncp", "out_prncp_inv", "total_pymnt",
#                        "total_pymnt_inv", "last_pymnt_amnt","last_pymnt_d", "total_rec_late_fee",
#                        "pymnt_plan", "initial_list_status", "policy_code", 'chargeoff_within_12_mths', 
#                        'debt_settlement_flag', "hardship_flag", 'nb_of_payments', "next_pymnt_d", 
#                        "verification_status", 'verification_status_joint', 'issue_d']
#         graded = ['loan_amnt', 'term', 'int_rate', 'installment', 'sub_grade', 'grade']
#         misc = ["zip_code", 'emp_title', 'title', 'purpose']
        
#         drop_list = list(set(data.columns) & set(unknown_prior_to_funding + graded + misc))
#         print("Drop Features (" + str(len(drop_list)) + ") : ", drop_list, "\n\n")
#         return data.drop(labels=drop_list, axis=1)

#     def clean_columns(data):
#         '''
#         Do not call prior to training, clean procedure already embedded in in ML pipeline column transformer
#         '''
#         # FICO reports high/low range. Combine by average
#         data['ficoscore'] = 0.5*data['fico_range_low'] + 0.5*data['fico_range_high']
#         data.drop(['fico_range_high', 'fico_range_low'], axis=1, inplace=True)
#         # These are lookahead variables, else redundant.
#         data.drop(['last_fico_range_high', 'last_fico_range_low'], axis=1, inplace=True)
#         # Clean percentage scores
#         data['revol_util'] = data['revol_util'].map(lambda x: str(x).replace('%','')).astype(np.float64)
#         # Clean string provided by emp_length
#         data['emp_length'].replace(to_replace='10+ years', value='10 years', inplace=True)
#         data['emp_length'].replace('< 1 year', '0 years', inplace=True)
#         data['emp_length'] = data['emp_length'].apply(lambda s: s if pd.isnull(s) else np.int8(s.split()[0]))
#         # temporary: leave morgage account value as total acc if morgage account is non-negative
#         total_acc_avg = data.groupby('total_acc').mean()['mort_acc'].fillna(0)
#         data['mort_acc'] = data.apply(lambda x: total_acc_avg[x['total_acc']] if x['mort_acc'] >=0 else x['mort_acc'], axis=1)
#         # Home_ownership - non-informative values - collapse into OTHER
#         data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER', inplace=True)
#         data['last_credit_pull_d'] = pd.to_datetime(data['last_credit_pull_d'])
#         data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'])
#         # Utilize total length of credit history: provide interaction to account for upward mobility 
#         data['time_since_earliest_cr_line'] = (data['last_credit_pull_d'] - data['earliest_cr_line']).dt.days
#         data.drop(['earliest_cr_line'], axis=1, inplace=True)
        
#         return data

    def interest_rate_vs_score(self, pipe, df):
        
        X_test = df.copy()
        X_test['score'] = pipe.predict_proba(X_test)[:, 1]
        # def NormalizeData(data):
        #     return (data - np.min(data)) / (np.max(data) - np.min(data))
        X_test['int_rate'] = X_test['int_rate'].apply(lambda x:float(x.replace("%","").replace(" ","")))
        # X_test['int_rate'] = NormalizeData(X_test['int_rate'])
        plt.style.use('ggplot')
        print('distinct interest rates')
        np.sort(X_test['int_rate'].unique())
        X_test['score_binned'] = pd.cut(X_test['score'],50)
        average_interest_per_score_bin =X_test.groupby('score_binned').agg({'int_rate':['mean', 'size']}).apply(lambda x:x).reset_index(level=0)
        fig = plt.figure(figsize = (30,10))
        ax = fig.add_subplot(111)
        ax.set_title(f"PD ranges vs. Interest Rates (with sample counts); Pearson r: {round(pearsonr(X_test['score'], X_test['int_rate'])[0],2)}", fontsize=18)
        dots = plt.scatter(range(50),average_interest_per_score_bin['int_rate']['mean'], color ='b', s = 20)
        for i in range(50):
            ax.annotate('%s' %average_interest_per_score_bin['int_rate']['size'][i], xy=(i,average_interest_per_score_bin['int_rate']['mean'][i]), textcoords='data', fontsize = 12)
        print('Correlation Coefficient:', pearsonr(X_test['score'], X_test['int_rate']))
        ax.set_xticks(ticks = range(50), labels =average_interest_per_score_bin['score_binned'], rotation = 45, weight='bold')
        ax.set_xlabel('Predicted PD ranges',fontsize=12, weight='bold')
        ax.set_ylabel('Interest Rate', fontsize=12,weight='bold')
        ax.legend()



    def high_missing_features(self, data, miss_rate_tolerance = 0.5, drop = False):
        missing_fractions = data.isnull().mean().sort_values(ascending=False)
        missing_fractions.head()
        plt.close()
        plt.figure(figsize=(6,4), dpi=90)
        missing_fractions.plot.hist(bins=100)
        plt.title('Histogram of Feature Incompleteness')
        plt.xlabel('Fraction of data missing')
        plt.ylabel('Feature count')
        plt.axvline(x = miss_rate_tolerance, color = "r", linestyle = "--", label = "remove past this line")
        plt.legend()
        plt.grid()
        drop_list = sorted(list(missing_fractions[missing_fractions > miss_rate_tolerance].index)) ##### experiment with 0.25
        print("Drop Features (" + str(len(drop_list)) + ") : ", drop_list, "\n\n")
        if drop:
            data.drop(labels=drop_list, axis=1, inplace=True)
        return drop_list 


    def save_data(self, X, y, data_path = "../data/", is_train_data = True):    
        if is_train_data:
            X.to_pickle(data_path + "train/X_train.pkl")
            y.to_pickle(data_path + "train/y_train.pkl")
        else:
            X.to_pickle(data_path + "test/X_test.pkl")
            y.to_pickle(data_path + "test/y_test.pkl")

                
    def load_data(self, data_path = "../data/", is_train_data = True):
        if is_train_data:
            return pd.read_pickle(data_path + "train/X_train.pkl"), \
                    pd.read_pickle(data_path + "train/y_train.pkl")
        else:
            return pd.read_pickle(data_path + "test/X_test.pkl"), \
                    pd.read_pickle(data_path + "test/y_test.pkl")
            
    
    def load_pipeline(self, pipeline_path=None):
        if pipeline_path == None: 
            pipeline_path = self.config["pipeline_path"] 
        if os.path.exists(pipeline_path):
            return pkl.load(open(pipeline_path, 'rb'))
        else:
            return None
       
    def save_pipeline(self, pipe, pipeline_path= None):
        if pipeline_path is None:
            pipeline_path = self.config["pipeline_path"] 
        pkl.dump(pipe,open(pipeline_path,"wb"))
 
    def draw_pipeline(self, pipe):
        from sklearn import set_config
        set_config(display="diagram")
        return pipe
    
    def plot_learning_curve(self):
        pass
    
    def evaluate_pipeline(self, pipe, X_test, y_test, metric = 'auc'):
        if metric == 'auc':
            y_test_score = pipe.predict_proba(X_test)[:, 1]
            res = roc_auc_score(y_test, y_test_score)
            print('ROC AUC: %.3f' % res)
        if metric == 'accuracy':
            y_predicted = pipe.predict(X_test)
            res = accuracy_score(y_test, y_predicted)
            print('Accuracy: %.3f' % res)            
        return res
    
    def score_pipeline(self, pipe, X_test):  # get scores 
        return pipe.predict_proba(X_test)[:, 1]
    
    def expected_value(self, pipe, X_test):
        scores = pipe.predict_proba(X_test)[:, 1]
        debt = X_test["Pentius Debt"].values
        return scores*debt
    
    def get_bucket_cutoffs(self, pipe, X_test, num_buckets = 5): 

        assert num_buckets <= len(string.ascii_uppercase), "num_buckets <= " + str(len(string.ascii_uppercase))
        q = np.cumsum((num_buckets)*[1/(num_buckets)])[:-1]
        scores = pipe.predict_proba(X_test)[:, 1]
        return np.quantile(scores, q)   # get q-th quantile
            
    def draw_histogram(self, pipe, X_test):
            
        scores = pipe.predict_proba(X_test)[:, 1]
        plt.close()
        plt.figure(0)
        plt.title("probability histogram")
        plt.hist(scores, bins=200, color = "k", alpha = 0.5)
        plt.xlabel("Probability of Default")
        plt.axvline(x=np.quantile(scores, 0.1), label = f'{sum(scores < np.quantile(scores, 0.1))} obs. < 0.1 quantile', linestyle = '--', color = 'r')
        plt.grid()
        plt.legend()
        print('Total # of predictions:', len(scores))
        print('0.1 Quantile Probability:', np.quantile(scores, 0.1).round(5))

    def grade_scores(self, scores, buckets):
        
        assert isinstance(buckets, (np.ndarray, np.generic)), "Grade buckets need to be a numpy array"
        
        bins = list(buckets) + [1]
        grades = np.digitize(scores, bins = bins, right = True)
        ZYX = np.flip(list(string.ascii_uppercase)[:len(bins)])
        grade_dict = dict(enumerate(ZYX))
        return np.vectorize(grade_dict.get)(grades)

    def plot_feature_importances(self, pipe, X_train, y_train, type = "gain", topk=13):

        myfeatures = pipe[:-1].get_feature_names_out()
        # preprocessor & power transformer
        X_train = pipe[:3].transform(X_train)
        # oversampler
        resampled_X, resampled_y = pipe[3].fit_resample(X_train, y_train)
        # undersampler
        resampled_X, resampled_y = pipe[4].fit_resample(resampled_X, resampled_y)
        # feature selector
        feature_selected_X = pipe[5].transform(resampled_X)
        X_train, y_train = feature_selected_X, resampled_y
        plt.close()
        
        if type == "weight":
            model = xgb.XGBClassifier(random_state=0)
            model.fit(X_train, y_train)
            dict_features = dict(enumerate(myfeatures))
            axsub = plot_importance(model, importance_type=type, max_num_features=topk)
            Text_yticklabels = list(axsub.get_yticklabels())
            lst_yticklabels = [Text_yticklabels[i].get_text().lstrip('f') for i in range(len(Text_yticklabels))]
            lst_yticklabels = [dict_features[int(i)] for i in lst_yticklabels]
            axsub.set_yticklabels(lst_yticklabels)
            # print(dict_features)
            plt.figure(figsize=(20,15))
            plt.show()
            
        elif type == "gain":
            model = self.explanation_model
            model.fit(X_train, y_train)
            fis = model.feature_importances_ # Gain percentage
            descending_importances = np.sort(fis)[::-1] # obtain feature importance from high to low
            descending_idx = fis.argsort()[::-1] # obtain INDEX of feature importance from high to low
            feature_names_in_descending_importance = myfeatures[descending_idx] # obtain feature names from importance high to low
            plt.figure(figsize=(13,5))
            df = pd.DataFrame(zip([feature_names_in_descending_importance[n] for n in range(topk)], descending_importances[:topk]), columns=['Features', 'Feature Importance'])
            df = df.set_index('Features').sort_values('Feature Importance')
            ax = df.plot.barh(color='red', alpha=0.5, grid=True, legend=False, title='Feature importance', figsize=(15, 5))
            # Annotate bar chart, adapted from this SO answer: https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots
            for p, value in zip(ax.patches, df['Feature Importance']):
                ax.annotate(round(value, 5), (p.get_width() * 1.005, p.get_y() * 1.005))
                
    def shap_explanation(self, pipe, X_train, y_train, shap_sample_size = 300):

            myfeatures = ['Num__acc_open_past_24mths', 'Num__all_util', 'Num__annual_inc',
                'Annual Income (Joint)', 'Num__avg_cur_bal', 'Num__bc_open_to_buy',
                'Num__bc_util', 'Num__collections_12_mths_ex_med',
                'Num__delinq_2yrs', 'Num__delinq_amnt', 'Num__dti',
                'Num__dti_joint', 'Employment Length', 'Num__il_util', '# of personal finance inquiries',
                '# of inquiries in past 12 months', '# of inquiries in past 6 months',
                'Num__last_credit_pull_d', 'Num__mo_sin_old_il_acct',
                'Num__mo_sin_old_rev_tl_op', 'Num__mo_sin_rcnt_rev_tl_op',
                'Num__mo_sin_rcnt_tl', '# of mortgage accounts', 'Num__mths_since_rcnt_il',
                'Num__mths_since_recent_bc', 'Num__mths_since_recent_inq',
                'Num__num_accts_ever_120_pd', '# of active bankcard tradelines',
                'Num__num_actv_rev_tl', 'Num__num_bc_sats', 'Num__num_bc_tl',
                'Num__num_op_rev_tl', 'Num__num_rev_accts',
                'Num__num_rev_tl_bal_gt_0', 'Num__num_sats',
                'Num__num_tl_90g_dpd_24m', 'Num__num_tl_op_past_12m',
                'Num__open_acc', 'Num__open_acc_6m', '# of active installment accounts',
                '# of installments opened in 12mo', '# of installments opened in 24mo', 'Num__open_rv_12m',
                'Num__open_rv_24m', 'Num__pct_tl_nvr_dlq', 'Num__percent_bc_gt_75',
                'Num__pub_rec_bankruptcies', 'Num__revol_bal',
                'Num__revol_bal_joint', 'Num__revol_util', 'Num__tot_coll_amt',
                'Num__tot_cur_bal', 'Num__tot_hi_cred_lim', 'Num__total_acc',
                'Num__total_bal_ex_mort', 'Num__total_bal_il',
                'Num__total_bc_limit', '# of finance trades',
                'Num__total_il_high_credit_limit', 'Num__total_rev_hi_lim',
                'FICO', 'Num__time_since_earliest_cr_line',
                'Cat__application_type_Joint App', 'Cat__home_ownership_OTHER',
                'Cat__home_ownership_OWN', 'homeownership = Rent',
                'Hi_Cat__addr_state']
            
            # preprocessor & power transformer
            X_train = pipe[:3].transform(X_train)
            # oversampler
            resampled_X, resampled_y = pipe[3].fit_resample(X_train, y_train)
            # undersampler
            resampled_X, resampled_y = pipe[4].fit_resample(resampled_X, resampled_y)
            # feature selector
            feature_selected_X = pipe[5].transform(resampled_X)
            X_train, y_train = feature_selected_X, resampled_y
            X_train = pd.DataFrame(X_train, columns = myfeatures)
            self.explanation_model.fit(X_train, y_train)
            
            shap.initjs()
            explainer = shap.TreeExplainer(self.explanation_model)
            X_train = X_train.sample(n = shap_sample_size)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, feature_names = myfeatures, plot_type="dot", max_display = 7)
            # shap.summary_plot(shap_values, X_train, feature_names = myfeatures, plot_type="violin")
            shap.summary_plot(shap_values, X_train, feature_names = myfeatures, plot_type="bar", max_display = 7)
            # shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")

            