import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA    
from sklearn.metrics import silhouette_score, silhouette_samples, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV,LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.externals import joblib


#Function to generate a csv with parameterised correlation range

def cor_csv(filename, lower_cor, upper_cor ):
    df_data = pd.read_csv(filename, encoding='latin1')
    numerical_df = df_data.select_dtypes(exclude=['object'])
    corr_matrix = numerical_df.corr()
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
    sol_df = pd.DataFrame(data=sol)
    f_df = sol_df.reset_index().rename(columns = {'level_0':'Variable_1', 'level_1':'Variable_2', 0:'Correlation' })
    f_df[ (f_df['Correlation']>=upper_cor) | (f_df['Correlation']<= lower_cor)].to_csv('../Data/Cor_df.csv')


#Function to generate a csv with parameterised correlation range

def cor_mrg_csv(filename1,filename2, lower_cor, upper_cor, j_type, j_col ):
    df_data1 = pd.read_csv(filename1, encoding='latin1')
    df_data2 = pd.read_csv(filename2, encoding='latin1')
    
    df_data = pd.merge(df_data1, df_data2, how=j_type, on=j_col) 
    
    
    #numerical_df = df_data.select_dtypes(exclude=['object'])
    corr_matrix = df_data.corr()
    
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))
    sol_df = pd.DataFrame(data=sol)
    f_df = sol_df.reset_index().rename(columns = {'level_0':'Variable_1', 'level_1':'Variable_2', 0:'Correlation' })
    f_df[ (f_df['Correlation']>=upper_cor) | (f_df['Correlation']<= lower_cor)].to_csv('../Data/Cor_mrg_df.csv')

# Function to generate dist plots

def dist_plot(filename, col_list):
    df_data = pd.read_csv(filename1, encoding='latin1')    
    numerical_df = df_data.select_dtypes(exclude=['object'])

    for i in col_list:
        fig, axes = plt.subplots(1,1, figsize=(10,6))
        dis_plot = sns.distplot(numerical_df[i])   
        d_plot = dis_plot.get_figure()
        pltpath = '../Output/Distplot/'
        d_plot.savefig(pltpath+i+'.png')
        
# Function to generate box plots

def box_plot(filename, col_list):
    df_data = pd.read_csv(filename1, encoding='latin1')    
    numerical_df = df_data.select_dtypes(exclude=['object'])
    #col_list = numerical_df.columns 
    for i in col_list:
        fig, axes = plt.subplots(1,1, figsize=(10,6))
        sns_plot = sns.boxplot(numerical_df[i])   
        s_plot = sns_plot.get_figure()
        pltpath = '../Output/Boxplot/'
        s_plot.savefig(pltpath+i+'.png')
        
# Remove outlier based on Z-Score

def drop_numerical_outliers(df, limit):
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < limit, reduce=False) \
        .all(axis=1)
    
    df.drop(df.index[~constrains], inplace=True)
    df.to_csv('../Data/cln_mrg_df.csv')


# Drop columns where percent null values greater than threshold    
def drop_null(filename, threshold):
    dfp = pd.read_csv(filename, encoding='latin1')
    na_data = dfp.isna().sum()[dfp.isna().sum()!=0]
    data_dict = {'column':na_data.index,
             'count':na_data.values , 
             'pct': np.round((na_data.values*100)/dfp.shape[0],2) 
            }

    init_data = pd.DataFrame(data=data_dict)
    drop_columns = init_data['column'][init_data['pct']>threshold]

    #Drop null columns
    dfp.drop(drop_columns, axis=1, inplace=True)
    dfp.to_csv('../Output/nadrop_df.csv')
        
 
 # Fill missing values with mean/median
 
 def fill_null(filename, fill_type, fill_col):
    dfp = pd.read_csv(filename, encoding='latin1')

    #fill null values
    if fill_type== 'mean':
        for i in fill_col:
            dfp[i].fillna(dfp[i].mean(), inplace=True)
    
    elif fill_type=='median':
        for i in fill_col:
            dfp[i].fillna(dfp[i].median(), inplace=True)
            
    dfp.to_csv('../Output/nafill_df.csv')
    
    
############ Elbow curve plot to determine number of clusters#################

def elbow_num_cluster(filename, con_var, l_range, u_range, scale_ind):
    df = pd.read_csv(filename, encoding='latin1')
    df_con = df[con_var].copy()
    
    if scale_ind=='Y':
        scaler = StandardScaler()
        d_scale = scaler.fit_transform(df_con)
        df_scale = pd.DataFrame(d_scale, columns=con_var)
    
    else:
        df_scale = df_con.copy()
        
    
    wcss=[]
    for i in range (l_range, u_range):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df_scale)
        wcss.append(kmeans.inertia_)
        
    plt.plot(range(l_range, u_range), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Intertia')
    pltpath = '../Output/Elbow/'
    plt.savefig(pltpath+'elbow_plot.png')


### K-Means clustering
def create_kmean_cluster(filename, con_var, num_clust):
    df = pd.read_csv(filename, encoding='latin1')
    
    df_cont = df[con_var].copy()
    
    imputer = Imputer(strategy='mean')
    d_imp = imputer.fit_transform(df_cont)
    df_imp = pd.DataFrame(d_imp, columns=con_var)
    
    scaler = StandardScaler()
    d_scale = scaler.fit_transform(df_imp)
    df_scale = pd.DataFrame(d_scale, columns=con_var)
    
    kmeans = KMeans(n_clusters=num_clust, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    df_k = kmeans.fit_predict(df_scale)
    df['Cluster'] = df_k
    df.to_csv('../Output/kmeans.csv')


### Dimentionality Reduction using PCA    
def pca_plot(dim, filename,con_var):
    
    df = pd.read_csv(filename, encoding='latin1')
    df_cont = df[con_var].copy()
    
    imputer = Imputer(strategy='mean')
    d_imp = imputer.fit_transform(df_cont)
    df_imp = pd.DataFrame(d_imp, columns=con_var)
    
    scaler = StandardScaler()
    d_scale = scaler.fit_transform(df_imp)
    df_scale = pd.DataFrame(d_scale, columns=con_var)
    
    pca = PCA(dim)
    pca_ftr = pca.fit_transform(df_scale)
    df_pca_ftr = pd.DataFrame(pca_ftr, columns=con_var)
    df_pca_ftr.to_csv('../Output/dim_red_df.csv')
    
    num_cmpnt=len(pca.explained_variance_ratio_)
    ind = np.arange(num_cmpnt)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(18, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_cmpnt):
        ax.annotate(r"%s" % ((str(vals[i]*100)[:3])), (ind[i], vals[i]), va="bottom", ha="center", fontsize=4.5)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=10)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
    pltpath = '../Output/PCA/'
    plt.savefig(pltpath+'pca_plot.png')

def regression_func(filename, feature_list, label, reg_algo):
    df = pd.read_csv(filename, encoding='latin1')
    
    X = df[feature_list].copy()
    y = df[label].copy()
    
    r2score = []
    rmsescore = []
    
    if reg_algo == 'linear_regression':
        reg = LinearRegression()     
    
        cv = KFold(n_splits=5, random_state=40, shuffle=True)
        for train_index, test_index in cv.split(X):

            X_train, X_test, y_train, y_test = X.loc[train_index,:], X.loc[test_index,:], y.loc[train_index,:], y.loc[test_index,:]
        
            reg.fit(X_train,y_train)
            y_pred = reg.predict(X_test)
        
            rmsescore.append(mean_squared_error(y_test, y_pred))
            r2score.append(r2_score(y_test, y_pred))
                    
        result_dict = {'Test Mean Squared Error' : np.mean(rmsescore),
                        'Test R2 Score' : np.mean(r2score)}
        
    elif reg_algo == 'RandomForestRegressor':
        reg = RandomForestRegressor(random_state=40)     
    
        cv = KFold(n_splits=5, random_state=40, shuffle=True)
        for train_index, test_index in cv.split(X):

            X_train, X_test, y_train, y_test = X.loc[train_index,:], X.loc[test_index,:], y.loc[train_index,:], y.loc[test_index,:]
        
            reg.fit(X_train,y_train)
            y_pred = reg.predict(X_test)
        
            rmsescore.append(mean_squared_error(y_test, y_pred))
            r2score.append(r2_score(y_test, y_pred))
                    
        result_dict = {'Test Mean Squared Error' : np.mean(rmsescore),
                        'Test R2 Score' : np.mean(r2score)}
        
    m_filename = '../Output/reg_model.sav'
    joblib.dump(reg, m_filename)
    return result_dict 
