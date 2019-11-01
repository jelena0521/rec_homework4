import os
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#读取数据

train_data=pd.read_csv('titanic/train.csv')
test_data=pd.read_csv('titanic/test.csv')

#查看基本信息
print(train_data.info())
#判断是否存在nan的列
print(train_data.isnull().any())

#年龄为nan的取平均值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)


#港口取众值
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('Stes',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

# 特征选择
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

#特征向量化
dv=DictVectorizer(sparse=False) #对字典形式的非数值value进行onehot
train_features=dv.fit_transform(train_features.to_dict(orient='record')) #to_dict转为字典形式
test_features=dv.transform(test_features.to_dict(orient='record'))

#定义模型
#定义分类器
classifiers=[SVC(random_state=1),DecisionTreeClassifier(random_state=1),KNeighborsClassifier(),LogisticRegression(random_state=1),
             CatBoostClassifier(random_state=1),XGBRFClassifier(random_state=1),LGBMClassifier(random_state=1)]

#定义分类器名字
classifiers_names=['svc','dt','knn','lr','cbc','xgbfc','lgbmc']

#定义参数
# classifiers_param=[{'svc_C':[0.1,1,10]},{'dt_min_samples_split':[1,3,5]},{'knn_n_neighbors':[3,5,7]},{'lr_c':[0.1,1,10]},
#                    {'cbc_learning_rate':[0.01,0.05,0.1]},{'xgbfc_learning_rate':[0.01,0.05,0.1]},{'lgbmc_learning_rate':[0.01,0.05,0.1]}]
classifiers_param=[{'C':[0.1,0.5,1]},{'criterion':['gini','entropy']},{'n_neighbors':[1,2,3]},{'C':[0.1,0.5,1]},
                   {'learning_rate':[0.01,0.05,0.1]},{'learning_rate':[0.01,0.05,0.1]},{'learning_rate':[0.01,0.05,0.1]}]


# def GridSearchCV_work(pipeline,train_features,train_labels,test_features,param_grid):
#     gridsearch=GridSearchCV(estimator=pipeline,param_grid=param_grid)
#     search=gridsearch.fit(train_features,train_labels)
#     print('最优评估器：',search.best_estimator_)
#     print('最优参数：',search.best_params_)
#     print('最优分数：',search.best_score_)
#     # predict=gridsearch.predict(test_features)
#     # acc_score=metrics.accuracy_score(predict,test_labels)
#     return search

for model,model_name,param in zip(classifiers,classifiers_names,classifiers_param):
    pipeline=Pipeline([('ss',StandardScaler()),(model_name,GridSearchCV(model,param))])
    search=pipeline.fit(train_features,train_labels)
    print('评估器：',model_name)
    print('最优分数：',search.score(train_features,train_labels))

# clf=DecisionTreeClassifier(criterion='gini')
# clf.fit(train_features,train_labels)
# acc_decision_tree = round(clf.score(train_features, train_labels), 6)
# print(u'score准确率为 %.4lf' % acc_decision_tree)
'''
评估器： svc
最优分数： 0.8428731762065096
评估器： dt
最优分数： 0.9820426487093153
评估器： knn
最优分数： 0.8731762065095399
评估器： lr
最优分数： 0.8013468013468014
评估器： cbc
最优分数： 0.920314253647587
评估器： xgbfc
最优分数： 0.8305274971941639
评估器： lgbmc
最优分数： 0.9012345679012346
'''