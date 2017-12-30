# -*- coding: utf-8 -*-
import numpy as np 
import matplotlib.pyplot as plt
import os
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets, preprocessing,cross_validation, feature_extraction
from sklearn import linear_model, svm, metrics, ensemble, tree, ensemble
from copy import copy
import urllib
import csv
from knn import *



class IDS:

	def __init__(self):
		""" Initalization of dataset configure predefined columns
		"""
		self.train_data_from_text = urllib.urlopen('kddcup.data_10_percent_corrected')
		self.test_data_from_text = urllib.urlopen('corrected')
		""" Train data read from frame """
		self.class_train = pd.read_csv(self.train_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
		self.class_test = pd.read_csv(self.test_data_from_text, quotechar=',', skipinitialspace=True, names=['Duration', 'protocol_type', 'Service', 'Flag', 'src_bytes', 'dst_bytes', 'Land', 'wrong_fragment', 'Urgent', 'Hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'Count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate','Class'])
		""" Change of train classes by tagging normal or attack """
		self.class_train.loc[(self.class_train['Class'] !='normal.'),'Class'] = 'attack'
		self.class_train.loc[(self.class_train['Class'] =='normal.'),'Class'] = 'normal'
		""" Change of test classes of tagging normal or attack """
		self.class_test.loc[(self.class_test['Class'] !='normal.'),'Class'] = 'attack'
		self.class_test.loc[(self.class_test['Class'] =='normal.'),'Class'] = 'normal'
		""" Trainset encoding- decoding """
		self.attribute_encoder = feature_extraction.DictVectorizer(sparse=False)
		self.label_encoder = preprocessing.LabelEncoder()
        
		self.neighbors = 5
		
		self.train_data_dataframe = self.attribute_encoder.fit_transform(self.class_train.iloc[:,:-1].T.to_dict().values())
		self.train_target_dataframe= self.label_encoder.fit_transform(self.class_train.iloc[:,-1])
		
		self.train_data_decoded = pd.DataFrame(self.train_data_dataframe)
		self.train_target_decoded = pd.DataFrame(self.train_target_dataframe)
		
		self.test_data_dataframe = self.attribute_encoder.transform(self.class_test.iloc[:,:-1].T.to_dict().values())
		self.test_target_dataframe = self.label_encoder.transform(self.class_test.iloc[:,-1])
		
		self.test_data_decoded = pd.DataFrame(self.test_data_dataframe)
		self.test_target_decoded = pd.DataFrame(self.test_target_dataframe)
		self.usedThresholds={}
		self.Tree=np.ones((1000,1))
		self.Thresholds=np.ones((1000,1))
		self.decisions={}
		self.Tree=-1*self.Tree
		for i in range(0,29):
				self.usedThresholds[i]=set()
		
		print "************************************************"
		print "Train Data Dimensions Without Feature Selections"
		print self.train_data_decoded.shape
		print "Test Data Dimensions Without Feature Selections"
		print self.test_data_decoded.shape
		print "************************************************"
	

    
	""" Train and test data feature reduction"""
	def feature_reduction(self) :
		self.feature_reducted_train_data = PCA(n_components=27).fit_transform(self.train_data_decoded)	
		self.feature_reducted_test_data = PCA(n_components=27).fit_transform(self.test_data_decoded)
		
		self.train_data_pca_reducted = pd.DataFrame(self.feature_reducted_train_data)
		self.test_data_pca_reducted = pd.DataFrame(self.feature_reducted_test_data)
		
		print "*********************************************"
		print "Train Data Dimensions With Feature Selections"
		print self.train_data_pca_reducted.shape
		print "Test Data Dimensions With Feature Selections"
		print self.test_data_pca_reducted.shape
		print "*********************************************"
		
	""" Normalizing dataset  """
	def normalizing_datasets(self) :
		standard_scaler = preprocessing.StandardScaler()
		self.train_ratio_normalized_scaled_values = standard_scaler.fit_transform(self.train_data_pca_reducted.values)
		self.train_data_scaled_normalized = pd.DataFrame(self.train_ratio_normalized_scaled_values)

		self.test_ratio_normalized_scaled_values = standard_scaler.fit_transform(self.test_data_pca_reducted.values)
		self.test_data_scaled_normalized = pd.DataFrame(self.test_ratio_normalized_scaled_values)
		
	
	""" SVM with third party tools """
	def svm_with_third_party(self) :
		svm_object = svm.SVC(kernel='linear', max_iter=100000000)
		svm_object.fit(self.train_data_scaled_normalized, self.train_target_decoded[0])
		svm_predict = svm_object.predict(self.test_data_scaled_normalized)
		print svm_predict.score(self.test_data_scaled_normalized, self.test_target_decoded)
		print metrics.classification_report(self.test_target_decoded, lin_predict)

	def score(self,y_test, y_predict):
		return np.sum(y_predict == y_test)/float(len(y_test))*100.0
    
	def prepare_dataset_for_written_knn(self):
		smaller_trainX = self.train_data_scaled_normalized [:1000]        
		smaller_trainY =  self.train_target_decoded[:1000]
		smaller_trainX = smaller_trainX.values
		smaller_trainY = smaller_trainY[0].values 
		smaller_testX = self.test_data_scaled_normalized[:200]
		decoder_testY = self.test_target_decoded[:200]
		smaller_testX = smaller_testX.values
		decoder_testY = decoder_testY[0].values 
		model = KNN(smaller_trainX,smaller_testX)
		pred = model.knn_main_code(smaller_trainX,smaller_testX,smaller_trainY,5)
		print('got ' + str(model.get_accuracy(pred,decoder_testY)*100) + '% predictions correct') 
       
   
   
		
		
ids = IDS()
ids.feature_reduction()
ids.normalizing_datasets()
ids.prepare_dataset_for_written_knn()
#ids.prepare_dataset_decision_tree()
ids.svm_with_third_party()
		
		
		
		
		
		
		
		
		
		
		
		
		
		