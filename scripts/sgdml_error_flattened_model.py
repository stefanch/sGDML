#!/usr/bin/python

# MIT License
#
# Copyright (c) 2018 Stefan Chmiela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function

import argparse
import os
import re
import shutil
import sys
import time
from functools import partial

import numpy as np

from sgdml import __version__
from sgdml.predict import GDMLPredict
from sgdml.train import GDMLTrain
from sgdml.utils import io, ui
from sgdml.cli import _print_splash

from sgdml import cli


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = 'sgdml'

def dataset_r_to_desc(dataset):
	'''Convert positions to the descriptor
	
	Descriptor is taken from sgdml (desc.r_t_to_desc)
	
	Arguments:
		dataset {npz} -- Dataset given by user
	
	Returns:
		numpy array -- Descriptor for all samples in the given dataset
	'''

	from scipy.spatial.distance import pdist
	from sgdml.utils import desc

	R=dataset['R']
	R_new=[]

	for r_i in R:
		pdist=desc.pdist(r_i)
		R_new.append(desc.r_to_desc(r_i,pdist))

	return np.array(R_new)

def dataset_extract_E(dataset):
	return dataset['E']

def dataset_extract_F_concat(dataset):
	F=dataset['F']
	n_samples,n_atoms,n_dim=F.shape
	return np.reshape(F,(n_samples,n_atoms*n_dim))

def dataset_extract_R_concat(dataset):
	R=dataset['R']
	n_samples,n_atoms,n_dim=R.shape
	return np.reshape(R,(n_samples,n_atoms*n_dim))


parameters={
	# These parameters control the settings for this script and serve
	# as a way to easily adjust the algorithm to one's needs.
	# When the input needed is 'func', it requires a string that 
	# corresponds to a function name in this file
	
	'var_funcs':{
		# Functions to be called when loading the dataset
		# This is the time to compute e.g. descriptors 
		# Parameters are {index}:{func},
		# Input given to the function is the dataset as provided by the user
		0:'dataset_r_to_desc',
		1:'dataset_extract_E',
		2:'dataset_extract_F_concat',
		3:'dataset_extract_R_concat',
	},

	'clusters':{
		# Information about clustering schemes
		'init_cluster':[0,1],    #array of indices*, defines sequence of initial clustering algorithms for the entire datasets
		'recluster':[2],    #array of indices*, defines sequence of reclustering algorithms for learning subsets
		# *indices: int: correspond to the clustering algorithms as described/defined below

		0:{
			'type':'agglomerative_clustering',    # func, input: initial dataset, subset indices, clustering index (as defnied here); output: cluster indices
			'n_clusters':10, 
			'initial_number':10000,    # agglo. cl. first clusters a subset of {initial_number} points, then fills the clusters with the rest of the data
			'distance_matrix_function':'distance_matrix_euclidean',    #func, input: samples indices; output: pairwise distance matrix
			'linkage':'complete',    #string, defines linkage type
			'cluster_choice_criterion':'smallest_max_distance_euclidean',    #func, input: sample, clusters (in descriptor space); output: cluster index that sample belongs to
			'var_index':0,    # int: which descriptor fron var_funcs to use for clustering
			},

		1:{
			'type':'kmeans_clustering',
			'n_clusters':5,
			'var_index':1,
			},

		2:{
			'type':'agglomerative_clustering',
			'n_clusters':'auto', 
			#'n_clusters_args':[400],
			'initial_number':10000, #1 means the entire subset
			'distance_matrix_function':'distance_matrix_euclidean',
			'linkage':'complete',
			'cluster_choice_criterion':'smallest_max_distance_euclidean',
			'var_index':0,
			},
	}, #end of 'clusters'

	'predict_error':{
		# defines how to calculate prediction errors
		'predict':'forces',   # string='forces' or 'energies'
		'input_var_index':3,    # int, which descriptor from var_funcs to use as input for prediction function (i.e. R_concat)
		'comparison_var_index':2,    # int, what to compare it to (i.e. forces)
		'error_func':'mean_squared_error_sample_wise',    # array, input: real values, predicted values, output: array of sample-wise errors
	}, #end of 'predict_error'

	'generate_subset':{
		'func':'cluster_above_mse',    # func, input: iterator, args below, output: array of all indices in chosen clusters
		'args':[1],    
	}, #end of 'generate_subset'

	'generate_training_data':{
		'func':'within_cluster_weighted_err_N',    # func, input: iterator, number of points N, output: array of indices (length N)
	},

} #end of parameters

def smallest_max_distance_euclidean(sample,clusters):
	'''
	Finds the cluster that the given sample belongs to. Simple euclidean distance is used.
	(The metric used should be the same as for the agglomerative clustering.)
	
	Paramters:
		-sample: 
			numpy array containing positions of atoms of one samples
			Dimensions: (n_atoms,n_dimensions)
		
		-clusters: 
			numpy array containing positions within each cluster
			Dimensions: (n_clusters,n_atoms,n_dimensions)
					
	Returns:
		-index of cluster that the sample belongs to / closest cluster
										 
	'''
	
	g=np.zeros(len(clusters))
	for i in range(len(clusters)):
		g[i]=np.max(np.sum(np.square(clusters[i]-sample),1))   #numpy difference=>clusters[c]-sample elementwise for each c
	return np.argmin(g)
	
def distance_matrix_euclidean(data):
	return euclidean_distances(data,data)

cluster_funcs={}

def find_function(s):
	s=str(s)
	if s in globals() and callable(globals()[s]):
		return globals()[s] 

	sys.exit(
		ui.fail_str('[FAIL]')
		+ f' Could not find function of name: {s}'
	)

def agglomerative_clustering(data_set,indices,clustering_index):


	para=data_set.para['clusters'][clustering_index]
	f_name=para["distance_matrix_function"]
	matrix_f=find_function(f_name)

	data,N,ind_cluster,cluster_data=None,None,[],[]
	n_clusters=para['n_clusters']

	##generate data
	if not (indices is None):
		data=data_set.vars[para['var_index']][indices]
	else:
		data=data_set.vars[para['var_index']]


	##get number of initial clusters
	if para["initial_number"]>1:
		N=para["initial_number"]
	else:
		N=len(data)*para["initial_number"]

	ui.progr_toggle(False,"Agglomerative clustering",f"(initial subset, {N} points)")

	##prepare agglomerative vars
	ind_all=np.arange(len(data))
	ind_init=np.random.permutation(ind_all)[:N] 
	data_init=data[ind_init]
	ind_rest=np.delete(ind_all,ind_init)
	data_rest=data[ind_rest]
	M=matrix_f(data_init)


	cinit_labels=AgglomerativeClustering(affinity="precomputed",n_clusters=n_clusters,linkage=para.get('linkage','complete')).fit_predict(M)

	ui.progr_toggle(True,"Agglomerative clustering",f"(initial subset, {N} points)")

	cluster_ind=[]
	for i in range(n_clusters):
		ind=np.concatenate(np.argwhere(cinit_labels==i))

		#convert back to initial set of indices
		ind=ind_init[ind]

		cluster_ind.append(ind.tolist())
		cluster_data.append(np.array(data[cluster_ind[i]]))


	#divide rest into clusters
	#using para['cluster_choice_criterion']
	#+ni to find the old index back from entire dataset
	#print("Clustering rest of data...")
	#outs=np.trunc(np.linspace(0,len(data_rest),99))
	choice_func=find_function(para["cluster_choice_criterion"])


	len_data=len(data_rest)
	for i in range(len_data):
		c=choice_func(data_rest[i],cluster_data) #c is the cluster# it belongs to
		cluster_ind[c].append(ind_rest[i])
		ui.progr_bar(i,len_data,"Agglomerative Clustering",f"(rest of data, {len_data} points)")

	ui.progr_bar(len_data,len_data,"Agglomerative Clustering",f"(rest of data, {len_data} points)")

	if indices is None:
		return cluster_ind

	#if needed, change the indices of every cluster back corresponding to original data set
	for cl in cluster_ind:
		for i in range(len(cl)):
			cl[i]=indices[cl[i]]
	ui.progr_toggle(True,"Agglomerative clustering",None)
	return cluster_ind

def kmeans_clustering(data_base,indices,clustering_index):

	ui.progr_toggle(False,f"KMeans clustering ({len(indices)} points)",None)
	para=data_base.para['clusters'][clustering_index]
	data,n_clusters,cluster_ind=None,para["n_clusters"],[]
	if not (indices is None):
		data=data_base.vars[para["var_index"]][indices]
	else:
		data=data_base.vars[para["var_index"]]


	cluster_labels=KMeans(n_clusters=n_clusters,init="k-means++").fit_predict(data)

	for i in range(n_clusters):
		ind=np.concatenate(np.argwhere(cluster_labels==i).tolist())
		#convert back to initial set of indices
		#no need here
		cluster_ind.append(ind)

	if indices is None:
		return cluster_ind

	#if needed, change the indices of every cluster back corresponding to original data set
	for cl in cluster_ind:
		for i in range(len(cl)):
			cl[i]=indices[cl[i]]
	ui.progr_toggle(True,f"KMeans clustering ({len(indices)} points)",None)

	return cluster_ind

def worst_N_clusters(self,N,*args):
	mse=self.cluster_err
	cl_ind=self.cluster_indices
	sorted_ind=np.argsort(mse)
	clusters=np.array(cl_ind)[sorted_ind[-N:]]
	ind=np.concatenate(clusters)
	return ind

def mean_squared_error_sample_wise(x,y):
	err=(np.array(x)-np.array(y))**2
	return err.mean(axis=1)

def weighted_distribution(N,weights):
	weights=np.array(weights)/np.sum(weights)
	a=(weights*N)
	b=a.astype(int)
	c=a-b
	s=np.sum(b)

	for i in range(N-s):
		arg=np.argmax(c)
		c[arg]=0
		b[arg]=b[arg]+1

	return b

def within_cluster_weighted_err_N(self,N):

	cl_ind,err=self.recluster_indices,self.sample_err 

	new_ind=[]

	#find cluster errors and pops
	mean_error=np.array([np.mean(err[x]) for x in cl_ind])
	pop=np.array([len(x) for x in cl_ind])

	weights=(mean_error/np.sum(mean_error))*(pop/np.sum(pop))
	Ns=weighted_distribution(N,weights)

	for  i in range(len(cl_ind)):
		ind=np.array(cl_ind[i])
		cl_err=err[ind]
		ni=Ns[i]
		argmax=np.argsort(-cl_err)[:ni]
		new_ind.extend(ind[argmax])

	return new_ind

def cluster_above_mse(self,fact,*args):
	mse=np.array(self.cluster_err)
	mmse=np.mean(mse)
	cl_ind=self.cluster_indices
	cl_ind_new=np.concatenate(np.argwhere(mse>mmse*fact))
	clusters=np.array(cl_ind)[cl_ind_new]
	ind=np.concatenate(clusters)
	return ind

class iterator():

	def __init__(self,args,para):
		self.args=args
		self.para=para
		self.n_steps=args['n_steps']

		#step size
		self.step_size=(self.args['n_train_final']-self.args['n_train_init'])//self.args['n_steps']

		#change n_clusters='auto' parameters to 2x step_size
		for k,v in para['clusters'].items():
			if (type(v) is dict) and ('n_clusters' in v) and (v['n_clusters']=='auto'):
				v['n_clusters']=int(2*self.step_size)

		self.create_storage_dir()

	current_stage=0
	current_step=0
	stages=[
		'load_dataset',
		'train_initial_model',
		'cluster_dataset',
		'stop_or_loop',
		'compute_prediction_errors',
		'generate_subset',
		'recluster_subset',
		'generate_training_data',
		'train_step_model',
		'stop_or_loop',
	]

	def create_storage_dir(self):
		name=os.path.basename(self.args['dataset'][0]).split('.')[0]

		dir_id=0
		basedir=os.path.join(os.getcwd(),f"{name}_i{self.args['n_train_init']}_{self.args['n_train_final']}_s{self.args['n_steps']}")
		basedir_id=basedir
		while os.path.exists(basedir_id):
			dir_id=dir_id+1
			basedir_id=f"{basedir}_{dir_id}"

		os.mkdir(basedir_id)
		self.dir=basedir_id

	def go_to_stage(self,s):
		for i in range(len(self.stages)):
			if self.stages[i]==s:
				self.current_stage=i 
				break 

	running=False
	def run(self):
		self.running=True

		print(
			ui.white_back_str(f" INIT ")
			+ui.white_bold_str(" Error-flattened model \n \n")
			+f"Total steps:   {self.n_steps:<6} \n"
			+f"Initial model: {self.args['n_train_init']:<6} training points \n"
			+f"Final model:   {self.args['n_train_final']:<6} training points \n"
			+f"Step size:     {self.step_size:<6} training points"
			)

		while self.running:
			self.proceed()

		print(ui.green_back_str('  DONE  ') + ' Error-flattening successfully completed.')
		print(f'         Latest model:       {self.step_model_lastest_path}')
		print(f'         Training poiints:   {len(self.model["idxs_train"])}')

	def proceed(self):

		if self.current_stage<len(self.stages):
			func=getattr(self,self.stages[self.current_stage],None)
			if func is None:
				sys.exit(
					ui.fail_str('[FAIL]')
					+ f' Could not find function associated with stage: {self.stages[self.current_stage]}'
				)
			else:
				func()
			self.current_stage+=1

		else:
			self.running=False

	def model_name(self,tag,N):
		return os.path.join(self.dir,f"model_{tag}_{N}.npz")

	def task_dir_name(self,tag,N):
		return os.path.join(self.dir,f"task_{tag}_{N}.npz")

	def stop_or_loop(self):
		if self.current_step<self.n_steps:

			self.go_to_stage('compute_prediction_errors')
			self.current_step+=1
			self.current_stage-=1 #because in proceed, self.current_stage+=1 will still be called AFTER stop_or_loop

			print("\n"+
				ui.white_back_str(f" STEP {self.current_step}/{self.n_steps} ")
				+ ui.white_bold_str(" Improving model...")
				+ "\n"
				)            
		else:
			self.running=False

	def load_dataset(self):
		self.dataset=self.args['dataset'][1]
		self.vars=[]

		for x in self.para['var_funcs'].values():
			if x in globals():
				self.vars.append(globals()[x](self.dataset))

	def _ui_train_print(self,args,s):
		print("\n"+
			ui.white_back_str(f" STAGE ")
			+ ui.white_bold_str(s)
			+"\n \n"
			+f"Training set:      {args['dataset'][0]} \n"
			+f"Number of points:  {args['n_train']} \n"
			)

	def train_initial_model(self):
		# def all(
		#     dataset,
		#     valid_dataset,
		#     test_dataset,
		#     n_train,
		#     n_valid,
		#     n_test,
		#     sigs,
		#     gdml,
		#     use_E,
		#     use_E_cstr,
		#     use_cprsn,
		#     overwrite,
		#     max_processes,
		#     use_torch,
		#     solver,
		#     task_dir=None,
		#     model_file=None,
		#     **kwargs
		# ):
		args=self.args.copy()
		task_dir,model_name=self.task_dir_name('init',args['n_train_init']),self.model_name('init',args['n_train_init'])
		args['task_dir']=task_dir
		args['n_test']=0
		args['test_dataset']=args['dataset']
		args['n_train']=args['n_train_init']
		args['model_file']=model_name
		args['overwrite']=True
		args['command']='all'
		self._ui_train_print(args," Train initial model")
		cli.all(**args)

		model_path=os.path.join(task_dir,model_name)
		try:
			model=np.load(model_path,allow_pickle=True)
			self.training_indices=model['idxs_train']
			self.model=GDMLPredict(model)

		except Exception as e:
			sys.exit(
				ui.fail_str('[FAIL]')
				+ f' Unable to load post-training init .npz model file at {model_path}.'
			)
	def train_step_model(self):
		# def all(
		#     dataset,
		#     valid_dataset,
		#     test_dataset,
		#     n_train,
		#     n_valid,
		#     n_test,
		#     sigs,
		#     gdml,
		#     use_E,
		#     use_E_cstr,
		#     use_cprsn,
		#     overwrite,
		#     max_processes,
		#     use_torch,
		#     solver,
		#     task_dir=None,
		#     model_file=None,
		#     **kwargs
		# ):
		args=self.args.copy()
		n_train=len(self.training_indices)

		task_dir,model_name=self.task_dir_name('step',n_train),self.model_name('step',n_train)
		args['task_dir']=task_dir
		args['n_test']=(self.is_last_step() and 1000) or 0
		args['valid_dataset']=self.args['dataset']
		args['test_dataset']=self.args['dataset']
		args['dataset']=self.step_dataset
		args['n_train']=n_train
		args['model_file']=model_name
		args['overwrite']=True
		args['command']='all'
		self._ui_train_print(args," Train step model")
		cli.all(**args)

		model_path=os.path.join(task_dir,model_name)
		
		model=np.load(model_path,allow_pickle=True)

		#fix 'idxs_train'
		model_fix=dict(model)
		model_fix['idxs_train']=self.training_indices
		np.savez_compressed(model_path, **model_fix)

		#load again
		model=np.load(model_path,allow_pickle=True)
		self.step_model_lastest_path=model_path
		self.model=GDMLPredict(model)

	def predict(self,R):
		return self.model.predict(R)

	def compute_prediction_errors(self):

		para=self.para['predict_error']

		#helping variables
		cluster_indices=self.cluster_indices
		n_clusters=len(cluster_indices)

		input_values=self.vars[para['input_var_index']]
		comparison_values=self.vars[para['comparison_var_index']]
		energies,forces=self.predict(input_values)

		if para['predict']=='energies':
			predict_values=energies
		else:  
			predict_values=forces

		error_func=find_function(para['error_func'])
		err=error_func(predict_values,comparison_values)
		mse=err.mean()
		self.sample_err=err
		self.cluster_err=[err[x].mean() for x in cluster_indices]

	def cluster_dataset(self):
		print("\n"+
			ui.white_back_str(f" STAGE ")
			+ ui.white_bold_str(" Clustering dataset")
			+ "\n"
			)   
		self.cluster_indices=self.cluster_do(None,'init_cluster')

	def recluster_subset(self):

		print("\n"+
			ui.white_back_str(f" STAGE ")
			+ ui.white_bold_str(" Reclustering ")
			+ "\n"
			)

		self.recluster_indices=self.cluster_do(self.subset_ind,'recluster')

	def cluster_do(self,init_indices,para_ind):
		para=self.para['clusters']
		cluster_para_ind=para.get(para_ind,None)
		n_clusters=((not cluster_para_ind is None) and len(cluster_para_ind)) or 0
		if n_clusters==0:
			return 

		#perform first clusterisation
		cl_type=para[cluster_para_ind[0]]['type']
		cl_func=find_function(cl_type)
		cl_ind=cl_func(self,init_indices,cluster_para_ind[0])



		#perform further clusterisations  
		for i in cluster_para_ind[1:]:
			cl_ind_new=[]
			for cl in cl_ind:
				cl_type=para[i]['type']
				cl_func=find_function(cl_type)
				cl_cl_ind=cl_func(self,cl,i)
				for j in cl_cl_ind:
					cl_ind_new.append(j)

			cl_ind=cl_ind_new

		return cl_ind

	def is_last_step(self):
		return self.current_step==self.n_steps

	def generate_subset(self):  
		para=self.para['generate_subset']
		func_name=para['func']
		func=find_function(func_name)
		self.subset_ind=func(self,*para['args'])

	def generate_training_data(self):
		cl_ind,err=self.recluster_indices,self.sample_err
		para=self.para['generate_training_data']


		#generate new indices
		indices_func=find_function(para['func'])
		if self.is_last_step(): #if last step, complete the n_train exactly (avoid rouning errors)
			step_size=self.args['n_train_final']-len(self.training_indices)
		else:
			step_size=self.step_size
		ind=indices_func(self,step_size)

		print("\n"+
			ui.white_back_str(f" STAGE ")
			+ ui.white_bold_str(" Generate new training set ")
			+ "\n"   
			)

		self.training_indices=np.concatenate([self.training_indices,ind])
		self.dataset_subset()

	def dataset_subset(self):
		dataset=dict(self.dataset)
		indices=self.training_indices
		new={}
		for k,v in dataset.items():
			if k=='md5':
				pass
			elif k in ['R','E','F']:
				new[k]=v[indices]
			else:
				new[k]=v 

		new['md5']=io.dataset_md5(new)
		name=os.path.join(self.dir,f"training_set_{len(indices)}.npz")

		print(ui.white_bold_str("Saving dataset: \n")
			+f"Name:      {name} \n"
			+f"Points:    {len(indices)} \n"
			)

		np.savez_compressed(name,**new)
		a=np.load(name,allow_pickle=True)
		self.step_dataset=[name,a]

	def _print_stage_title(self):
		print(
			ui.info_str('[INFO]')
			+ ' Symmetry search is limited to a random subset of 1000/'
			+ str(n_train)
			+ ' training points for faster convergence.'
		)

def parse_arguments():
	'''Parses arguments

	Mostly equivalent/copied from cli.py file

	Returns:
		args -- dictionary
	'''

	def _add_argument_dataset(parser, help='path to dataset file'):
		parser.add_argument(
			'dataset',
			metavar='<dataset_file>',
			type=lambda x: io.is_file_type(x, 'dataset'),
			help=help,
		)

	def _add_argument_sample_size(parser, subset_str):
		parser.add_argument(
			'n_%s' % subset_str,
			metavar='<n_%s>' % subset_str,
			type=io.is_strict_pos_int,
			help='%s sample size' % subset_str,
		)

	def _add_argument_dir_with_file_type(parser, type, or_file=False):
		parser.add_argument(
			'%s_dir' % type,
			metavar='<%s_dir%s>' % (type, '_or_file' if or_file else ''),
			type=lambda x: io.is_dir_with_file_type(x, type, or_file=or_file),
			help='path to %s directory%s' % (type, ' or file' if or_file else ''),
		)
   
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--version',
		action='version',
		version='%(prog)s '
		+ __version__
		+ ' [python '
		+ '.'.join(map(str, sys.version_info[:3]))
		+ ']',
	)

	parser.add_argument(
		'-o',
		'--overwrite',
		dest='overwrite',
		action='store_true',
		help='overwrite existing files',
	)

	parser.add_argument(
		'-p',
		'--max_processes',
		metavar='<max_processes>',
		type=io.is_strict_pos_int,
		help='limit the number of processes for this application',
	)

	parser.add_argument(
		'--torch',
		dest='use_torch',
		action='store_true',
		help='use PyTorch for validation and test',
	)
	
	_add_argument_dataset(parser)
	_add_argument_sample_size(parser, 'train_init')
	_add_argument_sample_size(parser, 'train_final')

	parser.add_argument(
		'n_steps',
		metavar='<n_steps>',
		type=io.is_strict_pos_int,
		help='number of steps',
	)

	_add_argument_sample_size(parser, 'valid')

	parser.add_argument(
		'-v',
		'--validation_dataset',
		metavar='<validation_dataset_file>',
		dest='valid_dataset',
		type=lambda x: io.is_file_type(x, 'dataset'),
		help='path to validation dataset file',
	)

	parser.add_argument(
		'-s',
		'--sig',
		metavar=('<s1>', '<s2>'),
		dest='sigs',
		type=io.parse_list_or_range,
		help='integer list and/or range <start>:[<step>:]<stop> for the kernel hyper-parameter sigma',
		nargs='+',
	)

	parser.add_argument(
		'--task_dir',
		metavar='<task_dir>',
		dest='task_dir',
		help='user-defined task output dir name',
	)

	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'--gdml',
		action='store_true',
		help='don\'t include symmetries in the model (GDML)',
	)
	group.add_argument(
		'--cprsn',
		dest='use_cprsn',
		action='store_true',
		help='compress kernel matrix along symmetric degrees of freedom',
	)

	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'--no_E',
		dest='use_E',
		action='store_false',
		help='only reconstruct force field w/o potential energy surface',
	)
	group.add_argument(
		'--E_cstr',
		dest='use_E_cstr',
		action='store_true',
		help='include the energy constraints in the kernel',
	)

	parser.add_argument(
		'n_test',
		metavar='<n_test>',
		type=io.is_strict_pos_int,
		help='test sample size',
		nargs='?',
		default=None,
	)

	parser.add_argument(
		'--model_file',
		metavar='<model_file>',
		dest='model_file',
		help='user-defined model output file name',
	)


	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'--cg',
		dest='use_cg',
		action='store_true',
		help='use iterative solver (conjugate gradient) with Nystroem preconditioner',
		#help=argparse.SUPPRESS
	)
	group.add_argument(
		'--fk',
		dest='use_fk',
		action='store_true',
		help='use iterative solver (conjugate gradient) with Nystroem approximation',
		#help=argparse.SUPPRESS
	)

	args = parser.parse_args()

	# post-processing for optional sig argument
	if 'sigs' in args and args.sigs is not None:
		args.sigs = np.hstack(
			args.sigs
		).tolist()  # flatten list, if (part of it) was generated using the range syntax
		args.sigs = sorted(list(set(args.sigs)))  # remove potential duplicates

	# post-processing for optional model output file argument
	if 'model_file' in args and args.model_file is not None:
		if not args.model_file.endswith('.npz'):
			args.model_file += '.npz'

	_print_splash()

	# check PyTorch GPU support
	if 'use_torch' in args and args.use_torch:
		try:
			import torch
		except ImportError:
			pass
		else:
			if not torch.cuda.is_available():
				print(
					ui.warn_str('\n[WARN]')
					+ ' Your PyTorch installation does not support GPU computation!'
					+ '\n       We recommend running CPU calculations without \'--torch\' for improved performance.'
				)

	# replace solver flags with keyword
	args = vars(args)
	args['solver'] = 'analytic'
	if 'use_cg' in args and args['use_cg']:
		args['solver'] = 'cg'
	elif 'use_fk' in args and args['use_fk']:
		args['solver'] = 'fk'
	args.pop('use_cg', None)
	args.pop('use_fk', None)

	return args

if __name__=='__main__':
	args=parse_arguments()
	it=iterator(args,parameters)
	it.run()

























