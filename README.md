##  PISAHUT-pytorch

##  Introduction

We designed the PISAHUT model, it contains four parts:  
1. We construct a multi-modal knowledge base and express it by learning knowledge from GNN.
2. We learn users' implicit preferences from multi-view user-content interaction.
3. We model and represent the basic information of users, historical texts of users and sentimental sentences.
4. We integrate heterogeneous knowledge with sentimental sentence representation.

## Environment Requirement

`pip install -r requirements.txt`
Also, please make sure you have the Chatglm-6b model in your runtime environment.

## Dataset

We use an implicit sentiment dataset D-implicit and a universal sentiment dataset D-general .

## An example to  run the model

* Modify dataset path
Change `dataset` in `model.py`

* Multi-view user implicit preference vector file using corresponding dataset
Change `file1、file2、file3` in `model.py` using `user1_embedding.pkl、user2_embedding.pkl、user3_embedding.pkl` we provided.
(*These three files are the vector representations learned from the graph neural network model DMV-GCN mentioned in this paper.*)

 * Use the graph node vector file
   
	Change `kg_file` in `model.py` using `kg_emb.pkl` we provided.
	  (*This file is a vector representation learned from the multi-modal knowledge base mentioned in this paper.*)
	
 * Run the model
 After making the above modifications, you can execute `python ../model.py` through the command line to train the model.
