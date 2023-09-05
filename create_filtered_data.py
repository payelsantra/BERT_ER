import pickle
from ast import literal_eval
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import pickle as pkl
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch

def unique_claim_evidence_pair(val_evidence_gen,zip_new_val_temp_opp):
    ist_for_all=[]
    unique_dict={}
    uniq_id_ls=[]
    uniq_id_clm=[]
    for p in val_evidence_gen:
      for i,j in enumerate(val_evidence_gen[p]['claim']):
        id=zip_new_val_temp_opp[j]
        gen_sen=val_evidence_gen[p]['generated_sen'][i*20:(i+1)*20]
        trans_val=val_evidence_gen[p]['transition_val'][i*20:(i+1)*20]
        sen_value=dict(zip(gen_sen,trans_val))
        sen_value_temp={}
        for k,l in enumerate(sen_value):
          sen_id="{}_{}".format(id,k)
          sen_value_temp[l]=[]
          sen_value_temp[l].append(sen_id)
          sen_value_temp[l].append(sen_value[l])
          uniq_id_ls.append(sen_id)
          uniq_id_clm.append(l)
        list_for_all.append(sen_value_temp)
        unique_dict[id]={"claim":j,"evi_pair":sen_value_temp}
    return unique_dict,list_for_all,uniq_id_ls,uniq_id_clm

def make_lists_for_cev(unique_dict_tr):
    claim_list_tr=[]
    evi_list_tr=[]
    claim_lb_list_tr=[]
    evi_label_list_tr=[]
    evi_score_list_tr=[]
    label_list_tr=[]
    for i in tqdm(unique_dict_tr):
      label=zip_new_tr_id_lbl[i]
      #print(label)
      claim=unique_dict_tr[i]['claim'].strip("Please provide a rationale either in favour of or against the assertion ").replace("'","")
      gen_pair=unique_dict_tr[i]['evi_pair']
      for k in gen_pair:
        evi_id_tok=gen_pair[k][0]
        evi_prob_tok=gen_pair[k][1]
        claim_lb_list_tr.append(i)
        evi_label_list_tr.append(evi_id_tok)
        claim_list_tr.append(claim)
        evi_score_list_tr.append(evi_prob_tok)
        evi_list_tr.append(k)
        label_list_tr.append(label)
    return claim_lb_list_tr,evi_label_list_tr,claim_list_tr,evi_score_list_tr,evi_list_tr,label_list_tr
    
def result_file_to_dict(path_of_result_file):
    f_lm = open(path_of_result_file, 'r') 
    lines=f_lm.readlines()
    result={}
    for i,x in enumerate(lines):
      if i!=0:
      #result.append(x.split(' ')[1])
        splitted_res=x.strip().split('\t')
        id=splitted_res[0]
        value=splitted_res[2]
        result[id]=literal_eval(value)
    return result

def nqc_result_file(result):
    nqc_id=list(result.keys())
    nqc=result.values()
    max_nqc=max(nqc)
    min_nqc=min(nqc)
    result_nqc={}
    for k in result:
      norm_value= result[k]/max_nqc
      result_nqc[k]=norm_value 
    sen_id=[]
    for i in result_nqc:
      nqc_val=result_nqc[i]
      if nqc_val>0.010958547123313385:
        sen_id.append(i)
    return sen_id

def form_sorted_dict(unique_dict_tr,sen_id):
    sorted_dict_tr={}
    for i in tqdm(unique_dict_tr):
      evi_pair=unique_dict_tr[i]['evi_pair']
      claim=unique_dict_tr[i]['claim']
      sorted_created_pair=[]
      for j in evi_pair:
        id=evi_pair[j][0]
        if id in sen_id:
          sorted_created_pair.append(j)
      recent_dict={}
      for k in sorted_created_pair:
        recent_dict[k]=evi_pair[k]
      sorted_dict_tr[i]={"claim":claim,"sorted_pair":recent_dict}
    return sorted_dict_tr
    
def make_list_for_csv(sorted_dict_tr,zip_new_tr_id_lbl):
    claim_list_tr=[]
    evi_list_tr=[]
    claim_lb_list_tr=[]
    evi_label_list_tr=[]
    evi_score_list_tr=[]
    label_list_tr=[]
    for i in tqdm(sorted_dict_tr):
      label=zip_new_tr_id_lbl[i]
      #print(label)
      claim=sorted_dict_tr[i]['claim'].strip("Please provide a rationale either in favour of or against the assertion ").replace("'","")
      gen_pair=sorted_dict_tr[i]['sorted_pair']
      for k in gen_pair:
        evi_id_tok=gen_pair[k][0]
        evi_prob_tok=gen_pair[k][1]
        claim_lb_list_tr.append(i)
        evi_label_list_tr.append(evi_id_tok)
        claim_list_tr.append(claim)
        evi_score_list_tr.append(evi_prob_tok)
        evi_list_tr.append(k)
        label_list_tr.append(label)
    return claim_lb_list_tr,claim_list_tr,evi_label_list_tr,evi_list_tr,evi_score_list_tr,label_list_tr
    

# create a basic csv file from the FEVER jsonl files, where the columns are "id","Claim","label".
training_data = pd.read_csv(train_csv_file)
valid_data = pd.read_csv(validation_csv_file)
test_data = pd.read_csv(test_csv_file)

X_train=list(training_data["CLAIM"])
X_train_id=list(training_data['ID'])
y_train=list(training_data["LABEL"])
X_test=list(test_data["CLAIM"])
y_test=list(test_data["LABEL"])
X_test_id=list(test_data['ID'])
X_val=list(valid_data["CLAIM"])
y_val=list(valid_data["LABEL"])
X_val_id=list(valid_data['ID'])

zip_new_tr=dict(zip(X_train_id,X_train))
zip_new_ts=dict(zip(X_test_id,X_test))
zip_new_val=dict(zip(X_val_id,X_val))
zip_new_tr_id_lbl=dict(zip(X_train_id,y_train))
zip_new_ts_id_lbl=dict(zip(X_test_id,y_test))
zip_new_val_id_lbl=dict(zip(X_val_id,y_val))

zip_new_tr_temp_opp={zip_new_tr_temp[k]:k for k in zip_new_tr_temp}
zip_new_val_temp_opp={zip_new_val_temp[k]:k for k in zip_new_val_temp}
zip_new_ts_temp_opp={zip_new_ts_temp[k]:k for k in zip_new_ts_temp}

#load the generated_sentences dumped in generated_sentence_creation.py
train_evidence_gen=pickle.load(open(train_generated_sen,"rb"))
val_evidence_gen=pickle.load(open(val_generated_sen,"rb"))
test_evidence_gen=pickle.load(open(test_generated_sen,"rb"))

unique_dict_tr,list_for_all_tr,uniq_id_ls_tr,uniq_id_clm_tr=unique_claim_evidence_pair(train_evidence_gen,zip_new_tr_temp_opp)
unique_dict,list_for_all,uniq_id_ls,uniq_id_clm=unique_claim_evidence_pair(val_evidence_gen,zip_new_val_temp_opp)
unique_dict_ts,list_for_all_ts,uniq_id_ls_ts,uniq_id_clm_ts=unique_claim_evidence_pair(test_evidence_gen,zip_new_ts_temp_opp)


# create the result file for training and testing data using java qpp_eval code
result_tr=result_file_to_dict(path_of_result_training_file)
result_ts=result_file_to_dict(path_of_result_testing_file)

#the evidences if which have NQC values greater tha threshold
sen_id_tr=nqc_result_file(result_tr)
sen_id_ts=nqc_result_file(result_ts)

#create dictionary after taking the threshold
sorted_dict_tr=form_sorted_dict(unique_dict_tr,sen_id_tr)
sorted_dict_dev=form_sorted_dict(unique_dict,sen_id_tr)
sorted_dict_ts=form_sorted_dict(unique_dict_ts,sen_id_ts)

#create the lists for creating csv s
claim_lb_list_tr,claim_list_tr,evi_label_list_tr,evi_list_tr,evi_score_list_tr,label_list_tr=make_list_for_csv(sorted_dict_tr,zip_new_tr_id_lbl)
claim_lb_list_ts, claim_list_ts, evi_label_list_ts, evi_list_ts, evi_score_list_ts, label_list_ts=make_list_for_csv(sorted_dict_ts,zip_new_ts_id_lbl)
claim_lb_list, claim_list, evi_label_list, evi_list, evi_score_list,label_list=make_list_for_csv(sorted_dict_dev,zip_new_val_id_lbl)
   
    
df = pd.DataFrame(list(zip(claim_lb_list, claim_list, evi_label_list, evi_list, evi_score_list,label_list)),columns =['Claim_id', 'Claim', 'Evi_id', 'Evidence', 'Prob_sc', 'label'])
df_ts = pd.DataFrame(list(zip(claim_lb_list_ts, claim_list_ts, evi_label_list_ts, evi_list_ts, evi_score_list_ts, label_list_ts)),columns =['Claim_id', 'Claim', 'Evi_id', 'Evidence', 'Prob_sc', 'label'])
df_tr = pd.DataFrame(list(zip(claim_lb_list_tr, claim_list_tr, evi_label_list_tr, evi_list_tr, evi_score_list_tr, label_list_tr)),columns =['Claim_id', 'Claim', 'Evi_id', 'Evidence', 'Prob_sc', 'label'])

df.to_csv("./dev_file_filtered.csv")
df_ts.to_csv("./test_file_filtered.csv")
df_tr.to_csv("./train_file_filtered.csv")
