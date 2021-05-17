import numpy as np
import pandas as pd
import os

def read_mae(csv, n_ensemble):#read mae from the ROOST output file
    df=pd.read_csv(csv)
    maes = []
    for index, row in df.iterrows():
        true = float(row[['target']])
        pred = 0
        for i in range(n_ensemble):
            pred += float(row[['pred_%d'%(i)]])/n_ensemble
        maes.append(abs(float(true)-float(pred)))
    return (np.mean(maes))

def best_paras(n_ensemble):
    ratios = [0,0.2,0.4,0.6,0.8]
    lrs = [9e-6,5e-5,3e-4]
    wds = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
    best_error = 1000
    for r in ratios:
        for lr in lrs:
            for wd in wds:
                maes=[]
                for i in range(5):
                    print (r,lr,wd,i)
                    model_name = '%s_%s_%s'%(str(r),str(lr),str(wd))
                    os.system('python roost-example.py --data-path formula_training_sample.csv --train  --transfer models/mp_source/best-r%d.pth.tar --transfer_ratio %f --lr %f --model-name %s --run-id %d --weight-decay %f --val-size 0.2'%(i, r, lr, model_name, i, wd))
                    os.system('python roost-example.py --data-path formula_training_sample.csv --evaluate --model-name %s --run-id %d --test-size 0.2'%(model_name,i))
                    mae = read_mae('results/test_results_%s_r-%d.csv'%(model_name,i),1)
                    maes.append(mae)
                if np.mean(maes) < best_error:
                    best_para = [r,lr,wd]
                    best_error = mae
    return (best_para)
  
  def prediction(best_para):
    r,lr,wd = best_para
    model_name = '%s_%s_%s'%(str(r),str(lr),str(wd))
    maes = []
    for i in range(5):
        os.system('python roost-example.py --data-path formula_training_sample.csv --train --transfer models/mp_source/best-r%d.pth.tar --transfer_ratio %f --lr %f --model-name best --run-id %d --weight-decay %f'%(i, r, lr, i, wd))
        os.system('python roost-example.py --test-path formula_test_sample.csv --evaluate --model-name best --run-id %d'%(i))
        mae = read_mae('results/test_results_best_r-%d.csv'%(i),1)
        maes.append(mae)
    print (np.mean(maes),np.std(maes))

n_ensemble = 1
best_para = best_paras(n_ensemble)
prediction(best_para)
