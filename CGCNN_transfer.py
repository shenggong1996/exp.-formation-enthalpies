import os
import numpy as np

def read_mae(csv, size):
    f=open(csv)
    maes = []
    for i in range(size):
        id, true, pred = f.readline().split(',')
        maes.append(abs(float(true)-float(pred)))
    return (np.mean(maes))

def best_paras(direc,train_size):
    #define hyperparameters space
    learning_rates = [1e-2,1e-3,1e-4,1e-5,1e-6]
    layers = [20,22,24,26,28] 
    best_error = 1000    
    for r in learning_rates:
        for layer in layers:
            maes = []
            for i in range(10):
                print (layer,r,i)
                os.system('NV_GPU=0 nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mp_forma_energy/transfer_learning nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 main.py  --optim Adam --epochs 500 --train-size %d --val-size %d --test-size %d --n-conv 4 --layersKept %d --lr %f --resume source/model_%d.pth.tar training_sample/'%(int(train_size*0.6), int(train_size*0.2), int(train_size*0.2), layer, r, i))
                os.system('mv -f model_best.pth.tar %s/model_%d_%s_%d.pth.tar'%(direc,layer,str(r),i))
                os.system('mv -f test_results.csv %s/test_results_%d_%s_%d.csv'%(direc,layer,str(r),i))
                mae = read_mae('%s/test_results_%d_%s_%d.csv'%(direc,layer,str(r),i),int(train_size*0.2))
                maes.append(mae)
            if np.mean(maes) < best_error:
                best_para = [layer,r]
                best_error = np.mean(maes)
    return (best_para)
  
def prediction(best_para,direc,train_size):
    layer,r = best_para
    maes = []
    for i in range(10):
        os.system('NV_GPU=0 nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mp_forma_energy/transfer_learning nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 main.py  --optim Adam --epochs 500 --train-size %d --val-size %d --test-size %d --n-conv 4 --layersKept %d --lr %f --resume source/model_%d.pth.tar training_sample/'%(int(train_size*0.799), int(train_size*0.2), int(train_size*0.001), layer, r, i))
        os.system('rm -f test_results.csv')
        os.system('python predict.py model_best.pth.tar test_sample')
        os.system('mv -f model_best.pth.tar %s/model_best_%d.pth.tar'%(direc,i))
        os.system('mv -f test_results.csv %s/test_results_best_%d.csv'%(direc,i))

def returnDict(length, category, n):
    f=open('%s/test_results_best_%d.csv'%(category,n))
    data = {}
    for i in range(length):
        id, true, pred = f.readline().split(',')
        data[id] = (float(true),float(pred))
    return (data)

test_length = 229
length = 914#total number of training samples
ratio = 1.0 #ratio of training samples used here, adjust this one for each case
total_errors = []
splits = ['20210426']
for sp in splits:
    best_para = best_paras(sp, int(length*ratio))
    prediction(best_para, sp, int(length*ratio))

for i in range(10):
    total = returnDict(test_length, splits[0], i)
    mae_total = []
    for id in total.keys():
        mae_total.append(abs(total[id][0] - total[id][1]))
    total_errors.append(np.mean(mae_total))

print (np.mean(total_errors), np.std(total_errors))
       
