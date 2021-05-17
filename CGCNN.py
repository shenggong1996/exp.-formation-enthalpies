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
    n_convs = [2,3,4]
    afls = [8,16]
    nhs = [1,2]
    hls = [16,32]
    wds = [0,1e-5]
    best_error = 1000
    for ncv in n_convs:
        for afl in afls:
            for nh in nhs:
                for hl in hls:
                    for wd in wds:
                        print (ncv,afl,nh,hl,wd)
                        # Please revise the format if want to use GPU.
                        os.system('NV_GPU=%d nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mp_forma_energy/deep_learning_exp nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 main.py --optim Adam --epochs 500 --train-size %d --val-size %d --test-size %d --n-conv %d --atom-fea-len %d --n-h %d --h-fea-len %d --wd %f training_sample/'%(nh, int(train_size*0.6), int(train_size*0.2), int(train_size*0.2), ncv, afl, nh, hl, wd))
                        os.system('mv -f model_best.pth.tar %s/model_%d_%d_%d_%d_%s.pth.tar'%(direc,ncv,afl,nh,hl,str(wd)))
                        os.system('mv -f test_results.csv %s/test_results_%d_%d_%d_%d_%s.csv'%(direc,ncv,afl,nh,hl,str(wd)))
                        mae = read_mae('%s/test_results_%d_%d_%d_%d_%s.csv'%(direc,ncv,afl,nh,hl,str(wd)),int(train_size*0.2))
                        if mae < best_error:
                            best_para = [ncv,afl,nh,hl,wd]
                            best_error = mae
    return (best_para)
 
def prediction(best_para,direc,train_size):
    ncv,afl,nh,hl,wd = best_para
    maes = []
    for i in range(10):
        os.system('NV_GPU=%d nvidia-docker run     -v /home/shengg:/home/shengg     -v /data:/data     --rm -w /home/shengg/mp_forma_energy/deep_learning_exp nvidia/cuda:9.0-base /home/shengg/anaconda3/bin/python3 main.py --optim Adam --epochs 500 --train-size %d --val-size %d --test-size %d --n-conv %d --atom-fea-len %d --n-h %d --h-fea-len %d --wd %f training_sample/'%(nh, int(train_size*0.799), int(train_size*0.2), int(train_size*0.001), ncv, afl, nh, hl, wd))
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
length = 914 #total number of training samples
ratio = 1.0  #ratio of training samples used here
total_errors = []
splits = ['20210425']
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
