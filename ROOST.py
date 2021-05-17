def read_mae(csv, n_ensemble):
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
    gs = [2,3,4]
    efls = [8,16,32,64]
    wds = [0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1] # define hyper-parameter space to screen
    best_error = 1000
    for g in gs:
        for efl in efls:
            for wd in wds:
                print (g,efl,wd)
                model_name = '%s_%s_%s'%(str(g),str(efl),str(wd))
                os.system('python roost-example.py --data-path formula_training_sample.csv --train --n-graph %d --elem-fea-len %d --weight-decay %f --model-name %s --ensemble %d --val-size 0.2'%(g, efl, wd, model_name, n_ensemble))
                os.system('python roost-example.py --data-path formula_training_sample.csv --evaluate --model-name %s --ensemble %d --test-size 0.2'%(model_name,n_ensemble))
                mae = read_mae('results/ensemble_results_%s.csv'%(model_name),n_ensemble)
                if mae < best_error:
                    best_para = [g,efl,wd,model_name,n_ensemble]
                    best_error = mae
    return (best_para)

def prediction(best_para):
    g,efl,wd,_,e_ensemble = best_para
    maes = []
    os.system('python roost-example.py --data-path formula_training_sample.csv --train --n-graph %d --elem-fea-len %d --weight-decay %f --model-name best --ensemble %d'%(g, efl, wd, n_ensemble))
    os.system('python roost-example.py --test-path formula_test_sample.csv --evaluate --model-name best --ensemble %d'%(n_ensemble))

n_ensemble = 1
best_para = best_paras(n_ensemble)
prediction(best_para)
