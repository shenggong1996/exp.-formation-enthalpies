import os
from math import *
from pymatgen import Structure
import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Initialize dataframe
def InitializeDF(source,csv): # source: where the cif structures is stored; csv: where the csv file with information of materials is stored.
    f = open('%s'%(csv))
    length = len(f.readlines())
    f.close()
    f = open('%s'%(csv))
    material_id = []
    formula = []
    structure = []
    diffs = []
    exps = []
    mps = []
    for i in range(length):
        if i == 0:
            f.readline()
            continue
        _,ID,exp,mp,diff = f.readline().split(',')
        diff=diff.replace('\n','')
        diff=float(diff)
        exp=float(exp)
        mp=float(mp)
        ID = ID.replace('\n','')
        ID = ID.replace('.0','')
        material_id.append(ID)
        struct = Structure.from_file('%s/%s.cif'%(source, ID))
        structure.append(struct)
        formula.append(struct.composition.reduced_formula)
        print (struct.composition.reduced_formula)
        diffs.append(diff)
        exps.append(exp)
        mps.append(mp)
    data = {}
    data['material_id'] = material_id
    data['formula'] = formula
    data['structure'] = structure
    data['diff'] = diffs
    data['exp'] = exps
    data['mp'] = mps
    df = pd.DataFrame(data)
    return (df)

def AddFeatures(df): # Add features by Matminer
    from matminer.featurizers.conversions import StrToComposition
    df = StrToComposition().featurize_dataframe(df, "formula")

    from matminer.featurizers.composition import ElementProperty

    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition")  # input the "composition" column to the featurizer

    from matminer.featurizers.conversions import CompositionToOxidComposition
    from matminer.featurizers.composition import OxidationStates

    df = CompositionToOxidComposition().featurize_dataframe(df, "composition")

    os_feat = OxidationStates()
    df = os_feat.featurize_dataframe(df, "composition_oxid")

    from matminer.featurizers.composition import ElectronAffinity

    ea_feat = ElectronAffinity()
    df = ea_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import BandCenter

    bc_feat = BandCenter()
    df = bc_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import CohesiveEnergy

    ce_feat = CohesiveEnergy()
    df = ce_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import Miedema

    m_feat = Miedema()
    df = m_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import TMetalFraction

    tmf_feat = TMetalFraction()
    df = tmf_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import ValenceOrbital

    vo_feat = ValenceOrbital()
    df = vo_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.composition import YangSolidSolution

    yss_feat = YangSolidSolution()
    df = yss_feat.featurize_dataframe(df, "composition_oxid",ignore_errors=True)

    from matminer.featurizers.structure import GlobalSymmetryFeatures

    # This is the border between compositional features and structural features. Comment out the following featurizers to use only compostional features.    
    
    gsf_feat = GlobalSymmetryFeatures()
    df = gsf_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import StructuralComplexity
    sc_feat = StructuralComplexity()
    df = sc_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import ChemicalOrdering
    co_feat = ChemicalOrdering()
    df = co_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import MaximumPackingEfficiency
    mpe_feat = MaximumPackingEfficiency()
    df = mpe_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import MinimumRelativeDistances
    mrd_feat = MinimumRelativeDistances()
    df = mrd_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import StructuralHeterogeneity
    sh_feat = StructuralHeterogeneity()
    df = sh_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.structure import SiteStatsFingerprint

    from matminer.featurizers.site import AverageBondLength
    from pymatgen.analysis.local_env import CrystalNN
    bl_feat = SiteStatsFingerprint(AverageBondLength(CrystalNN(search_cutoff=20)))
    df = bl_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.site import AverageBondAngle
    ba_feat = SiteStatsFingerprint(AverageBondAngle(CrystalNN(search_cutoff=20)))
    df = ba_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.site import BondOrientationalParameter
    bop_feat = SiteStatsFingerprint(BondOrientationalParameter())
    df = bop_feat.featurize_dataframe(df, "structure",ignore_errors=True)

    from matminer.featurizers.site import CoordinationNumber
    cn_feat = SiteStatsFingerprint(CoordinationNumber())
    df = cn_feat.featurize_dataframe(df, "structure", ignore_errors=True)

    from matminer.featurizers.structure import DensityFeatures
    df_feat = DensityFeatures()
    df = df_feat.featurize_dataframe(df, "structure", ignore_errors=True)
    return (df)

def best_parameters(X,y,i,arch): # The function to find the best hyper-parameters by grid search.
    
    if arch == 'random forest':  
        parameters = {'n_estimators':[100,200], 'max_depth':[None, 16, 64], 'min_samples_split':[2,4,8]}
        model = RandomForestRegressor(criterion = 'mae', random_state=i, verbose=True, n_jobs=5, warm_start= True)
        clf = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error',n_jobs=5,verbose=3) # Five-fold cross validation 
        clf.fit(X, y)
        print (clf.best_params_)
        return (clf.best_params_)
    
    if arch == 'mlp':
        depth = [3,4,5,6,7,8,9,10]
        width = [199,299,399]
        hidden = []
        for d in depth:
            for w in width:
                hidden.append(tuple([w]*d))
        parameters = {'hidden_layer_sizes':hidden, 'alpha':[1e-6, 1e-5,1e-4,1e-3,1e-2,1e-1]}
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, test_size=0.2) # For mlp we only split the validation set once due to the slow speed of mlp
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        best_error = 1e8
        for h in parameters['hidden_layer_sizes']:
            for a in parameters['alpha']:
                model = MLPRegressor(hidden_layer_sizes = h, alpha=a, random_state=i, activation='relu',max_iter=1000,tol=1e-15,shuffle=True)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                if mae < best_error:
                    best_error = mae
                    best_paras = {'hidden_layer_sizes':h, 'alpha':a}
        print (best_paras)
        return (best_paras)

arch = 'random forest' # Or 'mlp' to use mlp as the machine learning architecture    
    
df_train = InitializeDF('sample','y_train.csv')

df_train = AddFeatures(df_train)

df_train = df_train.fillna(0)

df_train.to_csv('train.csv')

df_train = pd.read_csv('train.csv')

X_train = df_train.select_dtypes(include=['float64','float32','int'])

X_train = X_train.fillna(0)

df_test = InitializeDF('sample','y_test.csv')

df_test = AddFeatures(df_test)

df_test = df_test.fillna(0)

df_test.to_csv('test.csv')

df_test = pd.read_csv('test.csv')

X_test = df_test.select_dtypes(include=['float64','float32','int'])

X_test = X_test.fillna(0)

y_train = df_train[['diff','exp','mp']]

y_test = df_test[['diff','exp','mp']]

excluded = ['Unnamed: 0','diff','exp','mp'] # To use the DFT formation enthalpy as an input, remove 'mp' from the "excluded" here.
X_train=X_train.drop(excluded, axis=1); X_test = X_test.drop(excluded, axis=1)

maes = []

for i in range(1,10): # repeat 10 times
    if arch == 'random forest':
        para_set = best_parameters(X_train, y_train[['diff']], i, arch) #Get the best hyper-parameters; if want to directly predict exp. formation enthalpy, then switch the training label to "y_train[['exp']]".
        n = para_set['n_estimators']; depth = para_set['max_depth']; split = para_set['min_samples_split']
        model = RandomForestRegressor(n_estimators=n, max_depth = depth, min_samples_split = split, criterion = 'mae', random_state=i, verbose=True, n_jobs=5)
        model.fit(X_train, y_train[['diff']]) 
        y_pred = model.predict(X_test)
        maes.append(mean_absolute_error(y_test[['diff']], y_pred))
        
    if arch == 'mlp':
        para_set = best_parameters(X_train, y_train[['diff']], i, arch)
        h = para_set['hidden_layer_sizes']; alpha = para_set['alpha']
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        model = MLPRegressor(hidden_layer_sizes = h,
                        activation = 'relu',
                        random_state=i,
                        max_iter=1000,
                        tol = 1e-15,
                        shuffle=True,
                        verbose=True)
        model.fit(X_train, y_train[['diff']])
        y_pred = model.predict(X_test)
        maes.append(mean_absolute_error(y_test[['diff']], y_pred))
        
print (np.mean(maes), np.std(maes))
