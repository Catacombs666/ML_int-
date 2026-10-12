"""Observed CTR reference model and safe review ranking; no future or causal label."""
from __future__ import annotations
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/raw/content_refresh_anonymized.csv'
SEED=42
SOURCE_REF='3b657dda40c1b668072a53371883fd4413d60f08'
NUM=['avg_position','log_word_count','content_age_days','days_since_last_update','log_search_volume','competition','log_cpc']
CAT=['content_type','main_intent']
FEATURES=NUM+CAT
BANNED={'ctr','clicks_90d','trend_pct','trend_direction','client_id','content_id','clicks_last_30d','clicks_prev_30d','sessions_90d','engagement_rate','scroll_rate','ai_traffic_pct'}
assert not set(FEATURES)&BANNED
BINS=[0,3,10,20,50,np.inf]
BAND_NAMES=['1–3','4–10','11–20','21–50','>50']

def read_data():
    raw=pd.read_csv(DATA)
    required={'content_id','client_id','impressions_90d','clicks_90d','avg_position',*CAT}
    assert required.issubset(raw.columns)
    assert raw.content_id.notna().all() and raw.client_id.notna().all()
    assert not raw.content_id.duplicated().any(), 'Content grain must be unique.'
    assert ((raw.clicks_90d>=0)&(raw.clicks_90d<=raw.impressions_90d)).all()
    volume=raw.impressions_90d>=500
    valid_position=raw.avg_position.notna()&(raw.avg_position>0)
    d=raw.loc[volume&valid_position].copy().reset_index(drop=True)
    d['target_ctr_pp']=100*d.clicks_90d/d.impressions_90d
    for src,dst in [('word_count','log_word_count'),('search_volume','log_search_volume'),('cpc','log_cpc')]:
        d[dst]=np.log1p(pd.to_numeric(d[src],errors='coerce').clip(lower=0))
    d['position_band']=pd.cut(d.avg_position,BINS,labels=BAND_NAMES)
    audit={'raw_rows':len(raw),'raw_clients':int(raw.client_id.nunique()),'raw_columns':len(raw.columns),'duplicate_content_rows':int(raw.content_id.duplicated().sum()),'excluded_below_500':int((~volume).sum()),'excluded_position_after_volume':int((volume&~valid_position).sum()),'eligible_rows':len(d),'eligible_clients':int(d.client_id.nunique()),'position_zero_raw':int((raw.avg_position==0).sum()),'ctr_rounding_max_pp':float(np.abs(raw.ctr-100*raw.clicks_90d/raw.impressions_90d).max()),'sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'source_git_blob':'2f2f41fdb07868dc5599449ce9d817f58f3b2ae9','source_commit':SOURCE_REF,'date_window':'Trailing 90-day snapshot; absolute export/end dates not present in the starter CSV. Warehouse dates are not assigned to this slice.','missing_pct_eligible':{c:round(float(d[c].isna().mean()*100),3) for c in ['word_count','search_volume','cpc','competition','main_intent']}}
    return d,audit

def split_clients(d):
    clients=np.array(sorted(d.client_id.unique()))
    order=np.random.default_rng(SEED).permutation(clients)
    nt=max(2,int(np.ceil(.2*len(order))))
    test_ids=set(order[:nt]); dev_ids=order[nt:]
    nv=max(2,int(np.ceil(.2*len(dev_ids))))
    validation_ids=set(dev_ids[:nv]); train_ids=set(dev_ids[nv:])
    indices={k:np.flatnonzero(d.client_id.isin(v)) for k,v in [('train',train_ids),('validation',validation_ids),('test',test_ids)]}
    assert train_ids.isdisjoint(validation_ids|test_ids) and validation_ids.isdisjoint(test_ids)
    assert sum(map(len,indices.values()))==len(d)
    counts={k:{'rows':len(v),'clients':int(d.iloc[v].client_id.nunique())} for k,v in indices.items()}
    return indices,counts

class PositionBaseline:
    def fit(self,d):
        total=float(d.clicks_90d.sum()/d.impressions_90d.sum())
        g=d.groupby('position_band',observed=False)[['clicks_90d','impressions_90d']].sum()
        self.rates=((g.clicks_90d+500*total)/(g.impressions_90d+500)*100).to_dict()
        self.global_pp=total*100
        return self
    def predict(self,d):
        return np.array([self.rates.get(b,self.global_pp) for b in d.position_band],dtype=float)

def make_model(name):
    preprocess=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),NUM),('category',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unknown')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)])
    if name=='random_forest':
        reg=RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=30,n_jobs=2,random_state=SEED)
    else:
        reg=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,min_samples_leaf=40,learning_rate=.06,l2_regularization=5,early_stopping=False,random_state=SEED)
    return Pipeline([('preprocess',preprocess),('model',reg)])

def metrics(d,p):
    p=np.clip(np.asarray(p),0,100); y=d.target_ctr_pp.to_numpy(); w=d.impressions_90d.to_numpy()
    return {'mae_pp':float(np.abs(p-y).mean()),'weighted_mae_pp':float(np.average(np.abs(p-y),weights=w)),'rmse_pp':float(np.sqrt(np.mean((p-y)**2))),'observed_ctr_pp':float(np.average(y,weights=w)),'predicted_ctr_pp':float(np.average(p,weights=w)),'n':len(d)}

def bootstrap_difference(d,selected,baseline,repeats=2000):
    temp=pd.DataFrame({'client':d.client_id,'weight':d.impressions_90d,'delta':(abs(selected-d.target_ctr_pp)-abs(baseline-d.target_ctr_pp))*d.impressions_90d})
    g=temp.groupby('client')[['weight','delta']].sum().to_numpy()
    rng=np.random.default_rng(SEED); draw=rng.integers(0,len(g),size=(repeats,len(g))); a=g[draw].sum(axis=1); dif=a[:,1]/a[:,0]
    return {'difference_selected_minus_baseline_pp':float(g[:,1].sum()/g[:,0].sum()),'client_bootstrap_95pct_interval_pp':np.quantile(dif,[.025,.975]).tolist(),'replicates':repeats,'clusters':len(g),'interpretation':'Exploratory client resampling with few clusters; negative favors selected method.'}

def build_queue(test,p):
    q=test.copy(); q['predicted_ctr_pp']=np.clip(p,0,100)
    q['reference_click_gap']=np.maximum(0,(q.predicted_ctr_pp-q.target_ctr_pp)*q.impressions_90d/100)
    q['priority_score']=100*q.reference_click_gap.rank(method='average',pct=True)
    q=q.sort_values(['reference_click_gap','content_id'],ascending=[False,True]).reset_index(drop=True)
    q['review_id']=[f'Review {i+1:02d}' for i in range(len(q))]
    q['action']=np.where(q.avg_position<=20,'Metadata and intent review','Visibility and content review')
    q['reason_codes']=q.apply(lambda r: '; '.join(['below_grouped_ctr_reference', 'visible_position_band' if r.avg_position<=20 else 'outside_page_two', 'word_count_missing' if pd.isna(r.word_count) else 'short_content_context' if r.word_count<1200 else 'content_depth_context', 'stale_context' if r.days_since_last_update>=180 else 'recent_update_context']),axis=1)
    q['evidence_level']=np.where((q.impressions_90d>=3000)&(q.clicks_90d>=20),'Higher measurement volume','Limited click evidence')
    records=[]
    for i,r in q.loc[q.reference_click_gap>0].head(20).iterrows():
        records.append({'rank':len(records)+1,'review_id':f'Review {len(records)+1:02d}','priority_score':round(float(r.priority_score),2),'reference_gap_clicks':round(float(r.reference_click_gap),1),'position_band':str(r.position_band),'action':r.action,'reason_codes':r.reason_codes.split('; '),'evidence_level':r.evidence_level})
    return q,records

def figures(result,calibration,importance,queue,out):
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    names=list(result['test']); vals=[result['test'][n]['weighted_mae_pp'] for n in names]
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh([n.replace('_',' ').title() for n in names],vals,color=['#748c9a','#13ad99','#d89542']); ax.set_xlabel('Impression-weighted CTR MAE (percentage points)'); ax.set_title('Held-out clients: all methods on the same pages'); fig.tight_layout(); fig.savefig(out/'model_comparison.svg'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(calibration['predicted_ctr_pp'],calibration['observed_ctr_pp'],'o-',color='#13ad99'); mx=max(calibration['predicted_ctr_pp'].max(),calibration['observed_ctr_pp'].max())*1.1; ax.plot([0,mx],[0,mx],'--',color='#748c9a'); ax.set(xlabel='Predicted weighted CTR (%)',ylabel='Observed weighted CTR (%)',title='Calibration by predicted-rate decile'); fig.tight_layout(); fig.savefig(out/'calibration.svg'); plt.close(fig)
    if importance:
        x=sorted(importance,key=lambda r:r['weighted_mae_increase_pp'])
        fig,ax=plt.subplots(figsize=(8,4.4)); ax.barh([r['feature'] for r in x],[r['weighted_mae_increase_pp'] for r in x],color='#13ad99'); ax.set(xlabel='Weighted MAE increase after shuffling (pp)',title='Held-out permutation sensitivity (association only)'); fig.tight_layout(); fig.savefig(out/'feature_sensitivity.svg'); plt.close(fig)
    g=queue.head(20).action.value_counts(); fig,ax=plt.subplots(figsize=(7,3)); ax.barh(g.index,g.values,color='#d89542'); ax.set(xlabel='Pages among the top 20 review candidates',title='Recommended review actions'); fig.tight_layout(); fig.savefig(out/'action_mix.svg'); plt.close(fig)

def run(output=None):
    output=Path(output or ROOT/'work/outputs'); output.mkdir(parents=True,exist_ok=True)
    d,audit=read_data(); idx,counts=split_clients(d)
    train,validation,test=(d.iloc[idx[k]].copy() for k in ['train','validation','test'])
    names=['position_baseline','random_forest','hist_gradient_boosting']
    validation_metrics={}; fitted={}
    for name in names:
        m=PositionBaseline().fit(train) if name=='position_baseline' else make_model(name)
        if name!='position_baseline': m.fit(train[FEATURES],train.target_ctr_pp,model__sample_weight=np.sqrt(train.impressions_90d))
        fitted[name]=m; p=m.predict(validation if name=='position_baseline' else validation[FEATURES]); validation_metrics[name]=metrics(validation,p)
    selected=min(names,key=lambda n:validation_metrics[n]['weighted_mae_pp'])
    development=pd.concat([train,validation]).sort_index(); predictions={}; models={}
    for name in names:
        m=PositionBaseline().fit(development) if name=='position_baseline' else make_model(name)
        if name!='position_baseline':m.fit(development[FEATURES],development.target_ctr_pp,model__sample_weight=np.sqrt(development.impressions_90d))
        models[name]=m; predictions[name]=np.clip(m.predict(test if name=='position_baseline' else test[FEATURES]),0,100)
    test_metrics={n:metrics(test,p) for n,p in predictions.items()}
    selected_p=predictions[selected]
    importance=[]
    if selected!='position_baseline':
        w=test.impressions_90d.to_numpy()
        def score(est,X,y):return -float(np.average(np.abs(est.predict(X)-y),weights=w))
        permutation=permutation_importance(models[selected],test[FEATURES],test.target_ctr_pp,n_repeats=5,random_state=SEED,scoring=score,n_jobs=2)
        importance=[{'feature':c,'weighted_mae_increase_pp':float(v),'sd_pp':float(sd)} for c,v,sd in zip(FEATURES,permutation.importances_mean,permutation.importances_std)]
    queue,recommendations=build_queue(test,selected_p)
    deciles=pd.qcut(pd.Series(selected_p,index=test.index).rank(method='first'),10,labels=False)
    calibration=[]
    for decile in range(10):
        mask=deciles.to_numpy()==decile; part=test.iloc[np.flatnonzero(mask)]; w=part.impressions_90d
        calibration.append({'decile':decile+1,'n':len(part),'predicted_ctr_pp':float(np.average(selected_p[mask],weights=w)),'observed_ctr_pp':float(np.average(part.target_ctr_pp,weights=w))})
    subgroups=[]
    for field in ['position_band','content_type']:
        for value,part in test.groupby(field,observed=True):
            if len(part)<20:continue
            loc=test.index.get_indexer(part.index)
            subgroups.append({'group_field':field,'group':str(value),'n':len(part),'selected_mae_pp':metrics(part,selected_p[loc])['weighted_mae_pp'],'baseline_mae_pp':metrics(part,predictions['position_baseline'][loc])['weighted_mae_pp']})
    sensitivity=[{'minimum_impressions':floor,**metrics(test.loc[test.impressions_90d>=floor],selected_p[test.impressions_90d.to_numpy()>=floor])} for floor in [500,1000,3000]]
    errors=np.abs(selected_p-test.target_ctr_pp.to_numpy()); worst=np.argsort(errors)[-3:][::-1]
    wrong_cases=[{'diagnostic_case':chr(65+i),'position_band':str(test.iloc[j].position_band),'observed_ctr_pp':round(float(test.iloc[j].target_ctr_pp),3),'predicted_ctr_pp':round(float(selected_p[j]),3),'explanation':'Large residual on an individual page; query mix, brand demand and snippet intent are unmeasured.'} for i,j in enumerate(worst)]
    result={'status':'measured','lane':'CTR / Engagement Opportunity Scoring','seed':SEED,'data':audit,'feature_columns':FEATURES,'label':'100 * clicks_90d / impressions_90d; observed same-window CTR, not a future label','validation_design':'Disjoint client sets: train, validation for method selection, sealed test; no row-random fallback','splits':counts,'selection_metric':'Validation impression-weighted CTR MAE; model selected before test','selected_method':selected,'validation':validation_metrics,'test':test_metrics,'bootstrap':bootstrap_difference(test,selected_p,predictions['position_baseline']),'subgroup_errors':subgroups,'volume_sensitivity':sensitivity,'permutation_importance':importance,'calibration':calibration,'recommendations':recommendations,'wrong_cases':wrong_cases,'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'leakage_checks':{'client_overlap':0,'content_overlap':0,'target_or_identifier_features':0,'preprocessing_fit_on_train_only':True,'winner_selected_on_validation_only':True,'future_forecast_claim':False},'interpretation':'Reference click gaps are review heuristics, not recoverable-click estimates or causal effects.'}
    (output/'ctr_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    cache=ROOT/'work/.cache';cache.mkdir(exist_ok=True); queue.to_csv(cache/'review_queue.csv',index=False)
    figures(result,pd.DataFrame(calibration),importance,queue,ROOT/'work/figures')
    print(json.dumps({'selected_method':selected,'eligible_rows':audit['eligible_rows'],'splits':counts,'test':test_metrics,'bootstrap':result['bootstrap']},indent=2))
    return result

if __name__=='__main__':run()
