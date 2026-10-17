"""Standalone Python runner for the RMIT Hackathon Kaggle competition.

Run the TF-IDF baseline: python kaggle_train.py
Fine-tune a jailbreak detector on GPU: python kaggle_train.py --transformer
Change transformer training length: python kaggle_train.py --transformer --epochs 6
"""
# %% Imports and competition files

# %% Kaggle workflow cell 1
from pathlib import Path
import json, random, time, inspect
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT=Path('/kaggle/input/competitions/rmit-hackathon-2025')
OUT=Path('/kaggle/working'); OUT.mkdir(parents=True,exist_ok=True)
csvs=sorted(ROOT.rglob('*.csv'))
print('CSV files:',*csvs,sep='\n')
def find(names):
    for name in names:
        hits=[p for p in csvs if p.name.lower()==name.lower()]
        if hits: return hits[0]
    raise FileNotFoundError(f'File not found: {names}. Set its path manually.')
TRAIN_PATH=find(['train.csv','training.csv'])
TEST_PATH=find(['test.csv'])
SAMPLE_PATH=find(['sample_submission.csv','sample-submission.csv'])
train_df,test_df,sample_df=[pd.read_csv(p) for p in [TRAIN_PATH,TEST_PATH,SAMPLE_PATH]]
print('train',train_df.shape,train_df.columns.tolist())
print('test',test_df.shape,test_df.columns.tolist())
print('sample',sample_df.shape,sample_df.columns.tolist())

# %% Kaggle workflow cell 2
# Edit after checking the schemas printed above.
TEXT_COL,TARGET_COL='text','label'
ID_COL=next((c for c in sample_df.columns if str(c).strip().lower()=='id'),None)
if ID_COL is None: raise ValueError(f'Could not identify ID column in sample submission: {sample_df.columns.tolist()}')
TARGET_SUB_COL=next((c for c in sample_df.columns if str(c).strip().lower()=='target'),None)
if TARGET_SUB_COL is None: raise ValueError(f'Could not identify target column in sample submission: {sample_df.columns.tolist()}')
for df,name,cols in [(train_df,'train',[TEXT_COL,TARGET_COL]),(test_df,'test',[TEXT_COL,ID_COL]),(sample_df,'sample',[ID_COL,TARGET_SUB_COL])]:
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError(f'{name} missing {missing}; fix column config.')
if train_df[TARGET_COL].isna().any():
    raise ValueError('Training labels must not contain nulls.')
for df,name in [(test_df,'test'),(sample_df,'sample')]:
    if df[ID_COL].isna().any() or df[ID_COL].duplicated().any(): raise ValueError(f'{name} IDs must be unique/non-null.')
print('Rows:',len(train_df),len(test_df),len(sample_df))
print('Classes:',train_df[TARGET_COL].value_counts().sort_index().to_dict())
print('Duplicate train prompts:',int(train_df[TEXT_COL].dropna().duplicated().sum()))
print('Text length summary:',train_df[TEXT_COL].dropna().astype(str).str.len().describe().to_dict())

SEED,FRAC=42,0.20
texts=['' if pd.isna(x) else str(x) for x in train_df[TEXT_COL]]
raw_labels=train_df[TARGET_COL]
if pd.api.types.is_numeric_dtype(raw_labels):
    if not set(raw_labels.unique()).issubset({0,1}):
        raise ValueError(f'Expected labels 0/1 or benign/jailbreak; found {raw_labels.unique()[:10]}')
    labels=raw_labels.to_numpy(dtype=np.int64)
else:
    label_map={'benign':0,'jailbreak':1}
    normalized=raw_labels.astype(str).str.strip().str.lower()
    unknown=sorted(set(normalized)-set(label_map))
    if unknown: raise ValueError(f'Unrecognized labels {unknown}; expected benign/jailbreak.')
    labels=normalized.map(label_map).to_numpy(dtype=np.int64)
if set(np.unique(labels))!={0,1}: raise ValueError('Training data must contain both classes.')
print('Encoded class counts (benign=0, jailbreak=1):',pd.Series(labels).value_counts().sort_index().to_dict())
test_texts=['' if pd.isna(x) else str(x) for x in test_df[TEXT_COL]]
gm,groups={},[]
for i,t in enumerate(texts):
    key=('missing',i) if not t else ('text',t)
    gm.setdefault(key,len(gm)); groups.append(gm[key])
groups=np.asarray(groups)
n=min(max(2,round(1/FRAC)),int(np.unique(labels,return_counts=True)[1].min()),len(np.unique(groups)))
if n<2: raise ValueError('Insufficient class examples/prompt groups for validation.')
folds=[]
for tr,va in StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=SEED).split(np.zeros(len(labels)),labels,groups):
    if len(np.unique(labels[va]))==2:
        err=abs(len(va)/len(labels)-FRAC)+abs(float(labels[va].mean()-labels.mean()))
        folds.append((err,tr,va))
if not folds: raise ValueError('No validation fold contains both classes.')
_,tr_idx,va_idx=min(folds,key=lambda x:x[0])
assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))
print('Split rows:',len(tr_idx),'train /',len(va_idx),'validation')

def build(mode='word',c=4.,classifier='logistic'):
    word=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
    char=TfidfVectorizer(analyzer='char_wb',ngram_range=(2,5),min_df=1,sublinear_tf=True)
    feat={'word':word,'char':char,'combined':FeatureUnion([('word',word),('char',char)])}[mode]
    if classifier=='logistic':
        estimator=LogisticRegression(C=c,max_iter=2000,class_weight='balanced')
    elif classifier=='linear_svc':
        estimator=LinearSVC(C=c,class_weight='balanced',max_iter=10000,random_state=SEED)
    else:
        raise ValueError(f'Unknown classifier: {classifier}')
    return Pipeline([('features',feat),('classifier',estimator)])
def score(model,x):
    classifier=model.named_steps['classifier']
    if hasattr(classifier,'predict_proba'):
        return model.predict_proba(x)[:,list(model.classes_).index(1)]
    return model.decision_function(x)
def probability(model,x):
    classifier=model.named_steps['classifier']
    if hasattr(classifier,'predict_proba'):
        return score(model,x)
    return 1/(1+np.exp(-np.clip(score(model,x),-40,40)))
base=build('word',4.,'logistic').fit([texts[i] for i in tr_idx],labels[tr_idx])
base_auc=roc_auc_score(labels[va_idx],score(base,[texts[i] for i in va_idx]))
rows=[]
for classifier in ['logistic','linear_svc']:
    for c in [.5,2.,4.]:
        for mode in ['word','char','combined']:
            start=time.perf_counter(); model=build(mode,c,classifier).fit([texts[i] for i in tr_idx],labels[tr_idx])
            auc=roc_auc_score(labels[va_idx],score(model,[texts[i] for i in va_idx]))
            rows.append({'classifier':classifier,'mode':mode,'C':c,'roc_auc':float(auc),'seed':SEED,'seconds':round(time.perf_counter()-start,3)})
experiments=pd.DataFrame(rows).sort_values('roc_auc',ascending=False)
experiments.to_csv(OUT/'experiments.csv',index=False); print(experiments.to_string(index=False))
best=experiments.iloc[0]
final=build(str(best['mode']),float(best['C']),str(best['classifier'])).fit(texts,labels)
pred=probability(final,test_texts)
if len(pred)!=len(sample_df) or not np.isfinite(pred).all() or ((pred<0)|(pred>1)).any():
    raise ValueError('Invalid prediction length/probabilities.')
if not np.array_equal(test_df[ID_COL].to_numpy(),sample_df[ID_COL].to_numpy()):
    raise ValueError('Test IDs/order do not match sample submission.')
submission=sample_df.copy(); submission[TARGET_SUB_COL]=pred
submission.to_csv(OUT/'submission.csv',index=False)
report={'baseline_auc':float(base_auc),'best_classifier':str(best['classifier']),'best_mode':str(best['mode']),'best_C':float(best['C']),
        'best_auc':float(best['roc_auc']),'seed':SEED,'train_rows':len(tr_idx),'validation_rows':len(va_idx)}
(OUT/'baseline_report.json').write_text(json.dumps(report,indent=2))
print('Saved /kaggle/working/submission.csv'); print(submission.head().to_string(index=False))

# %% Kaggle workflow cell 4
import argparse

parser=argparse.ArgumentParser(description='Train RMIT jailbreak classifier and write Kaggle submissions.')
parser.add_argument('--transformer',action='store_true',help='Fine-tune the jailbreak-specialized Transformer on the enabled GPU.')
parser.add_argument('--epochs',type=int,default=10,help='Maximum Transformer epochs (early stopping selects the best validation epoch).')
parser.add_argument('--model-id',default='madhurjindal/Jailbreak-Detector',help='Hugging Face sequence-classification checkpoint.')
cli_args,_=parser.parse_known_args()
RUN_TRANSFORMER=cli_args.transformer
# Start from a jailbreak-specialized classifier instead of a generic MLM checkpoint.
MODEL_ID,MODEL_LICENSE=cli_args.model_id,'mit' if cli_args.model_id=='madhurjindal/Jailbreak-Detector' else 'see model card'
if RUN_TRANSFORMER:
    print(f'Fine-tuning Transformer: {MODEL_ID} (max epochs={cli_args.epochs}); this runs in addition to the TF-IDF baseline.')
    import torch
    from scipy.special import softmax
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,DataCollatorWithPadding,Trainer,TrainingArguments,EarlyStoppingCallback
    if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU accelerator.')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self,x,y=None):
            # Let DataCollatorWithPadding pad each batch to its longest prompt.
            self.enc=tok(x,truncation=True,max_length=512,padding=False)
            self.y=None if y is None else np.asarray(y,dtype=np.int64)
        def __len__(self): return len(self.enc['input_ids'])
        def __getitem__(self,i):
            row={k:torch.tensor(v[i]) for k,v in self.enc.items()}
            if self.y is not None: row['labels']=torch.tensor(self.y[i],dtype=torch.long)
            return row
    tr_ds=TextDataset([texts[i] for i in tr_idx],labels[tr_idx])
    va_ds=TextDataset([texts[i] for i in va_idx],labels[va_idx])
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_ID,num_labels=2)
    def metric(p):
        logits=p.predictions[0] if isinstance(p.predictions,tuple) else p.predictions
        return {'roc_auc':roc_auc_score(p.label_ids,softmax(logits,axis=1)[:,1])}
    key='eval_strategy' if 'eval_strategy' in inspect.signature(TrainingArguments).parameters else 'evaluation_strategy'
    args=TrainingArguments(output_dir='/kaggle/working/distilbert-checkpoints',
        save_strategy='epoch',logging_strategy='epoch',load_best_model_at_end=True,
        metric_for_best_model='eval_roc_auc',greater_is_better=True,save_total_limit=1,
        num_train_epochs=cli_args.epochs,learning_rate=1e-5,per_device_train_batch_size=8,
        per_device_eval_batch_size=16,gradient_accumulation_steps=2,
        weight_decay=.01,fp16=True,seed=SEED,report_to='none',
        **{key:'epoch'})
    trainer=Trainer(model=model,args=args,train_dataset=tr_ds,eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tok),compute_metrics=metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
    started=time.perf_counter(); trainer.train()
    va_logits=trainer.predict(va_ds).predictions
    auc=roc_auc_score(labels[va_idx],softmax(va_logits,axis=1)[:,1])
    te_logits=trainer.predict(TextDataset(test_texts)).predictions
    tr_pred=sample_df.copy(); tr_pred[TARGET_SUB_COL]=softmax(te_logits,axis=1)[:,1]
    tr_pred.to_csv(OUT/'submission_transformer.csv',index=False)
    # Make the main submission file correspond to the model just trained, so it
    # is the obvious file to download when the script is run with --transformer.
    tr_pred.to_csv(OUT/'submission.csv',index=False)
    tr_report={'model_id':MODEL_ID,'license':MODEL_LICENSE,'device':'cuda','seed':SEED,
        'transformer_auc':float(auc),'best_linear_auc':float(best['roc_auc']),
        'auc_delta_vs_best_linear':float(auc-best['roc_auc']),
        'runtime_seconds':round(time.perf_counter()-started,2),
        'public_leaderboard_auc':None,'remaining_submission_budget':None}
    (OUT/'transformer_report.json').write_text(json.dumps(tr_report,indent=2))
    print(json.dumps(tr_report,indent=2))
    print('Saved /kaggle/working/submission.csv and /kaggle/working/submission_transformer.csv')
    print(tr_pred.head().to_string(index=False))
