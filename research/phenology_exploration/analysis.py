"""Independent grouped reconstruction, paired spatial uncertainty and diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import OUT,STAGES
from .core import macro,sha,valid_prediction

MODELS=['calendar','T','context_T','nonlinear_T','phase_T','residual_RF','residual_MLP','guarded_RF']


def site_errors(frame,error):
    q=frame[['head','site','site_year','block']].copy();q['error']=np.asarray(error)
    y=q.groupby(['head','site','block','site_year']).error.mean()
    return y.groupby(level=['head','site','block']).mean().reset_index()


def block_interval(frame,replicates=2000,seed=20261023):
    blocks=sorted(frame.block.unique());heads=sorted(frame['head'].unique())
    totals=[];counts=[]
    for h in heads:
        g=frame[frame['head'].eq(h)].groupby('block').error.agg(['sum','count']).reindex(blocks).fillna(0)
        totals.append(g['sum'].to_numpy());counts.append(g['count'].to_numpy())
    rng=np.random.default_rng(seed)
    draw=rng.multinomial(len(blocks),np.full(len(blocks),1/len(blocks)),size=replicates)
    numerator=draw@np.array(totals).T;denominator=draw@np.array(counts).T
    complete=(denominator>0).all(1)
    if complete.mean()<.95:return [np.nan,np.nan]
    # Never change the native-head population within a bootstrap replicate.
    values=(numerator[complete]/denominator[complete]).mean(1)
    return np.quantile(values,[.025,.975]).tolist()


def benefit_harm(frame,baseline,prediction):
    observed=frame.observed.to_numpy()
    a=np.where(valid_prediction(frame,baseline),abs(baseline-observed),366.)
    b=np.where(valid_prediction(frame,prediction),abs(prediction-observed),366.)
    gain=a-b
    benefit=macro(frame,np.maximum(gain,0));harm=macro(frame,np.maximum(-gain,0))
    return dict(benefit=benefit,harm=harm,net_gain=benefit-harm,
                improved_fraction=macro(frame,(gain>0).astype(float)),
                harmed_fraction=macro(frame,(gain<0).astype(float)))


def common_resolved(frame,models):
    return np.logical_and.reduce([valid_prediction(frame,frame[m].to_numpy()) for m in models])


def recipe_contrast(random,coverage):
    keys=['head','site','block']
    a=random.groupby(keys).error.mean().reset_index()
    b=coverage.groupby(keys).error.mean().reset_index()
    q=a.merge(b,on=keys,validate='one_to_one',suffixes=('_random','_coverage'))
    if len(q)!=len(a) or len(q)!=len(b):raise ValueError('Recipe target supports differ')
    q['error']=q.error_coverage-q.error_random;lo,hi=block_interval(q)
    return dict(difference=float(q.groupby('head').error.mean().mean()),ci_low=lo,ci_high=hi)


def run(partial=False):
    config=json.loads((OUT/'cases.json').read_text());sites=[];metrics=[];decomposition=[];parameters=[];receipts=[];order=[]
    for case in config['cases']:
        folder=OUT/'cases'/case['case'];done=folder/'complete.json'
        if not done.exists():
            if partial:continue
            raise ValueError(f'Incomplete frozen case {case["case"]}')
        receipt=json.loads(done.read_text())
        for name,digest in receipt['outputs'].items():
            if sha(folder/name)!=digest:raise ValueError('Changed case output')
        frame=pd.read_csv(folder/'predictions.csv.gz')
        assert frame.label_id.is_unique
        signature=receipt['signature'];assert signature['case']==case
        info=dict(experiment=case['case'],fold=case['fold'],recipe=case['recipe'],n_sites=case['n_sites'],
                  draw_seed=case['draw_seed'],source_ez=-1 if case['source_ez'] is None else case['source_ez'])
        for period,q in frame.groupby('case'):
            models=[c for c in frame if c in MODELS or c.startswith('residual_MLP_init')]
            common=common_resolved(q,models)
            for model in models:
                ok=valid_prediction(q,q[model]);error=np.where(ok,abs(q[model]-q.observed),366.)
                table=site_errors(q,error).assign(period=period,model=model,**info)
                table['difference_vs_T']=site_errors(q,error-np.where(valid_prediction(q,q['T']),abs(q['T']-q.observed),366.)).error
                table=table.merge(q[['site','ez']].drop_duplicates(),on='site',validate='many_to_one')
                sites.append(table)
                value=table.groupby('head').error.mean().mean()
                metrics.append(dict(period=period,model=model,mae=float(value),completion=float(ok.mean()),
                                    common_resolved_mae=macro(q.loc[common],abs(q.loc[common,model]-q.loc[common,'observed'])) if common.any() else None,
                                    common_resolved_labels=int(common.sum()),
                                    labels=len(q),target_sites=q.site.nunique(),**info))
                piv=q.assign(value=np.where(ok,q[model],np.nan)).pivot(index='unit',columns='head',values='value').reindex(columns=range(7))
                pairs=piv.notna().to_numpy()[:,:-1]&piv.notna().to_numpy()[:,1:]
                values=piv.to_numpy();reversals=(values[:,:-1]>values[:,1:])&pairs
                order.append(dict(period=period,model=model,valid_adjacent_pairs=int(pairs.sum()),reversed_pairs=int(reversals.sum()),
                                  coincident_pairs=int(((values[:,:-1]==values[:,1:])&pairs).sum()),**info))
                decomposition.append(dict(period=period,model=model,**benefit_harm(q,q['T'].to_numpy(),q[model].to_numpy()),**info))
                if case['source_ez'] is not None:
                    for ez,z in q.groupby('ez'):
                        err=np.where(valid_prediction(z,z[model]),abs(z[model]-z.observed),366.)
                        metrics.append(dict(period=period,model=model,mae=macro(z,err),completion=float(valid_prediction(z,z[model]).mean()),
                                            labels=len(z),target_sites=z.site.nunique(),target_ez=int(ez),**info))
        for model,trials in receipt['selection_trials'].items():
            if model=='context_T':winner=min(trials,key=lambda t:(t['mae'],-t['parameter']))
            elif model in ('nonlinear_T','phase_T'):winner=min(trials,key=lambda t:(t['mae'],abs(t['parameter']-1)))
            elif model=='guarded_RF':winner=min(trials,key=lambda t:(t['mae'],t['strength'],t['k']))
            elif model=='residual_MLP':winner=min(trials,key=lambda t:(t['mae'],t['strength'],t['steps']))
            else:winner=min(trials,key=lambda t:(t['mae'],t['strength']))
            parameters.append(dict(model=model,**winner,**info))
        receipts.append(dict(case=case['case'],seconds=receipt['elapsed_seconds'],
                             target_sha256=receipt['target_label_sha256']))
    if not sites:raise ValueError('No completed experiments')
    table=pd.concat(sites,ignore_index=True);m=pd.DataFrame(metrics);d=pd.DataFrame(decomposition)
    table.to_csv(OUT/'site_errors.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    m.to_csv(OUT/'case_metrics.csv',index=False);d.to_csv(OUT/'benefit_harm.csv',index=False)
    pd.DataFrame(parameters).to_csv(OUT/'source_selected_settings.csv',index=False)
    pd.DataFrame(order).to_csv(OUT/'stage_order_audit.csv',index=False)
    grouping=['period','recipe','n_sites','source_ez','model'];draws=[]
    for keys,q in table.groupby(grouping+['draw_seed']):
        value=q.groupby('head').error.mean().mean()
        draws.append(dict(zip(grouping+['draw_seed'],keys),mae=float(value)))
    draws=pd.DataFrame(draws);draws.to_csv(OUT/'pooled_draw_metrics.csv',index=False)
    summary=[];stages=[]
    for keys,q in table.groupby(grouping):
        meta=dict(zip(grouping,keys))
        # Mean errors over draws, never an ensemble of predictions.
        average=q.groupby(['head','site','block'])[['error','difference_vs_T']].mean().reset_index()
        lo,hi=block_interval(average)
        contrast=average.rename(columns={'error':'absolute_error','difference_vs_T':'error'})
        dlo,dhi=block_interval(contrast)
        select=np.ones(len(draws),bool)
        for key,value in meta.items():select &=draws[key].eq(value)
        values=draws.loc[select,'mae']
        summary.append(dict(**meta,mae=float(average.groupby('head').error.mean().mean()),
                            ci_low=lo,ci_high=hi,difference_vs_T=float(average.groupby('head').difference_vs_T.mean().mean()),
                            difference_ci_low=dlo,difference_ci_high=dhi,source_draw_sd=float(values.std()) if len(values)>1 else None,
                            draws=len(values),folds=q.fold.nunique(),sites=average.site.nunique()))
        for head,h in average.groupby('head'):
            hlo,hhi=block_interval(h)
            stages.append(dict(**meta,head=int(head),stage=STAGES[int(head)],mae=float(h.error.mean()),
                               difference_vs_T=float(h.difference_vs_T.mean()),ci_low=hlo,ci_high=hhi))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(stages).to_csv(OUT/'stage_summary.csv',index=False)
    matrix=[]
    for (period,source_ez,target_ez,model),q in table[table.source_ez.ge(0)].groupby(['period','source_ez','ez','model']):
        lo,hi=block_interval(q)
        matrix.append(dict(period=period,source_ez=int(source_ez)+1,target_ez=int(target_ez)+1,model=model,
            mae=float(q.groupby('head').error.mean().mean()),difference_vs_T=float(q.groupby('head').difference_vs_T.mean().mean()),
            ci_low=lo,ci_high=hi,sites=q.site.nunique(),blocks=q.block.nunique(),folds=q.fold.nunique()))
    pd.DataFrame(matrix).to_csv(OUT/'zone_matrix_summary.csv',index=False)
    contrasts=[]
    primary=table[table.source_ez.eq(-1)]
    for (period,budget,model),q in primary.groupby(['period','n_sites','model']):
        r=q[q.recipe.eq('random')];c=q[q.recipe.eq('coverage')]
        contrasts.append(dict(period=period,n_sites=int(budget),model=model,**recipe_contrast(r,c)))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_recipe_contrasts.csv',index=False)
    pd.DataFrame(receipts).to_csv(OUT/'case_receipts.csv',index=False)
    audit=dict(complete=not partial,completed_cases=len(receipts),frozen_cases=len(config['cases']),
        unsupported_cases=config['skipped'],grouped_metric_reconstruction=True,
        uncertainty='2,000 paired spatial-block resamples of draw-averaged errors; frozen fits, not refit uncertainty',
        bootstrap_scope='target sampling; source-draw SD and neural-initialization variability are separate',
        files={p.name:sha(p) for p in OUT.glob('*.csv*')})
    (OUT/'analysis_verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(summary[summary.source_ez.eq(-1)&summary.period.eq('later_held')&summary.model.isin(MODELS)]
          [['recipe','n_sites','model','mae','difference_vs_T','source_draw_sd']].round(3).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args();run(args.partial)
