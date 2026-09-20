"""Build local, reduced-input release drafts without copying expression/count matrices.

Reads trusted local frozen checkpoints only. No upload or source-file modification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('NUMBA_CACHE_DIR', '/private/tmp/numba-release-cache')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/matplotlib-cache')
import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model/scMultiNODE'))
from dataset_protocols import GASTRULATION, PANCREAS, PALATE, HUMAN_CEREBRAL

SOURCES = {
 'gastrulation': ('GSE205117', 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205117'),
 'pancreas': ('GSE275562', 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275562'),
 'palate': ('GSE218576', 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE218576'),
 'human_cerebral': ('E-MTAB-12001; E-MTAB-11998; Zenodo 5242913', 'https://zenodo.org/records/5242913'),
 'scmultisim': ('scMultiSim; local 5scRNA simulation', 'https://github.com/ZhangLabGT/scMultiSim'),
}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
 return h.hexdigest()


def read_metadata(path):
 # Avoid loading multi-GB counts and malformed legacy layers.
 with h5py.File(path) as f:
  return read_elem(f['obs']), {k:read_elem(f['obsm'][k]) for k in f['obsm']}


def archive(path, keys):
 with np.load(path, allow_pickle=False) as z:
  return [np.asarray(z[k], dtype=np.float32) for k in keys]


def frozen_umap(path, raw, normalized):
 import joblib
 reducer=joblib.load(path)
 source=np.asarray(reducer._raw_data, dtype=np.float32)
 if source.shape != raw.shape: raise ValueError(f'UMAP input shape mismatch: {path}')
 errors={name:float(np.max(np.abs(source-x))) for name,x in [('raw',raw),('normalized',normalized)]}
 mode=min(errors,key=errors.get)
 if errors[mode]>1e-6: raise ValueError(f'UMAP row/value mismatch: {path}: {errors}')
 return np.asarray(reducer.embedding_,dtype=np.float32), {'kind':'frozen_benchmark_reducer','file':path.name,'sha256':sha(path),'input_space':mode,'max_abs_input_error':errors[mode]}


def build(args):
 out=args.output.resolve()
 if out.exists() and any(out.iterdir()): raise FileExistsError(f'Use a new empty output directory: {out}')
 out.mkdir(parents=True,exist_ok=True)
 records=[]
 configs=[('gastrulation',GASTRULATION,['gastrulation_rna_processed.h5ad','gastrulation_atac_processed.h5ad'],['X_pca','X_lsi']),
 ('pancreas',PANCREAS,['rna_adata.h5ad','atac_adata.h5ad'],['X_pca','X_poissonvi']),
 ('palate',PALATE,['rna_dimReduced.h5ad','atac_dimReduced.h5ad'],['X_pca','X_lsi_filtered']),
 ('human_cerebral',HUMAN_CEREBRAL,[],[])]
 for name,p,hfs,embedding_keys in configs:
  folder=args.source_root/p.data_subdirectory
  for mi,(mod,nf,sf,dim) in enumerate(p.modalities):
   blocks=archive(folder/nf,p.keys); raw=np.vstack(blocks)
   assert raw.shape[1]==dim
   ti=np.concatenate([np.full(len(b),i,dtype=np.int32) for i,b in enumerate(blocks)])
   scale=float(torch.load(folder/sf,map_location='cpu',weights_only=False)['scale'])
   assert np.isfinite(scale) and scale>0
   normalized=(raw/scale).astype(np.float32)
   extra={}; provenance=[folder/nf,folder/sf]
   if name!='human_cerebral':
    obs,emb=read_metadata(folder/hfs[mi]); provenance.append(folder/hfs[mi])
    idx=np.concatenate([np.flatnonzero(obs.stage.astype(str).to_numpy()==s) for s in p.stages])
    obs=obs.iloc[idx].copy()
    native=np.asarray(emb[embedding_keys[mi]][idx,:dim],dtype=np.float32)
    np.testing.assert_allclose(native,raw,rtol=0,atol=1e-6)
    coords=np.asarray(emb['X_umap'][idx],dtype=np.float32)
    umeta={'kind':'source_h5ad_embedding','file':hfs[mi],'key':'X_umap','row_alignment':'stage order + numerical representation equality','note':'Source visualization; may have been fit on a different dimensional representation.'}
    extra={k:np.asarray(v[idx],dtype=np.float32) for k,v in emb.items() if 'umap' in k.lower() and k!='X_umap'}
    if name=='palate':
     coords,umeta=frozen_umap(ROOT/'results/palate_strict_loo_dual_umap_tn_reversed'/f'{mod}_umap_model.joblib',raw,normalized)
    if name=='gastrulation' and mod=='atac':
     coords,umeta=frozen_umap(folder/'atac_umap_model_14D.joblib',raw,normalized)
   else:
    mp=ROOT/'data/human_cerebral_7time_d4_d21_no_d16_rna_cytobridge_balanced.h5ad'
    obs,emb=read_metadata(mp); provenance.append(mp)
    with np.load(folder/'paired_metacell_ids_by_time.npz',allow_pickle=True) as z:
     ids=np.concatenate([z[k].astype(str) for k in p.keys])
    np.testing.assert_array_equal(obs.index.to_numpy(str),ids)
    np.testing.assert_array_equal(obs.time_index.to_numpy(),ti)
    if mod=='rna':np.testing.assert_allclose(emb['X_pca_raw'],raw,rtol=0,atol=1e-6)
    cp=ROOT/'results/human_cerebral_original_metacell_umaps/umap_coordinates.csv.gz'
    c=pd.read_csv(cp);c=c[c.representation==f'Model {mod.upper()}']
    np.testing.assert_array_equal(c.id.to_numpy(str),ids)
    coords=c[['UMAP1','UMAP2']].to_numpy(np.float32);provenance.append(cp)
    umeta={'kind':'frozen_benchmark_coordinates','file':cp.name,'sha256':sha(cp),'representation':f'Model {mod.upper()}','row_alignment':'exact paired metacell IDs'}
   keep=[c for c in ['stage','celltype','celltype_sub','celltype_major','cell_type','cell_type_refined','lineage','lineage_coarse','line','age_day','n_cells_RNA','n_cells_ATAC','population'] if c in obs]
   obs=obs[keep].copy();obs['time_index']=ti;obs['time_point_processed']=np.asarray(p.times)[ti];obs['stage']=np.asarray(p.stages)[ti];obs['source_time_key']=np.asarray(p.keys)[ti]
   obs['row_in_time']=np.concatenate([np.arange(len(b)) for b in blocks])
   for split,indices in p.splits.items():obs[f'train_{split}']=np.isin(ti,indices)
   pairing='Computationally paired metacells, not experimentally paired cells.' if name=='human_cerebral' else 'Source cell IDs retained; RNA/ATAC row identity checked across exports.'
   write(out,name,mod,raw,normalized,obs,coords,extra,scale,umeta,provenance,p.representation_policy,pairing,records,args)
 # Simulation has generated row IDs, not original shared biological barcodes.
 name='scmultisim';folder=args.source_root/'Synthetic/5scRNA';keys=[f'time{i}' for i in range(1,5)]
 with np.load(folder/'all_time_scRNA_label.npz',allow_pickle=True) as z: labels=np.concatenate([z[k].astype(str) for k in keys])
 for mod,nf,sf in [('rna','all_time_scRNA_pca10.npz','primal_norm_params_rna10_w2.pt'),('atac','all_time_scATAC_pca8_no_pc2.npz','secondary_norm_params_atac8_no_pc2_w2.pt')]:
  blocks=archive(folder/nf,keys);raw=np.vstack(blocks);scale=float(torch.load(folder/sf,map_location='cpu',weights_only=False)['scale']);normalized=(raw/scale).astype(np.float32)
  ti=np.concatenate([np.full(len(b),i,dtype=np.int32) for i,b in enumerate(blocks)])
  obs=pd.DataFrame({'population':labels,'time_index':ti,'time_point_processed':ti.astype(float),'stage':np.asarray(keys)[ti],'source_time_key':np.asarray(keys)[ti],'row_in_time':np.concatenate([np.arange(len(b)) for b in blocks]),'train_full':True},index=[f'{k}_{i:05d}' for k,b in zip(keys,blocks) for i in range(len(b))])
  if mod=='rna':
   cache=ROOT/'results/cytobridge_synthetic_rna10_n128_i3000_comparison/rna10_umap/observed_rna10_umap.npz'
   with np.load(cache,allow_pickle=False) as z:
    np.testing.assert_array_equal(z['observed_population'],labels)
    assert int(z['input_dim'])==raw.shape[1]
    coords=z['observed_umap'].copy()
   umeta={'kind':'frozen_benchmark_coordinate_cache','file':cache.name,'sha256':sha(cache),'row_alignment':'Original producer concatenates time1..time4; population sequence and dimension checked. Cache has no cell IDs or input matrix; independent per-row identity cannot be verified.','producer':'scripts/fit_project_synthetic_rna10_umap.py','input_space':'raw RNA PCA10'}
  else:
   import umap
   reducer=umap.UMAP(n_neighbors=30,min_dist=.25,random_state=42,transform_seed=42,n_jobs=1)
   coords=reducer.fit_transform(normalized).astype(np.float32)
   umeta={'kind':'new_release_visualization','note':'New visualization of the exact normalized ATAC8 inputs; NOT a manuscript frozen UMAP.','n_neighbors':30,'min_dist':.25,'random_state':42,'metric':'euclidean','umap_version':umap.__version__}
  write(out,name,mod,raw,normalized,obs,coords,{},scale,umeta,[folder/nf,folder/sf,folder/'all_time_scRNA_label.npz'],'Frozen RNA PCA10 / ATAC PCs 1,3,4,5,6,7,8,9; t=0,1,2,3.','Synthetic paired per-time row order; generated IDs, not independently verified source barcodes.',records,args)
 for name in SOURCES:
  r=ad.read_h5ad(out/name/'rna.h5ad');a=ad.read_h5ad(out/name/'atac.h5ad')
  np.testing.assert_array_equal(r.obs_names,a.obs_names)
  np.testing.assert_array_equal(r.obs.time_point_processed,a.obs.time_point_processed)
 (out/'manifest.json').write_text(json.dumps({'status':'local draft; rights review pending; no upload performed','files':records},indent=2)+'\n')
 print(json.dumps(records,indent=2))


def write(out,name,mod,raw,normalized,obs,coords,extra,scale,umeta,provenance,policy,pairing,records,args):
 assert len(obs)==len(raw)==len(coords) and obs.index.is_unique
 assert all(np.isfinite(x).all() for x in [raw,normalized,coords])
 # X and X_latent are normalized; raw reduced values remain separately available.
 a=ad.AnnData(X=normalized,obs=obs,var=pd.DataFrame(index=[f'{mod}_component_{i+1}' for i in range(raw.shape[1])]))
 a.obsm['X_latent']=normalized.copy();a.obsm['X_reduced_raw']=raw;a.obsm['X_umap']=coords
 for k,v in extra.items():a.obsm[k]=v
 a.uns['benchmark']={'dataset':name,'modality':mod,'scale':scale,'normalization':'X = X_latent = X_reduced_raw / scale; divide exactly once. X is NOT gene expression or peak counts.','representation_policy':policy,'pairing':pairing,'source_accession':SOURCES[name][0],'source_url':SOURCES[name][1],'umap':umeta,'redistribution_status':'draft_pending_source_rights_review'}
 a.uns['source_files']={str(i):{'name':p.name,'sha256':sha(p)} for i,p in enumerate(provenance)}
 target=out/name/f'{mod}.h5ad';target.parent.mkdir(parents=True,exist_ok=True);a.write_h5ad(target,compression='gzip')
 b=ad.read_h5ad(target)
 np.testing.assert_array_equal(b.obsm['X_reduced_raw'],raw);np.testing.assert_array_equal(b.X,normalized);np.testing.assert_array_equal(b.obsm['X_umap'],coords);np.testing.assert_array_equal(b.obs_names,obs.index)
 records.append({'dataset':name,'modality':mod,'file':str(target.relative_to(out)),'n_obs':len(obs),'n_components':raw.shape[1],'bytes':target.stat().st_size,'sha256':sha(target),'umap':umeta})
 # Local-only source paths enable assembling a future Zenodo deposit without copying GBs now.
 with (out/'local_source_inventory.jsonl').open('a') as f:
  for p in provenance:f.write(json.dumps({'dataset':name,'modality':mod,'path':str(p.resolve()),'bytes':p.stat().st_size})+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(5,4));sc=ax.scatter(coords[:,0],coords[:,1],c=obs.time_point_processed,s=1,cmap='viridis',rasterized=True);fig.colorbar(sc,ax=ax,label='Model time');ax.set(xlabel='UMAP 1',ylabel='UMAP 2',title=f'{name} {mod.upper()}');fig.tight_layout();fig.savefig(target.with_name(f'{mod}_umap.png'),dpi=160);plt.close(fig)


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source-root',type=Path,default=ROOT.parent/'TraInf/TraInf')
 p.add_argument('--output',type=Path,default=ROOT/'data/release_draft')
 build(p.parse_args())
