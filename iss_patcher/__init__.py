"""Approximate missing features from higher dimensionality data neighbours"""
__version__ = "0.1.0"

import scipy.spatial
import scipy.sparse
import anndata
import pandas as pd
import scanpy as sc
import annoy
import pynndescent
import numpy as np

def prepare_scaled(adata, min_genes=3):
    sc.pp.filter_cells(adata, min_genes=min_genes)
    #normalise to median
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

def get_knn_indices(iss, gex,
                    computation="annoy",
                    neighbours=15
                   ):
    if computation == "annoy":
        #build the GEX index
        ckd = annoy.AnnoyIndex(gex.X.shape[1], metric="euclidean")
        for i in np.arange(gex.X.shape[0]):
            ckd.add_item(i,gex.X[i,:])
        ckd.build(10)
        #query the GEX index with the ISS data
        ckdo_ind = []
        ckdo_dist = []
        for i in np.arange(iss.X.shape[0]):
            holder = ckd.get_nns_by_vector(iss.X[i,:], neighbours, include_distances=True)
            ckdo_ind.append(holder[0])
            ckdo_dist.append(holder[1])
        ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
    elif computation == "pynndescent":
        #build the GEX index
        ckd = pynndescent.NNDescent(gex.X, metric="euclidean", n_jobs=-1, random_state=0)
        ckd.prepare()
        #query the GEX index with the ISS data
        ckdout = ckd.query(iss.X, k=neighbours)
        #need to reverse this to match conventions
        ckdout = (ckdout[1], ckdout[0])
    elif computation == "cKDTree":
        #build the GEX index
        ckd = scipy.spatial.cKDTree(gex.X)
        #query the GEX index with the ISS data
        ckdout = ckd.query(x=iss.X, k=neighbours, workers=-1)
    else:
        raise ValueError("Invalid computation, must be 'annoy', 'pynndescent' or 'cKDTree'")
    return ckdout

def ckdout_to_sparse(ckdout, shape,
                     neighbours=15
                    ):
    #the indices need to be flattened, the default row-major style works
    indices = ckdout[1].flatten()
    #the indptr is once every neighbours, but needs an extra entry at the end
    indptr = neighbours * np.arange(shape[0]+1)
    #the data is ones. for now. use float32 as that's what scanpy likes as default
    data = np.ones(shape[0]*neighbours, dtype=np.float32)
    #construct the KNN graph!
    #need to specify the shape as there may be cells at the end that don't get picked up
    #and this will throw the dimensions off when doing matrix operations shortly
    pbs = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
    return pbs

def get_pbs_obs(iss, gex, pbs,
                obs_to_take=None,
                obsm_fraction=False
               ):
    #start the obs pool with what already resides in the ISS object
    pbs_obs = iss.obs.copy()
    #possibly store all computed fractions too, will live in obsm later
    if obsm_fraction:
        pbs_obsm = {}
    if obs_to_take is not None:
        #just in case a single is passed as a string
        if type(obs_to_take) is not list:
            obs_to_take = [obs_to_take]
        #now we can iterate over this nicely
        #using the logic of milopy's annotate_nhoods()
        for anno_col in obs_to_take:
            anno_dummies = pd.get_dummies(gex.obs[anno_col])
            anno_count = pbs.dot(anno_dummies.values)
            #apparently an np.array falls out from the above
            #which then in turn needs [:,None] here for the division to work properly
            anno_frac = np.array(anno_count / anno_count.sum(1)[:,None])
            anno_frac = pd.DataFrame(
                anno_frac,
                index=iss.obs_names,
                columns=anno_dummies.columns,
            )
            pbs_obs[anno_col] = anno_frac.idxmax(1)
            pbs_obs[anno_col + "_fraction"] = anno_frac.max(1)
            #possibly stash full thing for obsm insertion later
            if obsm_fraction:
                pbs_obsm[anno_col] = anno_frac.copy()
    if obsm_fraction:
        return pbs_obs, pbs_obsm
    else:
        return pbs_obs

def get_pbs_X(gex_only, pbs,
              round_counts=True,
              chunk_size=100000,
              neighbours=15
             ):
    #make a copy to not modify the original pbs
    pbs = pbs.copy()
    #the expression is a mean rather than a sum, make the data to add up to one per row
    pbs.data = (pbs.data / neighbours)
    #if we're rounding the data, compute it in chunks to reduce RAM footprint
    if round_counts:
        #we'll be vstacking to this shortly
        X = None
        #process chunk_size iss cells at a time
        for start_pos in np.arange(0, pbs.shape[0], chunk_size):
            #these are our pseudobulk definitions for this chunk
            pbs_sub = pbs[start_pos:(start_pos+chunk_size),:]
            #get the corresponding expression for the chunk
            X_sub = pbs_sub.dot(gex_only.X)
            #round the data to nearest integer
            X_sub.data = np.round(X_sub.data)
            X_sub.eliminate_zeros()
            #store the rounded data in a master matrix
            if X is None:
                X = X_sub.copy()
            else:
                X = scipy.sparse.vstack([X,X_sub])
    else:
        #no rounding, so no RAM footprint to be saved
        #just do it all at once
        X = pbs.dot(gex_only.X)
    return X

def knn(iss, gex, gex_only, 
        obs_to_take=None, 
        round_counts=True, 
        chunk_size=100000, 
        computation="annoy",  
        neighbours=15, 
        obsm_fraction=False,
        obsm_pbs=False
       ):
    #identify the KNN, preparing a (distances, indices) tuple
    ckdout = get_knn_indices(iss=iss,
                             gex=gex,
                             computation=computation,
                             neighbours=neighbours
                            )
    #turn KNN output into a scanpy-like graph
    pbs = ckdout_to_sparse(ckdout=ckdout,
                           shape=[iss.shape[0], gex.shape[0]],
                           neighbours=neighbours
                          )
    #get the annotations and fractions of the specified obs columns in the KNN
    if obsm_fraction:
        pbs_obs, pbs_obsm = get_pbs_obs(iss=iss,
                                        gex=gex,
                                        pbs=pbs,
                                        obs_to_take=obs_to_take,
                                        obsm_fraction=obsm_fraction
                                       )
    else:
        pbs_obs = get_pbs_obs(iss=iss,
                              gex=gex,
                              pbs=pbs,
                              obs_to_take=obs_to_take,
                              obsm_fraction=obsm_fraction
                             )
    #get the expression matrix
    X = get_pbs_X(gex_only=gex_only,
                  pbs=pbs,
                  round_counts=round_counts,
                  chunk_size=chunk_size,
                  neighbours=neighbours
                 )
    #now we can build the object easily
    out = anndata.AnnData(X, var=gex_only.var, obs=pbs_obs)
    #shove in the fractions from earlier if we need to
    if obsm_fraction:
        for anno_col in pbs_obsm:
            out.obsm[anno_col+"_fraction"] = pbs_obsm[anno_col]
    #shove in the pbs (KNN) if we need to
    if obsm_pbs:
        out.obsm['pbs'] = pbs
    return out

def patch(iss, gex, 
          min_genes=3, 
          obs_to_take=None, 
          round_counts=True, 
          chunk_size=100000, 
          computation="annoy", 
          neighbours=15, 
          obsm_fraction=False,
          obsm_pbs=False
         ):
    """
    Identify the nearest neighbours of low dimensionality observations 
    in related higher dimensionality data. Approximate features absent  
    from the low dimensionality data as high dimensionality neighbour 
    means. The data is log-normalised and z-scored prior to KNN 
    inference.
    
    Input
    -----
    iss : ``AnnData``
        The low dimensionality data object, with raw counts in ``.X``.
    gex : ``AnnData``
        The high dimensionality data object, with raw counts in ``.X``.
    min_genes : ``int``, optional (default: 3)
        Passed to ``scanpy.pp.filter_cells()`` ran on the shared feature 
        space of ``iss`` and ``gex``.
    obs_to_take : ``str`` or list of ``str``, optional (default: ``None``)
        If provided, will report the most common value of the specified 
        ``gex.obs`` column(s) for the neighbours of each ``iss`` cell.
    round_counts : ``bool``, optional (default: ``True``)
        If ``True``, will round the computed counts to the nearest 
        integer.
    chunk_size : ``int``, optional (default: 100000)
        If ``round_counts`` is ``True``, will compute ``iss`` profiles 
        these many observations at a time and round them to reduce RAM 
        use. A larger value means fewer matrix operations (i.e. quicker 
        run time) at the cost of more memory.
    computation : ``str``, optional (default: ``"annoy"``)
        The package supports KNN inference via annoy (specify 
        ``"annoy"``), PyNNDescent (specify ``"pynndescent"``) and scipy's 
        cKDTree (specify ``"cKDTree"``). Annoy 
        identifies approximate neighbours and runs quicker, cKDTree 
        identifies exact neighbours and is a bit slower.
    neighbours : ``int``, optional (default: 15)
        How many neighbours in ``gex`` to identify for each ``iss`` cell.
    obsm_fraction : ``bool``, optional (default: ``False``)
        If ``True``, will report the full fraction distribution of each 
        ``obs_to_take`` in ``.obsm`` of the resulting object.
    obsm_pbs : ``bool``, optional (default: ``False``)
        If ``True``, will store the identified ``gex`` neighbours for 
        each ``iss`` cell in ``.obsm['pbs']``.
    """
    #copy the objects to avoid modifying the originals
    iss = iss.copy()
    gex = gex.copy()
    #subset the ISS to genes that appear in the GEX
    iss = iss[:, [i in gex.var_names for i in iss.var_names]]
    #separate GEX into shared gene space and not shared gene space
    gex_only = gex[:, [i not in iss.var_names for i in gex.var_names]]
    gex = gex[:, iss.var_names]
    #turn both objects for KNNing into a log-normalised, z-scored form
    prepare_scaled(iss, min_genes=min_genes)
    prepare_scaled(gex, min_genes=min_genes)
    #this might remove some cells from the GEX, mirror in the gex_only
    gex_only = gex_only[gex.obs_names]
    #identify the KNN and use it to approximate expression
    return knn(iss=iss, 
               gex=gex, 
               gex_only=gex_only, 
               obs_to_take=obs_to_take, 
               round_counts=round_counts, 
               chunk_size=chunk_size, 
               computation=computation, 
               neighbours=neighbours, 
               obsm_fraction=obsm_fraction,
               obsm_pbs=obsm_pbs
              )
