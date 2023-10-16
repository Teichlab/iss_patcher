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

def prepare_scaled(adata, 
                   min_genes=3
                  ):
    """
    Log-normalise and z-score raw-count ``adata``, filtering to cells 
    with ``min_genes`` genes.
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    #normalise to median
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

def split_and_normalise_objects(iss, gex, 
                                min_genes=3
                               ):
    """
    Identify shared features between ``iss`` and ``gex``, split GEX 
    into two sub-objects - ``gex`` with features shared with ``iss`` 
    and ``gex_only`` that carries GEX-unique features. Filter ``iss`` 
    and ``gex`` to cells with at least ``min_genes`` genes, 
    log-normalise and z-score them, subset ``gex_only`` to match the 
    ``gex`` cell space, return all three objects.
    
    All arguments as in ``ip.patch()``.
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
    return iss, gex, gex_only

def get_knn_indices(issX, gexX, 
                    computation="annoy", 
                    neighbours=15
                   ):
    """
    Identify the neighbours of each ``issX`` observation in ``gexX``. 
    Return a ``scipy.spatial.cKDTree()``-style formatted output with 
    KNN indices for each low dimensional cell in the second element 
    of the tuple.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    issX : ``np.array``
        Low dimensionality processed expression data.
    gexX : ``np.array``
        High dimensionality processed expression data, with features 
        subset to match the low dimensionality data.
    """
    if computation == "annoy":
        #build the GEX index
        ckd = annoy.AnnoyIndex(gexX.shape[1], metric="euclidean")
        for i in np.arange(gexX.shape[0]):
            ckd.add_item(i,gexX[i,:])
        ckd.build(10)
        #query the GEX index with the ISS data
        ckdo_ind = []
        ckdo_dist = []
        for i in np.arange(issX.shape[0]):
            holder = ckd.get_nns_by_vector(issX[i,:], neighbours, include_distances=True)
            ckdo_ind.append(holder[0])
            ckdo_dist.append(holder[1])
        ckdout = (np.asarray(ckdo_dist),np.asarray(ckdo_ind))
    elif computation == "pynndescent":
        #build the GEX index
        ckd = pynndescent.NNDescent(gexX, metric="euclidean", n_jobs=-1, random_state=0)
        ckd.prepare()
        #query the GEX index with the ISS data
        ckdout = ckd.query(issX, k=neighbours)
        #need to reverse this to match conventions
        ckdout = (ckdout[1], ckdout[0])
    elif computation == "cKDTree":
        #build the GEX index
        ckd = scipy.spatial.cKDTree(gexX)
        #query the GEX index with the ISS data
        ckdout = ckd.query(x=issX, k=neighbours, workers=-1)
    else:
        raise ValueError("Invalid computation, must be 'annoy', 'pynndescent' or 'cKDTree'")
    return ckdout

def ckdout_to_sparse(ckdout, shape, 
                     neighbours=15
                    ):
    """
    Convert an array of KNN indices into a sparse matrix form. Return 
    the binary sparse matrix along with a copy where rows add up to 1 
    for easy mean computation.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    ckdout : tuple/list of ``np.array``
        Needs to have the KNN indices of low dimensionality cells in the 
        high dimensionality space as the second element (matching 
        ``scipy.sparse.cKDTree()`` output formatting).
    shape : list of ``int``
        The shape of the sparse matrix, low dimensionality cell count as 
        rows, high dimensionality cell count as columns.
    """
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
    #make a second version for means of stuff - divide data by neighbour count
    #this way each row adds up to 1 and this can be used for mean matrix operations
    pbs_means = pbs.copy()
    pbs_means.data = pbs_means.data/neighbours
    return pbs, pbs_means

def get_pbs_obs(iss, gex, pbs, pbs_means, 
                obs_to_take=None, 
                cont_obs_to_take=None, 
                nanmean=False, 
                obsm_fraction=False
               ):
    """
    Use the identified KNN to transfer ``.obs`` entries from the high 
    dimensionality object to the low dimensionality object. Returns a 
    majority vote/mean ``.obs`` data frame, and optionally a dictionary 
    of data frames capturing the complete fraction distribution of each 
    ``obs_to_take`` for each low dimensionality cell (for subsequent 
    ``.obsm`` insertion into the final object).
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    iss : ``AnnData``
        Low dimensionality data object with processed expression data in 
        ``.X``.
    gex : ``AnnData``
        High dimensionality data object with processed expression data in 
        ``.X``, subset to low dimensionality data object features.
    pbs : ``scipy.sparse.csr_matrix``
        Binary KNN graph, with low dimensionality cells as rows and 
        high dimensionality cells as columns.
    pbs_means : ``scipy_sparse.csr.matrix``
        KNN graph, with low dimensionality cells as rows and high 
        dimensionality cells as columns, with row values summing up to 1.
    """
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
    if cont_obs_to_take is not None:
        #just in case a single is passed as a string
        if type(cont_obs_to_take) is not list:
            cont_obs_to_take = [cont_obs_to_take]
        #compute the averages and turn them to a data frame
        cont_obs = pbs_means.dot(gex.obs[cont_obs_to_take].values)
        cont_obs = pd.DataFrame(
            cont_obs,
            index=iss.obs_names,
            columns=cont_obs_to_take
        )
        #compute a nanmean if instructed to
        if nanmean:
            #create a helper variable with 1 if non-nan value and 0 if nan
            non_nan_mask = gex.obs[cont_obs_to_take].values.copy()
            non_nan_mask[~np.isnan(non_nan_mask)] = 1
            non_nan_mask = np.nan_to_num(non_nan_mask)
            #now we can get the total non-nan counts for each cell and cont_obs
            non_nan_counts = pbs.dot(non_nan_mask)
            #and do 1/them to get the weights for a non-nan mean
            #as we'll only be averaging the non-nan elements by masking nan with 0
            non_nan_weights = 1/non_nan_counts
            #for instances where there are zero counts this is inf
            #we don't want inf, we want nan
            non_nan_weights[non_nan_weights == np.inf] = np.nan
            #we can now get sums of the non-nan values for the cont_obs
            #by filling in nans with zeroes prior to the operation
            cont_obs_nanmean = pbs.dot(gex.obs[cont_obs_to_take].fillna(0).values)
            #and now we can multiply them by the weights to get the means
            cont_obs_nanmean = cont_obs_nanmean * non_nan_weights
            #store both the means and the non-nan count total
            cont_obs_nanmean = pd.DataFrame(
                np.hstack((cont_obs_nanmean, non_nan_counts)),
                index=iss.obs_names,
                columns=[i+"_nanmean" for i in cont_obs_to_take] + [i+"_non_nan_count" for i in cont_obs_to_take]
            )
        for col in cont_obs_to_take:
            pbs_obs[col] = cont_obs[col]
        if nanmean:
            for col in cont_obs_nanmean:
                pbs_obs[col] = cont_obs_nanmean[col]
    if obsm_fraction:
        return pbs_obs, pbs_obsm
    else:
        return pbs_obs

def get_pbs_X(gex_only, pbs_means, 
              round_counts=True, 
              chunk_size=100000
             ):
    """
    Compute the expression of missing features in the low dimensionality 
    data as the mean of matching neighbours from the high dimensionality 
    data.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    gex_only : ``AnnData``
        High dimensionality data object with raw counts in ``.X``, subset to 
        features absent from the low dimensionality object.
    pbs_means : ``scipy_sparse.csr.matrix``
        KNN graph, with low dimensionality cells as rows and high 
        dimensionality cells as columns, with row values summing up to 1.
    """
    #if we're rounding the data, compute it in chunks to reduce RAM footprint
    if round_counts:
        #we'll be vstacking to this shortly
        X = None
        #process chunk_size iss cells at a time
        for start_pos in np.arange(0, pbs_means.shape[0], chunk_size):
            #these are our pseudobulk definitions for this chunk
            pbs_means_sub = pbs_means[start_pos:(start_pos+chunk_size),:]
            #get the corresponding expression for the chunk
            X_sub = pbs_means_sub.dot(gex_only.X)
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
        X = pbs_means.dot(gex_only.X)
    return X

def knn(iss, gex, gex_only, 
        obs_to_take=None, 
        cont_obs_to_take=None, 
        nanmean=False, 
        round_counts=True, 
        chunk_size=100000, 
        computation="annoy", 
        neighbours=15, 
        obsm_fraction=False, 
        obsm_pbs=False
       ):
    """
    ``ip.patch()`` without the normalisation, for when custom data 
    preparation is desired.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    iss : ``AnnData``
        Low dimensionality data object with processed expression data in 
        ``.X``.
    gex : ``AnnData``
        High dimensionality data object with processed expression data in 
        ``.X``, subset to low dimensionality data object features.
    gex_only : ``AnnData``
        High dimensionality data object with raw counts in ``.X``, subset to 
        features absent from the low dimensionality object.
    """
    #identify the KNN, preparing a (distances, indices) tuple
    ckdout = get_knn_indices(issX=iss.X, 
                             gexX=gex.X, 
                             computation=computation, 
                             neighbours=neighbours
                            )
    #turn KNN output into a scanpy-like graph
    #yields a version with both ones as data and ones as row sums
    #the latter is useful for matrix operation computation of means
    pbs, pbs_means = ckdout_to_sparse(ckdout=ckdout, 
                                      shape=[iss.shape[0], gex.shape[0]], 
                                      neighbours=neighbours
                                     )
    #get the annotations of the specified obs columns in the KNN
    pbs_obs = get_pbs_obs(iss=iss, 
                          gex=gex, 
                          pbs=pbs, 
                          pbs_means=pbs_means, 
                          obs_to_take=obs_to_take, 
                          cont_obs_to_take=cont_obs_to_take, 
                          nanmean=nanmean, 
                          obsm_fraction=obsm_fraction
                         )
    #if fractions are to be stored, this has two elements
    if obsm_fraction:
        pbs_obsm = pbs_obs[1]
        pbs_obs = pbs_obs[0]
    #get the expression matrix
    X = get_pbs_X(gex_only=gex_only, 
                  pbs_means=pbs_means, 
                  round_counts=round_counts, 
                  chunk_size=chunk_size
                 )
    #now we can build the object easily
    out = anndata.AnnData(X, var=gex_only.var, obs=pbs_obs, obsm=iss.obsm)
    #shove in the fractions from earlier if we need to
    if obsm_fraction:
        for anno_col in pbs_obsm:
            out.obsm[anno_col+"_fraction"] = pbs_obsm[anno_col]
    #shove in the pbs (KNN) if we need to
    #also stash gex obs names as there might have been filtering
    if obsm_pbs:
        out.obsm['pbs'] = pbs
        out.uns['pbs_gex_obs_names'] = np.array(gex.obs_names)
    return out

def patch(iss, gex, 
          min_genes=3, 
          obs_to_take=None, 
          cont_obs_to_take=None, 
          nanmean=False, 
          round_counts=True, 
          chunk_size=100000, 
          computation="annoy", 
          neighbours=15, 
          obsm_fraction=False, 
          obsm_pbs=False
         ):
    """
    Identify the nearest neighbours of low dimensionality observations 
    in related higher dimensionality data, approximate features absent  
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
        Discrete metadata only.
    cont_obs_to_take : ``str`` or list of ``str``, optional (default: ``None``)
        If provided, will report the average of the values of the 
        specified ``gex.obs`` column(s) for the neighbours of each 
        ``iss`` cell. Continuous metadata only.
    nanmean : ``bool``, optional (default: ``False``)
        If ``True``, will also compute an equivalent of ``np.nanmean()`` 
        for each ``cont_obs_to_take``.
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
        each ``iss`` cell in ``.obsm['pbs']``. A corresponding vector of 
        ``gex.obs_names`` will be stored in ``.uns['pbs_gex_obs_names']``.
    """
    #split up the GEX into ISS features and unique features
    #perform a quick normalisation of the ISS feature space objects
    #keep the GEX only features as raw counts
    iss, gex, gex_only = split_and_normalise_objects(iss=iss, 
                                                     gex=gex, 
                                                     min_genes=min_genes
                                                    )
    #identify the KNN and use it to approximate expression
    return knn(iss=iss, 
               gex=gex, 
               gex_only=gex_only, 
               obs_to_take=obs_to_take, 
               cont_obs_to_take=cont_obs_to_take, 
               nanmean=nanmean, 
               round_counts=round_counts, 
               chunk_size=chunk_size, 
               computation=computation, 
               neighbours=neighbours, 
               obsm_fraction=obsm_fraction, 
               obsm_pbs=obsm_pbs
              )

def patch_twostep(iss, gex, annot_key,  
                  min_genes=3, 
                  obs_to_take=None, 
                  cont_obs_to_take=None, 
                  nanmean=False, 
                  round_counts=True, 
                  chunk_size=100000, 
                  computation="annoy", 
                  neighbours=15, 
                  obsm_fraction=False, 
                  obsm_pbs=False
                 ):
    """
    A two-step version of the procedure, identifying each low dimensional cell's 
    KNN in the shared GEX feature space in the first go, and then 
    finding each ISS cell's KNN only among the high dimensional cells matching in 
    annotation. Prior to execution, high dimensional cells annotated with 
    categories with fewer than ``neighbours`` total representatives are 
    removed. ``annot_key``-derived entries in the output object are 
    based on the first-pass KNN on the full space, all other transferred 
    ``gex.obs`` are based on the second pass subset KNNs.
    
    All undescribed arguments as in ``ip.patch()``.
    
    Input
    -----
    annot_key : ``str``
        ``gex.obs`` key to use as the annotation.
    obsm_fraction : ``bool``, optional (default: ``False``)
        If ``True``, will additionally store the ``annot_key`` 
        cell fractions from the first pass KNN.
    obsm_pbs : ``bool``, optional (default: ``False``)
        If ``True``, will additionally store the first pass 
        annotation determining KNN in ``.obsm['pbs_annot']``.
    """
    #remove cells from GEX that are from rare annotations
    annot_count = gex.obs[annot_key].value_counts()
    annot_keep = list(annot_count[annot_count>=neighbours].index)
    gex = gex[np.isin(gex.obs[annot_key], annot_keep)]
    #split up the GEX into ISS features and unique features
    #perform a quick normalisation of the ISS feature space objects
    #keep the GEX only features as raw counts
    iss, gex, gex_only = split_and_normalise_objects(iss=iss, 
                                                     gex=gex, 
                                                     min_genes=min_genes
                                                    )
    #FIRST PASS - identify majority vote annotation for each iss cell
    #identify the KNN, preparing a (distances, indices) tuple
    ckdout = get_knn_indices(issX=iss.X, 
                             gexX=gex.X, 
                             computation=computation, 
                             neighbours=neighbours
                            )
    #turn KNN output into a scanpy-like graph
    #skip the row summing to one one, we don't need it
    pbs_annot, _ = ckdout_to_sparse(ckdout=ckdout, 
                                    shape=[iss.shape[0], gex.shape[0]], 
                                    neighbours=neighbours
                                   )
    #get the annotation voting based on the graph
    #pass none for both continuous-related inputs
    annot_obs = get_pbs_obs(iss=iss, 
                            gex=gex, 
                            pbs=pbs_annot, 
                            pbs_means=None, 
                            obs_to_take=annot_key, 
                            cont_obs_to_take=None, 
                            nanmean=False, 
                            obsm_fraction=obsm_fraction
                           )
    #split up the output if we're stashing the fraction
    if obsm_fraction:
        annot_obsm = annot_obs[1]
        annot_obs = annot_obs[0]
    #SECOND PASS - get neighbours only from a cell's annotation
    #prepare a variable where the identified KNN indices will go
    inds = np.zeros((iss.shape[0], neighbours))
    #plus indices for all the cells in the GEX
    gex_all_inds = np.arange(gex.shape[0])
    for celltype in np.unique(annot_obs[annot_key]):
        #get masks for where this is the annotation in ISS
        iss_mask = (annot_obs[annot_key].values == celltype)
        #and where this is the annotation in GEX
        gex_mask = (gex.obs[annot_key].values == celltype)
        #get indices of the GEX cells that will be used
        gex_inds = gex_all_inds[gex_mask]
        #get the KNN for the subset objects
        ckdout = get_knn_indices(issX=iss.X[iss_mask,:],
                                 gexX=gex.X[gex_mask,:],
                                 computation=computation,
                                 neighbours=neighbours
                                )
        #the indices in ckdout are relative to the subset
        #translate them back to original coordinates
        #and stash this in our master KNN index list
        inds[iss_mask,:] = gex_inds[ckdout[1]]
    #at this stage we can resume standard operation
    #turn KNN output into a scanpy-like graph
    #yields a version with both ones as data and ones as row sums
    #the latter is useful for matrix operation computation of means
    #the function only cares about the second element of ckdout
    #we just created it manually, place it into the second slot of a list
    pbs, pbs_means = ckdout_to_sparse(ckdout=[None, inds], 
                                      shape=[iss.shape[0], gex.shape[0]], 
                                      neighbours=neighbours
                                     )
    #get the annotations of the specified obs columns in the KNN
    pbs_obs = get_pbs_obs(iss=iss, 
                          gex=gex, 
                          pbs=pbs, 
                          pbs_means=pbs_means, 
                          obs_to_take=obs_to_take, 
                          cont_obs_to_take=cont_obs_to_take, 
                          nanmean=nanmean, 
                          obsm_fraction=obsm_fraction
                         )
    #if fractions are to be stored, this has two elements
    if obsm_fraction:
        pbs_obsm = pbs_obs[1]
        pbs_obs = pbs_obs[0]
    #get the expression matrix
    X = get_pbs_X(gex_only=gex_only, 
                  pbs_means=pbs_means, 
                  round_counts=round_counts, 
                  chunk_size=chunk_size
                 )
    #stash the annotation calls from earlier
    pbs_obs[annot_key] = annot_obs[annot_key]
    pbs_obs[annot_key+"_fraction"] = annot_obs[annot_key+"_fraction"]
    #now we can build the object easily
    out = anndata.AnnData(X, var=gex_only.var, obs=pbs_obs, obsm=iss.obsm)
    #shove in the fractions from earlier if we need to
    if obsm_fraction:
        #we've got the annotation fraction
        out.obsm[annot_key+"_fraction"] = annot_obsm[annot_key]
        for anno_col in pbs_obsm:
            out.obsm[anno_col+"_fraction"] = pbs_obsm[anno_col]
    #shove in the pbs (KNN) if we need to
    #also stash gex obs names as there might have been filtering
    if obsm_pbs:
        #there's also an annotation pbs
        out.obsm['pbs_annot'] = pbs_annot
        out.obsm['pbs'] = pbs
        out.uns['pbs_gex_obs_names'] = np.array(gex.obs_names)
    return out