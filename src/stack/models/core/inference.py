"""Inference and downstream utilities for the StateICL model."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import torch
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from torch.utils.data import DataLoader

from ...dataloader import TestSamplerDataset
from ..utils import align_result_to_adata_numpy


class InferenceMixin:
    @torch.no_grad()
    def predict(
        self, 
        features: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prediction without loss computation.
        
        Args:
            features: (batch_size, n_cells, n_genes) - original features (no log1p needed)
            mask: Optional mask to apply, if None, no masking
            
        Returns:
            Predictions dictionary
        """
        # Apply mask if provided (before log1p in forward)
        if mask is not None:
            features = features.clone()
            features[mask] = 0.0
        
        # Forward pass without loss (forward will apply log1p)
        result = self.forward(
            features, 
            return_loss=False
        )
        
        return result

    @torch.no_grad()
    def get_prediction(
        self,
        adata_path,
        genelist_path,
        gene_name_col: Optional[str] = None,
        mask_rate: float = 0.2,
        cell_ratio: float = 0.25,
        context_ratio: float = 0.25,
        is_masked_list: Optional[list] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        num_workers: int = 4,
        random_seed: Optional[int] = None,
        return_metrics: bool = False,  
        **dataloader_kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, float]]:  # Return type depends on return_metrics
        """
        Get predictions for masked genes in new AnnData object, or compute evaluation metrics.
        
        Args:
            adata_path: Path to h5ad file
            mask_rate: Masking rate to test (default 0.2)
            ratio: Cell ratio to mask (default 0.125)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar (default False)
            num_workers: Number of dataloader workers
            random_seed: Random seed for reproducible masking (default None)
            return_metrics: If True, returns evaluation metrics instead of predictions
            **dataloader_kwargs: Additional arguments for SplittableDatasetSamplerDataset
            
        Returns:
            If return_metrics=False:
                Tuple of (mean_predictions, dispersion_predictions):
                - mean_predictions: (n_cells, n_genes) - predicted means
                - dispersion_predictions: (n_cells, n_genes) - predicted dispersions
            If return_metrics=True:
                Dict of evaluation metrics from ``_compute_eval_metrics``:
                - masked_mae: Masked mean absolute error
                - masked_corr: Masked correlation averaged over valid cells
                - mask_rate: Fraction of entries evaluated
        """
        from tqdm.auto import tqdm
        
        self.eval()
        
        dataset = TestSamplerDataset(
            adata_path, 
            genelist_path,
            gene_name_col=gene_name_col,
            sample_size=self.n_cells,
            mode='eval',
            **dataloader_kwargs
        )
                     
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
                     
        if show_progress:
            pbar = tqdm(
                dataloader, 
                desc="Computing predictions" if not return_metrics else "Computing evaluation metrics",
                total=len(dataloader),
                unit="batch"
            )
            iterator = pbar
        else:
            iterator = dataloader
        
        # Initialize storage based on mode
        if return_metrics:
            helper_masked_mae_sum = 0.0
            helper_masked_corr_sum = 0.0
            helper_masked_entry_count = 0
            helper_corr_cell_count = 0
            helper_total_entries = 0
        else:
            # For predictions
            mean_predictions = []
            dispersion_predictions = []
            count_predictions = []
            logit_predictions = []

        # Set random seed if provided for reproducible masking
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
        
        device = next(self.parameters()).device
        
        # Process batches
        for i, batch in enumerate(iterator):
            features, metadata = batch
            
            features = features.to(device)
                             
            batch_size_curr, n_cells_batch, n_genes = features.shape
            gene_attn_mask = None
            if cell_ratio > 0 and cell_ratio < 1:
                gene_attn_mask = torch.zeros(n_cells_batch, n_cells_batch, dtype=torch.bool, device=device)
                gene_attn_mask[:int(cell_ratio * n_cells_batch), int(cell_ratio * n_cells_batch):] = True
                gene_attn_mask = gene_attn_mask.unsqueeze(0).unsqueeze(0)

            mask = torch.zeros(batch_size_curr, n_cells_batch, n_genes, dtype=torch.bool, device=device)
            #n_genes_to_mask = int(n_genes * mask_rate)
            n_genes_to_mask = 0
            mask_indices = torch.randperm(n_genes, device=device)[:n_genes_to_mask]
            if n_genes_to_mask > 0:
                mask[:, :, mask_indices] = True
            #mask[:,:int(n_cells_batch*cell_ratio),:] = False                
            
            # Calculate library size before masking
            observed_lib_size = features.sum(dim=-1, keepdim=True)  # (B, n_cells, 1)
            
            # Apply log1p transformation
            features_log = torch.log1p(features)

            masked_features_log = features_log.clone()
            masked_features_log[mask] = 0.0

            tokens = self._reduce_and_tokenize(masked_features_log)
            n_kept_cell = int(cell_ratio * n_cells_batch)
            n_context_cell = int((cell_ratio + context_ratio) * n_cells_batch)

            if getattr(self, "query_pos_embedding", None) is not None and n_kept_cell < n_cells_batch:
                qpe = self.query_pos_embedding.unsqueeze(0).unsqueeze(0).to(device=tokens.device, dtype=tokens.dtype)
                if is_masked_list:
                    sl = is_masked_list[i * batch_size_curr : (i + 1) * batch_size_curr]
                    is_masked = torch.as_tensor(np.stack(sl, axis=0), device=tokens.device, dtype=tokens.dtype)
                    logit = torch.cat(
                        [
                            torch.zeros(
                                [batch_size_curr, n_context_cell],
                                dtype=is_masked.dtype,
                                device=is_masked.device,
                            ),
                            is_masked,
                        ],
                        dim=1,
                    )
                    cell_mask = (logit > 0).to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
                    tokens = tokens + qpe * cell_mask
                else:
                    mask_expanded = torch.zeros_like(tokens)
                    mask_expanded[:, n_context_cell:, :, :] = 1.0
                    tokens = tokens + qpe * mask_expanded

            x = self._run_attention_layers(tokens, gene_attn_mask=gene_attn_mask)

            # Final cell embeddings
            final_cell_embeddings = x.reshape(batch_size_curr, n_cells_batch, -1)
            logits = None
            if getattr(self, "cls", None) is not None and n_context_cell < n_cells_batch:

                mean_kept_cell = final_cell_embeddings[:,:n_context_cell].mean(dim=1).detach()
                tail_feat = final_cell_embeddings[:, n_context_cell:, :].detach()  
                ctx_tail = mean_kept_cell.unsqueeze(1).expand(-1, tail_feat.size(1), -1)
                inp_tail = torch.cat([ctx_tail, tail_feat], dim=-1)
                logit_tail = self.cls(inp_tail).squeeze(-1)  # (B, Nt)

                logits = torch.cat([torch.zeros([batch_size_curr,n_context_cell],dtype=logit_tail.dtype,device=logit_tail.device), logit_tail], dim=1)          
            

            nb_mean, nb_dispersion, px_scale = self._compute_nb_parameters(
                final_cell_embeddings, observed_lib_size
            )
            
            if return_metrics:
                if mask_rate > 0:
                    n_genes_to_evaluate = max(1, int(n_genes * mask_rate))
                    eval_indices = torch.randperm(n_genes, device=device)[:n_genes_to_evaluate]
                    metric_mask = torch.zeros_like(mask, dtype=torch.bool)
                    metric_mask[:, :, eval_indices] = True
                else:
                    metric_mask = torch.ones_like(mask, dtype=torch.bool)

                helper_metrics = self._compute_eval_metrics(nb_mean, features, metric_mask)

                mask_entries = int(metric_mask.sum().item())
                helper_masked_entry_count += mask_entries
                helper_total_entries += metric_mask.numel()
                if mask_entries > 0:
                    helper_masked_mae_sum += helper_metrics["masked_mae"].item() * mask_entries

                mask_f = metric_mask.float()
                valid_cells = int((mask_f.sum(dim=-1) > 1).sum().item())
                if valid_cells > 0:
                    helper_masked_corr_sum += helper_metrics["masked_corr"].item() * valid_cells
                    helper_corr_cell_count += valid_cells
                
            else:
                # Store predictions
                nb_mean_flat = nb_mean.reshape(-1, n_genes)  # (B * n_cells, n_genes)
                nb_dispersion_median = nb_dispersion[:,:n_context_cell,:].median(dim=1, keepdim=True).values.expand(-1, n_cells_batch, -1)
                nb_dispersion_flat = nb_dispersion.reshape(-1, n_genes)  # (B * n_cells, n_genes)
                nb_dispersion_median_flat = nb_dispersion_median.reshape(-1, n_genes)
                nb_dist = NegativeBinomial(mu=nb_mean_flat, theta=nb_dispersion_median_flat)
                mean_predictions.append(nb_mean_flat.cpu().numpy())
                dispersion_predictions.append(nb_dispersion_flat.cpu().numpy())
                count_predictions.append(nb_dist.sample().cpu().numpy())
                if logits is not None:
                    logits_flat = logits.reshape(-1)
                    logit_predictions.append(logits_flat.cpu().numpy())
            
            if show_progress:
                if return_metrics:
                    current_cells = helper_corr_cell_count
                    pbar.set_postfix({
                        'cells_evaluated': f'{current_cells:,}',
                        'mask_rate': f'{mask_rate:.2f}'
                    })
                else:
                    current_cells = sum([pred.shape[0] for pred in mean_predictions])
                    pbar.set_postfix({
                        'cells_processed': f'{current_cells:,}',
                        'mask_rate': f'{mask_rate:.2f}'
                    })
        
        if show_progress:
            pbar.close()
        
        if return_metrics:
            # Return evaluation metrics
            masked_mae_helper = (
                helper_masked_mae_sum / helper_masked_entry_count
                if helper_masked_entry_count > 0
                else float("nan")
            )
            masked_corr_helper = (
                helper_masked_corr_sum / helper_corr_cell_count
                if helper_corr_cell_count > 0
                else float("nan")
            )
            mask_rate_helper = (
                helper_masked_entry_count / helper_total_entries
                if helper_total_entries > 0
                else float("nan")
            )
            return {
                "masked_mae": masked_mae_helper,
                "masked_corr": masked_corr_helper,
                "mask_rate": mask_rate_helper,
            }
        else:
            # Return predictions (original behavior)
            mean_predictions = np.concatenate(mean_predictions, axis=0)[:dataset.n_human_cells]
            dispersion_predictions = np.concatenate(dispersion_predictions, axis=0)[:dataset.n_human_cells]
            count_predictions = np.concatenate(count_predictions, axis=0)[:dataset.n_human_cells]
            if logit_predictions:
                logit_predictions = np.concatenate(logit_predictions, axis=0)[:dataset.n_human_cells]
            return mean_predictions, dispersion_predictions, count_predictions, logit_predictions

    @torch.no_grad()
    def get_attn(
        self,
        adata_path,
        genelist_path,
        gene_name_col: Optional[str] = None,
        mask_rate: float = 0.2,
        cell_ratio: float = 0.25,
        batch_size: int = 32,
        show_progress: bool = False,
        num_workers: int = 4,
        random_seed: Optional[int] = None,
        **dataloader_kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, float]]:  # Return type depends on return_metrics
        """
        Get attns for masked genes in new AnnData object\
        
        Args:
            adata_path: Path to h5ad file
            mask_rate: Masking rate to test (default 0.2)
            ratio: Cell ratio to mask (default 0.125)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar (default False)
            num_workers: Number of dataloader workers
            random_seed: Random seed for reproducible masking (default None)
            return_metrics: If True, returns evaluation metrics instead of predictions
            **dataloader_kwargs: Additional arguments for SplittableDatasetSamplerDataset
        """
        from tqdm.auto import tqdm
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        self.eval()
        
        dataset = TestSamplerDataset(
            adata_path, 
            genelist_path,
            gene_name_col=gene_name_col,
            sample_size=self.n_cells,
            mode='eval',
            **dataloader_kwargs
        )
                     
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
                     
        if show_progress:
            pbar = tqdm(
                dataloader, 
                desc="Computing attns",
                total=len(dataloader),
                unit="batch"
            )
            iterator = pbar
        else:
            iterator = dataloader
        
        all_layer_attns = [[] for _ in range(self.n_layers)]

        # Set random seed if provided for reproducible masking
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(random_seed)
        
        device = next(self.parameters()).device
        
        # Process batches
        for batch in iterator:
            features, metadata = batch
            
            features = features.to(device)
                             
            batch_size_curr, n_cells_batch, n_genes = features.shape
            gene_attn_mask = None
            if cell_ratio > 0 and cell_ratio < 1:
                gene_attn_mask = torch.zeros(n_cells_batch, n_cells_batch, dtype=torch.bool, device=device)
                gene_attn_mask[:int(cell_ratio * n_cells_batch), int(cell_ratio * n_cells_batch):] = True
                gene_attn_mask = gene_attn_mask.unsqueeze(0).unsqueeze(0)

            mask = torch.zeros(batch_size_curr, n_cells_batch, n_genes, dtype=torch.bool, device=device)
            n_genes_to_mask = int(n_genes * mask_rate)
            mask_indices = torch.randperm(n_genes, device=device)[:n_genes_to_mask]
            if n_genes_to_mask > 0:
                mask[:, :, mask_indices] = True              
            
            # Calculate library size before masking
            observed_lib_size = features.sum(dim=-1, keepdim=True)  # (B, n_cells, 1)
            
            # Apply log1p transformation
            features_log = torch.log1p(features)
            masked_features_log = features_log.clone()
            masked_features_log[mask] = 0.0

            tokens = self._reduce_and_tokenize(masked_features_log)
            x, attn_maps = self._run_attention_layers(
                tokens, gene_attn_mask=gene_attn_mask, return_attn=True
            )

            for layer_idx, attn in enumerate(attn_maps):
                all_layer_attns[layer_idx].append(
                    attn.mean(dim=1).reshape(-1, attn.shape[-1]).cpu().numpy()
                )
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
            
        final_attns = []
        for i in range(self.n_layers):
            layer_batches = all_layer_attns[i]
            concatenated_attn = np.concatenate(layer_batches, axis=0)[:dataset.n_human_cells]
            final_attns.append(concatenated_attn)

        return final_attns

    @torch.no_grad()
    def get_latent_representation(
        self,
        adata_path,
        genelist_path,
        gene_name_col: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        num_workers: int = 4,
        **dataloader_kwargs
    ) -> np.ndarray:
        """
        Get cell embeddings for AnnData object.
        
        Args:
            adata_path: Path to h5ad file
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar (default True)
            **dataloader_kwargs: Additional arguments for SplittableDatasetSamplerDataset
            
        Returns:
            cell_embeddings: (n_cells, n_hidden * token_dim) - cell embeddings as numpy array
        """
        from tqdm.auto import tqdm
        
        self.eval()
        dataset = TestSamplerDataset(
            adata_path, 
            genelist_path,
            gene_name_col=gene_name_col,
            sample_size=self.n_cells,
            mode='eval',
            **dataloader_kwargs
        )
                     
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # Don't shuffle dataloader since we already shuffled data
            num_workers=num_workers
        )
                     
        all_embeddings = []
        all_dsembeddings = []
        
        if show_progress:
            pbar = tqdm(
                dataloader, 
                desc="Extracting embeddings",
                total=len(dataloader),
                unit="batch"
            )
            iterator = pbar
        else:
            iterator = dataloader
                     
        # Process shuffled data
        device = next(self.parameters()).device
        for batch in iterator:
            features, metadata = batch

            features = features.to(device)
                             
            batch_size_curr, n_cells_batch, n_genes = features.shape
                             
            # Apply log1p transformation
            features_log = torch.log1p(features)

            tokens = self._reduce_and_tokenize(features_log)
            x = self._run_attention_layers(tokens)

            # Final cell embeddings
            batch_embeddings = x.reshape(batch_size_curr, n_cells_batch, -1)

            flat_embeddings = batch_embeddings.reshape(-1, batch_embeddings.shape[-1])

            all_embeddings.append(flat_embeddings.cpu().numpy())
            all_dsembeddings.append(flat_embeddings.mean(dim=1).cpu().numpy())
            
            if show_progress:
                current_cells = sum([emb.shape[0] for emb in all_embeddings])
                pbar.set_postfix({
                    'cells_processed': f'{current_cells:,}',
                    'embedding_dim': batch_embeddings.shape[-1]
                })
                         
        if show_progress:
            pbar.close()
                         
        # Concatenate embeddings
        cell_embeddings = np.concatenate(all_embeddings, axis=0)[0:dataset.n_human_cells]
        #print(cell_embeddings.shape)
        dataset_embeddings = np.concatenate(all_dsembeddings, axis=0)         
        return cell_embeddings, dataset_embeddings

    def decode(self, embeddings: torch.Tensor, lib_size: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings to gene expression values.
        
        Args:
            embeddings: (n_cells, n_hidden * token_dim) - cell embeddings
            lib_size: (n_cells, 1) or (n_cells,) - library sizes for each cell
            
        Returns:
            expressions: (n_cells, n_genes) - decoded gene expressions
        """
        # Handle input shapes
        if embeddings.ndim == 2:
            n_cells = embeddings.shape[0]
            embeddings = embeddings.unsqueeze(0)
            single_batch = True
        else:
            single_batch = False
            n_cells = embeddings.shape[1]

        batch_size, n_cells, embedding_dim = embeddings.shape

        lib_size_tensor = lib_size
        if lib_size_tensor.ndim == 1:
            lib_size_tensor = lib_size_tensor.unsqueeze(0)
        if lib_size_tensor.ndim == 3 and lib_size_tensor.shape[-1] == 1:
            lib_size_tensor = lib_size_tensor.squeeze(-1)
        lib_size_tensor = lib_size_tensor.reshape(-1, n_cells)
        if lib_size_tensor.shape[0] != batch_size:
            if lib_size_tensor.shape[0] == 1:
                lib_size_tensor = lib_size_tensor.expand(batch_size, -1)
            else:
                raise ValueError("lib_size shape does not match embeddings")

        observed_lib_size = lib_size_tensor.unsqueeze(-1)

        nb_mean, _, _ = self._compute_nb_parameters(embeddings, observed_lib_size)

        if single_batch:
            expressions = nb_mean.squeeze(0)
        else:
            expressions = nb_mean

        return expressions

    @torch.no_grad()
    def get_incontext_prediction(
        self,
        base_adata_or_path,
        test_adata_or_path,
        genelist_path: str,
        mask_rate: float = 0.8,
        prompt_ratio: float = 0.25,
        context_ratio: float = 0.25,
        mode: str = 'predict',
        gene_name_col: Optional[str] = None,
        test_logit: Optional[torch.Tensor] = None,
        is_masked: Optional[torch.Tensor] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        num_workers: int = 4,
        random_seed: Optional[int] = None,
        **dataloader_kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs in-context prediction by mixing a base dataset with a test dataset.
        A portion of cells from the base dataset provides context for the test cells
        within the same attention window (batch).

        Args:
            base_adata_path: Path to the base (context) .h5ad file.
            test_adata_path: Path to the test (target) .h5ad file.
            genelist_path: Path to the list of genes used by the model.
            ratio: The ratio of context cells in each mixed sample (e.g., 0.875 for 7/8).
            mode: 'latent' to return cell embeddings, 'predict' to return gene expression predictions, 'generate' to return expression prediction and uncertainty.
            gene_name_col: The column in .var to use as gene names for alignment.
            batch_size: Batch size for processing the mixed data.
            show_progress: Whether to show a progress bar.
            num_workers: Dataloader workers.
            random_seed: Seed for reproducible shuffling.
            **dataloader_kwargs: Additional arguments for TestSamplerDataset.

        Returns:
            - If mode='latent', returns a numpy array of latent representations for test cells.
            - If mode='predict', returns a tuple of (mean, dispersion) predictions for test cells.
        """
        ratio = prompt_ratio + context_ratio
        assert 0 < ratio < 1, "Ratio must be between 0 and 1."
        assert mode in ['latent', 'predict', 'generate'], "Mode must be 'latent', 'predict' or 'generate'."

        self.eval()

        if isinstance(base_adata_or_path, str):
            base_adata = ad.read_h5ad(base_adata_or_path)
        else:
            base_adata = base_adata_or_path.copy()

        if isinstance(test_adata_or_path, str):
            test_adata = ad.read_h5ad(test_adata_or_path)
        else:
            test_adata = test_adata_or_path.copy()

        # --- 2. Prepare for in-context batching ---
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        base_indices = np.random.permutation(base_adata.n_obs)
        test_indices = np.arange(test_adata.n_obs)
        
        # Calculate number of cells per sample from each dataset
        n_test_cells = max(1, int(self.n_cells * (1-ratio)))
        n_base_cells = self.n_cells - n_test_cells
        
        # We create one large AnnData to pass to the dataloader
        num_test_samples = math.ceil(len(test_indices) / n_test_cells)
        total_cells = num_test_samples * self.n_cells

        mixed_adata_list = []
        # Keep track of which cells are from the test set
        is_test_cell_mask = np.zeros(total_cells, dtype=bool)
        is_base_cell_mask = np.zeros(total_cells, dtype=bool)
        is_masked_list = []
        base_idx_ptr = 0
        for i in range(num_test_samples):
            # Get test cells
            start, end = i * n_test_cells, (i + 1) * n_test_cells
            current_test_indices = test_indices[start:end]
            len_test_indices = len(current_test_indices)
            if len_test_indices < n_test_cells:
                need = n_test_cells - len_test_indices
                pad = test_indices[:need]
                current_test_indices = np.concatenate([current_test_indices, pad])
            
            # Get base cells, looping if we run out
            idx_base = np.arange(base_idx_ptr, base_idx_ptr + n_base_cells) % len(base_indices)
            current_base_indices = base_indices[idx_base]
            base_idx_ptr = (base_idx_ptr + n_base_cells) % len(base_indices)

            # Concatenate AnnData objects for this sample
            sample_adata = ad.concat(
                [base_adata[current_base_indices], test_adata[current_test_indices]],
                join='inner',
                axis=0,
                label='origin',
                keys=['base', 'test']
            )
            mixed_adata_list.append(sample_adata)
            if is_masked is not None:
                is_masked_list.append(np.asarray(is_masked)[current_test_indices])

            # Update the mask
            mask_start_idx = i * self.n_cells
            is_test_cell_mask[mask_start_idx + n_base_cells : mask_start_idx + n_base_cells + len_test_indices] = True
            is_base_cell_mask[mask_start_idx : mask_start_idx + n_base_cells] = True

        # Final concatenated AnnData object
        full_mixed_adata = ad.concat(mixed_adata_list, axis=0, join='inner')

        if mode == 'latent':
            all_embeddings, _ = self.get_latent_representation(
                adata_path=full_mixed_adata,
                genelist_path=genelist_path,
                gene_name_col=gene_name_col,
                batch_size=batch_size,
                show_progress=show_progress,
                num_workers=num_workers,
                **dataloader_kwargs
            )
            result = all_embeddings[is_test_cell_mask]
        
        else:
            mean_preds, disp_preds, count_preds, logit_preds = self.get_prediction(
                adata_path=full_mixed_adata,
                genelist_path=genelist_path,
                gene_name_col=gene_name_col,
                mask_rate=0.2, # No mask for base data
                cell_ratio=prompt_ratio,
                context_ratio=context_ratio,
                is_masked_list=is_masked_list,
                batch_size=batch_size,
                show_progress=show_progress,
                num_workers=num_workers,
                **dataloader_kwargs
            )
            result = count_preds[is_test_cell_mask]
            cell_indices_to_keep = None
            
            if mode == 'generate':
                new_test_logit = logit_preds[is_test_cell_mask]
                cell_indices_to_keep = np.zeros(is_test_cell_mask.sum(), dtype=bool)
                cell_indices_to_keep[~is_masked] = True
                
                unmask_rate = (is_masked.sum() / is_masked.shape[0] - mask_rate) * is_masked.shape[0] / is_masked.sum()
                cell_indices_to_keep[is_masked] = new_test_logit[is_masked]>np.quantile(new_test_logit[is_masked],unmask_rate)
                #cell_indices_to_keep[new_test_logit>np.quantile(new_test_logit,mask_rate)] = True
                is_masked[~cell_indices_to_keep] = False
                is_masked[new_test_logit>0] = True
            result = align_result_to_adata_numpy(result, test_adata, genelist_path, gene_name_col, cell_indices_to_keep=cell_indices_to_keep)
            
            from scipy.sparse import csr_matrix
            # truncate
            #result[result < 0.2] = 0
            result = csr_matrix(result)
            if mode == 'generate':
                return result, new_test_logit, is_masked
        return result

    @torch.no_grad()
    def get_incontext_attn(
        self,
        base_adata_or_path,
        test_adata_or_path,
        genelist_path: str,
        mask_rate: float = 0.8,
        ratio: float = 0.875,
        mode: str = 'predict',
        gene_name_col: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        num_workers: int = 4,
        random_seed: Optional[int] = None,
        **dataloader_kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs in-context prediction by mixing a base dataset with a test dataset.
        A portion of cells from the base dataset provides context for the test cells
        within the same attention window (batch).

        Args:
            base_adata_path: Path to the base (context) .h5ad file.
            test_adata_path: Path to the test (target) .h5ad file.
            genelist_path: Path to the list of genes used by the model.
            ratio: The ratio of context cells in each mixed sample (e.g., 0.875 for 7/8).
            mode: 'latent' to return cell embeddings, 'predict' to return gene expression predictions, 'generate' to return expression prediction and uncertainty.
            gene_name_col: The column in .var to use as gene names for alignment.
            batch_size: Batch size for processing the mixed data.
            show_progress: Whether to show a progress bar.
            num_workers: Dataloader workers.
            random_seed: Seed for reproducible shuffling.
            **dataloader_kwargs: Additional arguments for TestSamplerDataset.

        Returns:
            - If mode='latent', returns a numpy array of latent representations for test cells.
            - If mode='predict', returns a tuple of (mean, dispersion) predictions for test cells.
        """
        assert 0 < ratio < 1, "Ratio must be between 0 and 1."
        assert mode in ['latent', 'predict', 'generate'], "Mode must be 'latent', 'predict' or 'generate'."

        self.eval()

        if isinstance(base_adata_or_path, str):
            base_adata = ad.read_h5ad(base_adata_or_path)
        else:
            base_adata = base_adata_or_path.copy()

        if isinstance(test_adata_or_path, str):
            test_adata = ad.read_h5ad(test_adata_or_path)
        else:
            test_adata = test_adata_or_path.copy()

        # --- 2. Prepare for in-context batching ---
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        base_indices = np.random.permutation(base_adata.n_obs)
        test_indices = np.arange(test_adata.n_obs)
        
        # Calculate number of cells per sample from each dataset
        n_test_cells = max(1, int(self.n_cells * (1-ratio)))
        n_base_cells = self.n_cells - n_test_cells
        
        # We create one large AnnData to pass to the dataloader
        num_test_samples = math.ceil(len(test_indices) / n_test_cells)
        total_cells = num_test_samples * self.n_cells

        mixed_adata_list = []
        # Keep track of which cells are from the test set
        is_test_cell_mask = np.zeros(total_cells, dtype=bool)
        is_base_cell_mask = np.zeros(total_cells, dtype=bool)
        base_indices_list = []
        
        base_idx_ptr = 0
        for i in range(num_test_samples):
            # Get test cells
            start, end = i * n_test_cells, (i + 1) * n_test_cells
            current_test_indices = test_indices[start:end]
            len_test_indices = len(current_test_indices)
            if len_test_indices < n_test_cells:
                need = n_test_cells - len_test_indices
                pad = test_indices[:need]
                current_test_indices = np.concatenate([current_test_indices, pad])
            
            # Get base cells, looping if we run out
            idx_base = np.arange(base_idx_ptr, base_idx_ptr + n_base_cells) % len(base_indices)
            current_base_indices = base_indices[idx_base]
            base_indices_list.append(current_base_indices)
            base_idx_ptr = (base_idx_ptr + n_base_cells) % len(base_indices)

            # Concatenate AnnData objects for this sample
            sample_adata = ad.concat(
                [base_adata[current_base_indices], test_adata[current_test_indices]],
                join='inner',
                axis=0,
                label='origin',
                keys=['base', 'test']
            )
            mixed_adata_list.append(sample_adata)

            # Update the mask
            mask_start_idx = i * self.n_cells
            is_test_cell_mask[mask_start_idx + n_base_cells : mask_start_idx + n_base_cells + len_test_indices] = True
            is_base_cell_mask[mask_start_idx : mask_start_idx + n_base_cells] = True

        # Final concatenated AnnData object
        full_mixed_adata = ad.concat(mixed_adata_list, axis=0, join='inner')

        final_attns = self.get_attn(
                adata_path=full_mixed_adata,
                genelist_path=genelist_path,
                gene_name_col=gene_name_col,
                mask_rate=mask_rate, # No mask for base data
                cell_ratio=ratio,
                batch_size=batch_size,
                show_progress=show_progress,
                num_workers=num_workers,
                **dataloader_kwargs
            )
        processed_attns = []
        for concatenated_attn in final_attns:
            processed_attns.append(concatenated_attn[is_test_cell_mask,0:n_base_cells])
        return processed_attns, base_indices_list

    @torch.no_grad()
    def get_incontext_generation(
        self,
        base_adata_or_path,
        test_adata_or_path,
        genelist_path: str,
        num_steps: int = 5,
        prompt_ratio: float = 0.25,
        context_ratio: float = 0.4,
        context_ratio_min: float = 0.2,
        mask_rate: float = 1.0,
        mode: str = 'vanilla',
        gene_name_col: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        num_workers: int = 4,
        random_seed: Optional[int] = 42,
        **dataloader_kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Performs in-context generation by mixing a base dataset with a test dataset.
        Difference to in-context prediction: generation comprises multiple steps and follows the schedule alpha = 1-1/t.
        """
        assert 0 < (prompt_ratio+context_ratio) < 1, "Ratio must be between 0 and 1."
        assert 0 <= context_ratio_min <= context_ratio < 1, "Min context ratio must be smaller than 1 and context ratio"
        from scipy.sparse import csr_matrix
        if num_steps is None:
            num_steps = math.ceil(1.0/mask_rate)

        t = (np.arange(num_steps, dtype=np.float32)+1) / num_steps
        if num_steps == 1:
            cr_list = np.array([context_ratio], dtype=np.float32)
        else:
            cr_list = np.linspace(context_ratio_min, context_ratio, num_steps, dtype=np.float32)
        mr_list = (1 - t) 
        mr_list[-1] = 0.0
        if isinstance(test_adata_or_path, str):
            test_adata = ad.read_h5ad(test_adata_or_path)
        else:
            test_adata = test_adata_or_path.copy()
        print(mr_list)
        i = 1
        result = None
        if mode == 'vanilla':
            for mr in mr_list:
                result = self.get_incontext_prediction(
                   base_adata_or_path = base_adata_or_path,
                   test_adata_or_path = test_adata,
                   genelist_path = genelist_path,
                   mask_rate = mr,
                   prompt_ratio = prompt_ratio,
                   context_ratio = context_ratio,
                   mode = 'predict',
                   gene_name_col = gene_name_col,
                   batch_size = batch_size,
                   show_progress = show_progress,
                   num_workers = num_workers,
                   random_seed = random_seed + i,
                   **dataloader_kwargs)
                i = i + 1
                if test_adata.raw is not None:
                    test_adata.raw.X = result
                else:
                    test_adata.X = result
            return result
        else:
            g = torch.Generator()
            if random_seed is not None:
                g.manual_seed(int(random_seed))
            test_logit = None
            is_masked = np.ones(test_adata.shape[0],dtype=bool)
            for mr, cr in zip(mr_list, cr_list):
                result, test_logit, is_masked = self.get_incontext_prediction(
                   base_adata_or_path = base_adata_or_path,
                   test_adata_or_path = test_adata,
                   genelist_path = genelist_path,
                   mask_rate = mr,
                   prompt_ratio = prompt_ratio,
                   context_ratio = cr,
                   mode = 'generate',
                   test_logit = test_logit,
                   is_masked = is_masked,
                   gene_name_col = gene_name_col,
                   batch_size = batch_size,
                   show_progress = show_progress,
                   num_workers = num_workers,
                   random_seed = random_seed + i,
                   **dataloader_kwargs)
                print("Generation Step", i)
                print((test_logit<0).sum()/test_logit.shape[0])
                i = i + 1
                if test_adata.raw is not None:
                    test_adata.raw.X = result
                else:
                    test_adata.X = result
                
            return result, test_logit


__all__ = ["InferenceMixin"]
