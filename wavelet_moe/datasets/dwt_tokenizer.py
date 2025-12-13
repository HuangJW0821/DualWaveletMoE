import math
import numpy as np
import pywt
import torch

class DWTTokenizer():
    def __init__(self, wavelet:str = "bior2.2", mode:str = "periodization", level:int = 2, axis:int = -1, patch_size: int = 8):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.axis = axis

        if patch_size%2 != 0:
            raise ValueError("Patch size must be an even number")
        if patch_size < 2**(self.level):
            raise ValueError("Patch size shold larger than 2^level")
        
        self.patch_size = patch_size

    def wavedec(self, data):
        return pywt.wavedec(data, self.wavelet, self.mode, self.level, self.axis)

    def waverec(self, coeffs):
        if len(coeffs)>1 and len(coeffs) < self.level+1: 
            raise ValueError(f"Length of coeffs {len(coeffs)} inconsistent with decomponent levels {self.level}")
        return pywt.waverec(coeffs, self.wavelet, self.mode, self.axis)

    def single_seq_index_to_coeff_index(self, t, l):
        """
        Map time-step t to corresponding coeff index at level l. \n

        TODO: Approximating DWT as binary tree is an imprecise approach
        that fails to account for the convolution step
        """
        return t // 2**l
    
    def patch_seq_index_to_coeff_index(self, start, end, l):
        """
        Map time step index (start : start+patch_size) to corresponding coeff index from level 0 to l
        """
        ans = []
        for i in range(0, l+1):
            left = self.single_seq_index_to_coeff_index(start, i)
            right = self.single_seq_index_to_coeff_index(end, i)
            ans.append(list(range(left, right)))
        
        return ans
    
    def patch_wise_tokenize(self, data):
        """
        Input [batch_sz, seq_len] time-series sequences data, tokenize it into [batch_sz, seq_len*2] token sequences. \n
        For each time-series sequence, first apply DWT to get coeffs matrix [approx_{level}, detail_{level}, ..., detail_1], and concat it into a coeffs sequence with same length as seq_len. \n
        Next, for both sequences, divide them into floor(seq_len/patch_sz) patches. \n
        Then, a token is a concat of time-series patch with corresponding coeffs patch, so we get a token seq of floor(seq_len/patch_sz) tokens. \n

        Args:
            data (`array-like`): time-series seq batch with shape [batch_sz, seq_len], seq_len must be multiple of patch_sz.

        Returns:
            (tokens_seqs, coeffs):
            tokens_seq: An array-like shape [batch_sz, seq_len*2], each row is a tokens_seq consist of seq_len/patch_sz tokens. \n
            coeffs: DWT coeff array, ie [approx_{level}, detail_{level}, ... detail_1], each detail_{i} is shape [batch_sz, seq_len/(2^i)]
        """
        data = np.asarray(data)

        # turn 1D data into 2D vector
        if data.ndim == 1:
            data = data[None, :]    # shape: (1, T)
        
        seq_num, seq_len = data.shape
        level = self.level
        patch_size = self.patch_size

        if seq_len%patch_size != 0:
            raise ValueError(f"Input sequence length must be multiple of patch_size {self.patch_size}")

        # dwt
        coeffs = self.wavedec(data)
        
        # concat seq patches & coeff patches
        all_new_data = []

        # for every sequence in data
        for seq_idx in range(seq_num):
            seq = data[seq_idx]
            patches = []

            # divide by patch
            for start in range(0, seq_len, patch_size):
                end = start + patch_size
                base_patch = seq[start:end]

                coeff_indcies = self.patch_seq_index_to_coeff_index(start, end, level)

                # init with approx_{length}
                coeff_patches = [ coeffs[0][seq_idx][coeff_indcies[-1][0] : coeff_indcies[-1][-1] + 1] ]
                
                # deal with details
                for l in range(1, level+1):
                    coeff_patch = coeffs[l][seq_idx][coeff_indcies[level-l+1][0] : coeff_indcies[level-l+1][-1] + 1]    # Attention
                    coeff_patches.append(coeff_patch)
                
                # concat base patch and coeff patch
                merged_patch = np.concatenate([base_patch] + coeff_patches)
                patches.append(merged_patch)
            
            # concat all patches
            patches = np.concatenate(patches)
            all_new_data.append(patches)
        
        return np.stack(all_new_data), coeffs


    def patch_wise_detokenize(self, data):
        """
        Detokenize input data into time-series seqs & coeffs seqs

        Args:
            data: [batch_sz, seq_len*2] tokens seqs.

        Returns:
            (time-series sequences, coeffs list): 
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy() if data.requires_grad else data.cpu().numpy()

        data = np.asarray(data)

        # turn 1D data into 2D vector
        if data.ndim == 1:
            data = data[None, :]    # shape: (1, T)
        
        seq_num, seq_len = data.shape
        level = self.level
        patch_size = self.patch_size

        collected_base_seqs = []
        collected_coeffs = []

        for seq_idx in range(seq_num):
            seq = data[seq_idx]
            
            base_seq = []
            coeff_patches = []
            coeffs = [[] for _ in range(level+1)]

            # extract base_seq & coeff_seq
            for start in range(0, seq_len, patch_size*2):
                # seq[start : start+patch_size*2] is the concat of base_patch & coeff_patch
                base_seq.extend(seq[start : start+patch_size])
                coeff_patches.append(seq[start+patch_size : start+patch_size*2])
            
            collected_base_seqs.append(base_seq)

            for patch in coeff_patches:
                # approx_{level}
                coeff_len = patch_size // (2**level)
                coeffs[0].extend(patch[:coeff_len])

                # detail_{level} -> detail_{1}
                for i in range(1, level+1):
                    coeffs[i].extend(patch[coeff_len : coeff_len*2])
                    coeff_len = coeff_len*2
            
            # [[coeffs of seq1], [coeffs of seq2], ...]
            collected_coeffs.append(coeffs)
        
        # turn back to [[a_{layer} of seqs], [d_{layer} of seqs], ..., [d_1 of seqs]]
        # ie. organized in coeff type
        # to keep consistent with the return of pywt.wavedec() 
        ret_coeff = [[] for _ in range(level+1)]
        for seq_coeffs in collected_coeffs:
            for i in range(len(seq_coeffs)):
                ret_coeff[i].append(seq_coeffs[i])
        
        for i in range(len(ret_coeff)):
                ret_coeff[i] = np.asarray(ret_coeff[i])
        
        collected_base_seqs = np.asarray(collected_base_seqs)
            
        return collected_base_seqs, ret_coeff

    
