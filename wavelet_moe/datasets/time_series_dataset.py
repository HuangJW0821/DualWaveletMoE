#!/usr/bin/env python
# -*- coding:utf-8 _*-
from abc import abstractmethod
import json
import os
import pickle
import gzip
import yaml
import numpy as np

# Copied from TimeMoE/time_moe/datasets


class TimeSeriesDataset:
    """
    Base class of dataset used in WaveletMoE
    """
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, seq_idx):
        pass

    @abstractmethod
    def get_num_tokens(self):
        pass

    @abstractmethod
    def get_sequence_length_by_idx(self, seq_idx):
        pass

    @staticmethod
    def is_valid_path(data_path):
        return True

    def __iter__(self):
        n_seqs = len(self)
        for i in range(n_seqs):
            yield self[i]


class GeneralDataset(TimeSeriesDataset):
    """Dataset class for [`json`, `jsonl`, `npy`, `npy.gz`, `pkl`] dataset"""
    def __init__(self, data_path):
        """Dataset class for [`json`, `jsonl`, `npy`, `npy.gz`, `pkl`] dataset"""
        self.data = self._read_file_by_extension(data_path)
        self.num_tokens = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, seq_idx):
        seq = self.data[seq_idx]
        if isinstance(seq, dict):
            seq = seq['sequence']
        return seq

    def get_num_tokens(self):
        if self.num_tokens is None:
            self.num_tokens = sum([len(seq) for seq in self])
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        seq = self[seq_idx]
        return len(seq)

    @staticmethod
    def is_valid_path(data_path):
        if os.path.exists(data_path) and os.path.isfile(data_path):
            parts = data_path.split('.')
            if len(parts) == 0:
                return False
            suffix = parts[-1]
            if suffix in ('json', 'jsonl', 'npy', 'npy.gz', 'pkl'):
                return True
            else:
                return False
        else:
            return False

    def _read_file_by_extension(self, data_path):
        if data_path.endswith('.json'):
            with open(data_path, encoding='utf-8') as file:
                data = json.load(file)
        elif data_path.endswith('.jsonl'):
            data = self._read_jsonl_to_list(data_path)
        elif data_path.endswith('.yaml'):
            data = self._load_yaml_file(data_path)
        elif data_path.endswith('.npy'):
            data = np.load(data_path, allow_pickle=True)
        elif data_path.endswith('.npz'):
            data = np.load(data_path, allow_pickle=True)
        elif data_path.endswith('.npy.gz'):
            with gzip.GzipFile(data_path, 'r') as file:
                data = np.load(file, allow_pickle=True)
        elif data_path.endswith('.pkl') or data_path.endswith('.pickle'):
            data = self._load_pkl_obj(data_path)
        else:
            raise RuntimeError(f'Unknown file extension: {data_path}')
        return data

    def _read_jsonl_to_list(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file.readlines()]

    def _load_yaml_file(self, data_path):
        if isinstance(data_path, str):
            with open(data_path, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config
        else:
            return data_path

    def _load_pkl_obj(self, data_path):
        out_list = []
        with open(data_path, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    out_list.append(data)
                except EOFError:
                    break
        if len(out_list) == 0:
            return None
        elif len(out_list) == 1:
            return out_list[0]
        else:
            return out_list


class BinaryDataset(TimeSeriesDataset):
    """Dataset class for bin dataset"""
    meta_file_name = 'meta.json'
    bin_file_name_template = 'data-{}-of-{}.bin'

    def __init__(self, data_path):
        """Dataset class for bin dataset"""
        if not self.is_valid_path(data_path):
            raise ValueError(f'Folder {data_path} is not a valid WaveletMoE dataset.')

        self.data_path = data_path

        # load meta file
        meta_file_path = os.path.join(data_path, self.meta_file_name)
        try:
            self.meta_info = self._load_json_file(meta_file_path)
        except Exception as e:
            print(f'Error when loading file {meta_file_path}: {e}')
            raise e

        self.num_sequences = self.meta_info['num_sequences']
        self.dtype = self.meta_info['dtype']
        self.seq_infos = self.meta_info['scales']

        # process the start index for each file
        self.file_start_idxes = []
        s_idx = 0
        for fn, length in sorted(self.meta_info['files'].items(), key=lambda x: int(x[0].split('-')[1])):
            self.file_start_idxes.append(
                (os.path.join(data_path, fn), s_idx, length)
            )
            s_idx += length
        self.num_tokens = s_idx

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, seq_idx):
        seq_info = self.seq_infos[seq_idx]
        read_info_list = self._get_read_infos_by_offset_length(seq_info['offset'], seq_info['length'])
        out = []
        for fn, offset_in_file, length in read_info_list:
            out.append(self._read_sequence_in_file(fn, offset_in_file, length))

        if len(out) == 1:
            sequence = out[0]
        else:
            sequence = np.concatenate(out, axis=0)

        if 'mean' in seq_info and 'std' in seq_info:
            return sequence * seq_info['std'] + seq_info['mean']
        else:
            return sequence

    def get_num_tokens(self):
        return self.num_tokens

    def get_sequence_length_by_idx(self, seq_idx):
        return self.seq_infos[seq_idx]['length']

    def _get_read_infos_by_offset_length(self, offset, length):
        # just use naive search
        binary_read_info_list = []
        end_offset = offset + length
        for fn, start_idx, fn_length in self.file_start_idxes:
            end_idx = start_idx + fn_length
            if start_idx <= offset < end_idx:
                if end_offset <= end_idx:
                    binary_read_info_list.append([fn, offset - start_idx, length])
                    break
                else:
                    binary_read_info_list.append([fn, offset - start_idx, end_idx - offset])
                    length = end_offset - end_idx
                    offset = end_idx
        return binary_read_info_list

    def _read_sequence_in_file(self, fn, offset_in_file, length):
        sentence = np.empty(length, dtype=self.dtype)
        with open(fn, mode='rb', buffering=0) as file_handler:
            file_handler.seek(offset_in_file * sentence.itemsize)
            file_handler.readinto(sentence)
        return sentence

    @staticmethod
    def is_valid_path(data_path):
        if (os.path.exists(data_path)
                and os.path.isdir(data_path)
                and os.path.exists(os.path.join(data_path, 'meta.json'))
        ):
            for sub in os.listdir(data_path):
                # TODO check if lack bin file
                if os.path.isfile(os.path.join(data_path, sub)) and sub.endswith('.bin'):
                    return True
        return False

    def _load_json_file(self, data_path):
        with open(data_path, encoding='utf-8') as file:
            data = json.load(file)
            return data