import os
import random
import torch
from logging import Logger


class DatasetLoader:
    def __init__(self, shard_directory: str, batch_size: int, val_split: float, embd_dropout: float):
        """

        :param shard_directory: Path to dir that holds the shard files
        :param batch_size: Batch size
        :param val_split: Value between 0 and 0.5. If 10 shard files and val_split=0.2, then 10 * 0.2 = 2 val shards
        :param embd_dropout: Value between 0 and 0.5. Replaces CLIP embedding with null embedding accordingly
        """
        self.batch_size = batch_size
        self.embd_dropout = embd_dropout

        filepaths = [os.path.join(shard_directory, p) for p in os.listdir(shard_directory) if p.endswith(".pt")]
        filepaths = sorted(filepaths)

        assert 0 < val_split < 0.5
        assert 0 <= embd_dropout < 0.5
        assert len(filepaths) >= 2  # Should have at least 2 shards. 1 for train and 1 for val

        n = len(filepaths) - max(1, int(len(filepaths) * val_split))  # Total - num_val_shards
        self.train_filepaths = filepaths[:n]
        self.val_filepaths = filepaths[n:]

        self.train_epoch = 0
        self.val_epoch = 0

        self.train_file_idx = 0
        self.val_file_idx = 0

        self.train_idx = 0
        self.val_idx = 0

        self.train_latent = None
        self.train_embd = None
        self.val_latent = None
        self.val_embd = None
        self.null_embd = None

        self._load_data(train=True)
        self._load_data(train=False)

    def _load_data(self, train: bool):
        if train:
            if self.train_file_idx == len(self.train_filepaths):
                self.train_epoch += 1
                self.train_file_idx = 0
                random.shuffle(self.train_filepaths)

            data_dict = torch.load(self.train_filepaths[self.train_file_idx], map_location="cpu")
            self.train_latent = data_dict["latents"]
            self.train_embd = data_dict["text_embd"]

            if self.null_embd is None:  # Only one time
                null_embd = data_dict["null_embd"]
                if len(null_embd.shape) == 3:  # Leftover artifact with batch dim for 1 example. Fine to leave for now
                    null_embd = null_embd.squeeze(0)
                self.null_embd = null_embd

            # Shuffle the order in-shards
            idx = torch.randperm(len(self.train_latent))
            self.train_latent = self.train_latent[idx]
            self.train_embd = self.train_embd[idx]

            self.train_file_idx += 1
        else:
            if self.val_file_idx == len(self.val_filepaths):
                self.val_epoch += 1
                self.val_file_idx = 0

            data_dict = torch.load(self.val_filepaths[self.val_file_idx], map_location="cpu")
            self.val_latent = data_dict["latents"]
            self.val_embd = data_dict["text_embd"]
            self.val_file_idx += 1

    def log_info(self, logger: Logger, effective_batch_size: int):
        logger.info("-" * 30)
        logger.info("DATASET METADATA")
        logger.info(f"Train shards:        {len(self.train_filepaths)}")
        logger.info(f"Val shards:          {len(self.val_filepaths)}")
        logger.info(f"Est. Train examples: {len(self.train_filepaths) * self.train_latent.shape[0]:,}")
        logger.info(f"Est. Val examples:   {len(self.val_filepaths) * self.val_latent.shape[0]:,}")
        logger.info(f"Batch Size:          {self.batch_size}")
        logger.info(f"Steps per Epoch:     {len(self.train_filepaths) * self.train_latent.shape[0] // effective_batch_size}")
        logger.info(f"CFG Dropout:         {self.embd_dropout * 100}%")
        logger.info(f"Latent Shape:        {self.train_latent.shape}")
        logger.info(f"Text Embd Shape:     {self.train_embd.shape}")
        logger.info(f"Null Embd Shape:     {self.null_embd.shape}")
        logger.info("-" * 30)

    def get_batch(self, train: bool):
        if train:
            if self.train_idx + self.batch_size > len(self.train_latent):
                self._load_data(train=True)
                self.train_idx = 0

            latent_data = self.train_latent[self.train_idx: self.train_idx + self.batch_size]
            embd_data = self.train_embd[self.train_idx: self.train_idx + self.batch_size]
            self.train_idx += self.batch_size
        else:
            if self.val_idx + self.batch_size > len(self.val_latent):
                self._load_data(train=False)
                self.val_idx = 0

            latent_data = self.val_latent[self.val_idx: self.val_idx + self.batch_size]
            embd_data = self.val_embd[self.val_idx: self.val_idx + self.batch_size]
            self.val_idx += self.batch_size

        if train and self.embd_dropout > 0:
            embd_data = embd_data.clone()
            B = embd_data.shape[0]
            drop_mask = torch.rand(B) < self.embd_dropout
            embd_data[drop_mask] = self.null_embd

        # Return latents, text_embd of shape (B, 4, H//8, W//8) and (B, 77, 512) respectively
        # Training loop can handle device cuda/cpu
        return latent_data, embd_data
