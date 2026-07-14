from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from .preprocessing import RandomDownsample
import numpy as np

class BaseDataset(object):
    logger = logging.getLogger("CRST.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
        self.random_downsample = RandomDownsample()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img_hr = self.transform(img)
        else:
            img_hr = img

        if torch.is_tensor(img_hr):
            img_lr, res_label = self.random_downsample(img_hr)
        else:
            img_lr = img_hr
            res_label = 0

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img_hr,
            'images_lr': img_lr, 
            'res_label': res_label,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None, res_label=0):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform
        self.res_label = res_label

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        current_res_label = self.res_label
        if isinstance(current_res_label, (list, tuple, np.ndarray, torch.Tensor)):
            current_res_label = current_res_label[index]
        
        return {
            'pids': pid,
            'images': img,
            'res_label': torch.tensor(current_res_label, dtype=torch.long)
        }


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        return {
            'pids': pid,
            'caption_ids': caption
        }


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()
        self.random_downsample = RandomDownsample()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img_hr = self.transform(img)
        else:
            img_hr = img
        if torch.is_tensor(img_hr):
            img_lr, res_label = self.random_downsample(img_hr)
        else:
            img_lr = img_hr
            res_label = 0
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_resolution_conditioned_masked_tokens_and_labels(
            caption_tokens.cpu().numpy(), res_label
        )

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img_hr,      # HR
            'images_lr': img_lr,   # LR
            'res_label': res_label,# Label
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _clean_bpe_piece(self, token_id):
        piece = self.tokenizer.decoder.get(int(token_id), '')
        return piece.replace('</w>', '').lower()

    def _is_attribute_piece(self, piece):
        attr_words = {
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'pink',
            'purple', 'brown', 'gray', 'grey', 'dark', 'light', 'bright', 'blond',
            'shirt', 'tshirt', 't-shirt', 'top', 'coat', 'jacket', 'hoodie', 'sweater',
            'dress', 'skirt', 'pants', 'trousers', 'jeans', 'shorts', 'suit', 'uniform',
            'shoe', 'shoes', 'sneaker', 'sneakers', 'boot', 'boots', 'sandal', 'hat',
            'cap', 'helmet', 'bag', 'backpack', 'handbag', 'purse', 'umbrella',
            'glasses', 'hair', 'beard', 'mask', 'scarf', 'tie', 'belt', 'sleeve',
            'striped', 'stripe', 'plaid', 'checked', 'checkered', 'pattern', 'logo',
            'print', 'denim', 'leather', 'cotton', 'long', 'short', 'sleeveless'
        }
        return piece in attr_words

    def _resolution_attribute_ratio(self, res_label):
        ratios = {0: 0.70, 1: 0.60, 2: 0.45, 3: 0.30}
        return ratios.get(int(res_label), 0.30)

    def _build_resolution_conditioned_masked_tokens_and_labels(self, tokens, res_label, mask_ratio=0.15):
        tokens = [int(t) for t in list(tokens)]
        original_tokens = list(tokens)
        labels = [0 for _ in tokens]
        mask_id = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))

        valid_positions = [i for i, t in enumerate(tokens) if 0 < t < 49405]
        if not valid_positions:
            return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

        attr_positions, coarse_positions = [], []
        for i in valid_positions:
            piece = self._clean_bpe_piece(tokens[i])
            if self._is_attribute_piece(piece):
                attr_positions.append(i)
            else:
                coarse_positions.append(i)

        total_to_mask = max(1, int(round(mask_ratio * len(valid_positions))))
        attr_ratio = self._resolution_attribute_ratio(res_label)
        num_attr = min(len(attr_positions), int(round(attr_ratio * total_to_mask)))
        num_coarse = min(len(coarse_positions), total_to_mask - num_attr)

        remaining = total_to_mask - num_attr - num_coarse
        if remaining > 0:
            attr_left = max(0, len(attr_positions) - num_attr)
            take_attr = min(attr_left, remaining)
            num_attr += take_attr
            remaining -= take_attr
        if remaining > 0:
            coarse_left = max(0, len(coarse_positions) - num_coarse)
            num_coarse += min(coarse_left, remaining)

        masked_positions = []
        if num_attr > 0:
            masked_positions.extend(random.sample(attr_positions, num_attr))
        if num_coarse > 0:
            masked_positions.extend(random.sample(coarse_positions, num_coarse))
        if not masked_positions:
            masked_positions = [random.choice(valid_positions)]

        for i in masked_positions:
            labels[i] = original_tokens[i]
            prob = random.random()
            if prob < 0.8:
                tokens[i] = mask_id
            elif prob < 0.9:
                tokens[i] = random.choice(token_range)

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
