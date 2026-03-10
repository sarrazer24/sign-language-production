import torch
from torch.utils.data import DataLoader

def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['pose_length'], reverse=True)

    pose_lengths = torch.LongTensor([item['pose_length'] for item in batch])
    T_max = pose_lengths[0].item()
    B     = len(batch)
    K     = batch[0]['poses'].shape[1]

    # Padding poses
    poses_padded = torch.zeros(B, T_max, K, 3)
    pose_mask    = torch.zeros(B, T_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        T = item['pose_length']
        poses_padded[i, :T] = item['poses']
        pose_mask[i, :T]    = True

    # Padding texte
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [item['input_ids'] for item in batch],
        batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [item['attention_mask'] for item in batch],
        batch_first=True, padding_value=0
    )

    return {
        'poses'         : poses_padded,    # (B, T_max, K, 3)
        'pose_mask'     : pose_mask,        # (B, T_max)
        'pose_lengths'  : pose_lengths,     # (B,)
        'input_ids'     : input_ids,        # (B, seq_len)
        'attention_mask': attention_mask,
        'texts'         : [item['text'] for item in batch]
    }