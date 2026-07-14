from configs.config import parse_args
from datasets.generate_dataset import VideoDataset, split_json
from torch.utils.data import DataLoader


if __name__ == '__main__':
    config = parse_args()
    dataset = VideoDataset(config, img_to_tensor=True, batchify_all_views=False)
    dataloader = dataloader = DataLoader(dataset, batch_size=None, shuffle=False, pin_memory=False, drop_last=False)
    dataset.write(dataloader)

    split_json(dataset.tgt_folder)