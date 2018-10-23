from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class MultiChannelDataset(Dataset):
    def __init__(self, input, output, transform=None):
        '''
        A pytorch Dataset that supports multichannel inputs and outputs. returns as a dictionary of numpy arrays (if no
         transform converts it to something else) of size (C,H,W) where C is the number of channels, H is the height
         and W is the width of the image.
        Args:
        :param input (list): a list of lists of input channel file paths
        :param output (list): a list of lists of outputs file paths
        :param transform (Compose): a list of transformations currently none supported TODO


        # Examples
        input_filenames = [
                    ['input/channel_0_img1.png',
                    'input/channel_0_img2.png',
                    'input/channel_0.img3png'],

                    ['input/channel_1_img1.png',
                    'input/channel_1_img2.png',
                    'input/channel_1.img3png'],

                    ['input/channel_2_img1.png',
                    'input/channel_2_img2.png',
                    'input/channel_2.img3png'],

                    ['input/channel_3_img1.png',
                    'input/channel_3_img2.png',
                    'input/channel_3.img3png']
                    ]
        output_filenames = [
            ['output/channel_0_img1.png',
            'output/channel_0_img2.png',
            'output/channel_0.img3png'],

            ['output/channel_1_img1.png',
            'output/channel_1_img2.png',
            'output/channel_1.img3png']
            ]

        my_ds = MultiChannelDataset(input_filenames, output_filenames)

        # to get a single item:
        my_ds[0]

        # to use in a DataLoader
        DataLoader(my_ds,batch_size=5,...)

        '''

        # Checks to make sure that the input channels and output channels have
        # the same number of entries
        for idx, (channel_input, channel_output) in enumerate(zip(input, output)):
            if len(channel_input) != len(input[0]):
                raise ValueError(f'input channel {idx} does not have the same length as the other channels')
            if len(channel_output) != len(output[0]):
                raise ValueError(f'output channel {idx} does not have the same length as the other channels')
            if len(channel_output) != len(channel_input):
                raise ValueError(f'the output and input channel {idx} have differing lengths')

        self.input = np.array(input).transpose()  # transpose for easier indexing in __getitem__
        self.output = np.array(output).transpose()  # transpose for easier indexing in __getitem__

        self.num_input_channels = len(input)
        self.num_output_channels = len(output)
        self.transform = transform

    def __len__(self):
        assert len(self.input[0]) == len(self.output[0])
        return len(self.input[0])

    def __getitem__(self, idx):
        item = {'input': self.load_multichannel_image(self.input[idx]),
                'output': self.load_multichannel_image(self.output[idx])}
        if self.transform:
            item = self.transform(item)
        return item

    @staticmethod
    def load_multichannel_image(file_name_list):
        ''' expects a list of single channel images'''
        channel_list = []
        for idx in range(len(file_name_list)):
            channel_list.append(np.array(Image.open(file_name_list[idx])))
        return np.stack(channel_list, axis=0)
