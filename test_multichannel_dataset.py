import pytest
import numpy as np
from multichannel_dataset import MultiChannelDataset
import os
from PIL import Image


def test_input_creating_dataset():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    ds = MultiChannelDataset(input, mask)
    assert np.array_equal(ds.input, np.array(input).transpose())


def test_mask_creating_dataset():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    ds = MultiChannelDataset(input, mask)
    assert np.array_equal(ds.output, np.array(mask).transpose())


def test_uneven_input_channel_length():
    input = [['a', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    with pytest.raises(ValueError):
        MultiChannelDataset(input, mask)


def test_uneven_mask_channel_length():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    with pytest.raises(ValueError):
        MultiChannelDataset(input, mask)


def test_uneven_channel_length_between_mask_and_input():
    input = [['a', 'c'], ['c', 'e'], ['f', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    with pytest.raises(ValueError):
        MultiChannelDataset(input, mask)


def test_number_of_input_channels_correct():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    ds = MultiChannelDataset(input, mask)
    assert ds.num_input_channels == 3


def test_number_of_mask_channels_correct():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    ds = MultiChannelDataset(input, mask)
    assert ds.num_output_channels == 3


def test_correct_length_is_given():
    input = [['a', 'b', 'c'], ['c', 'd', 'e'], ['f', 'g', 'h']]
    mask = [['ma', 'mb', 'mc'], ['mc', 'md', 'me'], ['mf', 'mg', 'mh']]
    ds = MultiChannelDataset(input, mask)
    assert len(ds) == 3


def test_load_multichannel_images():
    input_test_dir = './test_images/input'
    output_test_dir = './test_images/output'
    inputs = create_test_multichannel_images(input_test_dir, n_channels=5, n_images=10)
    outputs = create_test_multichannel_images(output_test_dir, n_channels=2, n_images=10)

    ds = MultiChannelDataset(inputs, outputs)
    item = ds[0]

    img0_channel_0_input = np.array(Image.open(os.path.join(input_test_dir, 'channel_000', 'channel_00_img000.png')))
    img0_channel_1_input = np.array(Image.open(os.path.join(input_test_dir, 'channel_001', 'channel_01_img000.png')))
    img0_channel_2_input = np.array(Image.open(os.path.join(input_test_dir, 'channel_002', 'channel_02_img000.png')))
    img0_channel_3_input = np.array(Image.open(os.path.join(input_test_dir, 'channel_003', 'channel_03_img000.png')))
    img0_channel_4_input = np.array(Image.open(os.path.join(input_test_dir, 'channel_004', 'channel_04_img000.png')))

    img0_channel_0_output = np.array(Image.open(os.path.join(output_test_dir, 'channel_000', 'channel_00_img000.png')))
    img0_channel_1_output = np.array(Image.open(os.path.join(output_test_dir, 'channel_001', 'channel_01_img000.png')))

    test_input_array = np.stack([img0_channel_0_input, img0_channel_1_input,
                                 img0_channel_2_input, img0_channel_3_input,
                                 img0_channel_4_input], axis=0)

    test_output_array = np.stack([img0_channel_0_output,
                                  img0_channel_1_output], axis=0)

    item_input = item['input']
    item_output = item['output']
    assert (item_input.shape == (5, 128, 128))
    assert (item_output.shape == (2, 128, 128))
    print(ds.input[0])
    assert (np.array_equal(item_input, test_input_array))
    assert (np.array_equal(item_output, test_output_array))


def create_test_multichannel_images(dir, n_channels, n_images, size=(128, 128)):
    channels = []
    for i in range(n_channels):
        channel_dir = os.path.join(dir, f'channel_{i:03d}')
        os.makedirs(channel_dir, exist_ok=True)
        channel = []
        for j in range(n_images):
            img_fp = os.path.join(channel_dir, f'channel_{i:02d}_img{j:03d}.png')
            channel.append(img_fp)
            Image.fromarray(np.random.randint(0, 256, size=size, dtype=np.uint8)).save(img_fp)
        channels.append(channel)
    return channels
