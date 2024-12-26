from mini_batch_loader_color import *
from chainer import serializers
from MyFCN import *
from chainer import cuda, optimizers, Variable
import sys
import math
import time
import chainerrl
import State
import os
from pixelwise_a3c import *

#_/_/_/ paths _/_/_/ 
TRAINING_DATA_PATH          = "/content/pixelRL/training_BSD68.txt"
TESTING_DATA_PATH           = "/content/pixelRL/demo.txt"
IMAGE_DIR_PATH              = "/content/pixelRL/"
SAVE_PATH            = "/content/pixelRL/denoise/model/denoise_myfcn_"
 
#_/_/_/ training parameters _/_/_/ 
LEARNING_RATE    = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1 #must be 1
N_EPISODES           = 30000
EPISODE_LEN = 5
GAMMA = 0.95 # discount factor

#noise setting
MEAN = 0
SIGMA = 15

N_ACTIONS = 9
MOVE_RANGE = 3 #number of actions that move the pixel values. e.g., when MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
CROP_SIZE = 70

GPU_ID = 0

def test_color(loader, agent, fout):
    # Initialize metrics
    sum_psnr = 0
    sum_reward = 0

    # Count the number of test data samples
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)

    # Create output directory if not exists
    output_dir = '/content/pixelRL/resultimage/'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through test data in batches
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        # Load test data for the current batch
        raw_x = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))

        # Handle color images (split into channels)
        _, c, h, w = raw_x.shape

        # Initialize storage for denoised channels
        denoised_channels = []

        for channel_idx in range(c):
            # Process each channel independently
            raw_channel = raw_x[:, channel_idx:channel_idx+1, :, :]
            raw_n = np.random.normal(MEAN, SIGMA, raw_channel.shape).astype(raw_channel.dtype) / 255
            noisy_channel = raw_channel + raw_n
            # Initialize state for the channel
            current_state = State.State((TEST_BATCH_SIZE, 1, h, w), MOVE_RANGE)
            current_state.reset(raw_channel, noisy_channel)
            current_states.append(current_state)
            raw_channels.append(raw_channel)
            noisy_channels.append(noisy_channel[0, 0, :, :])

            # Iterative denoising process
            for t in range(EPISODE_LEN):
                for i in range(c):
                   action, inner_state = agent.act(current_states[i].tensor)
                   current_states[i].step(action, inner_state)
                   denoised_channel = np.maximum(0, current_states[i].image)
                   denoised_channel = np.minimum(1, denoised_channel)
                   denoised_channels.append(denoised_channel[0, 0, :, :]) # Remove the channel dimension

        # Recombine channels into a single image
        denoised_image = np.stack(denoised_channels, axis=1)  # Shape: (batch_size, channels, height, width)
        denoised_image_uint8 = (denoised_image[0] * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)

        # Original and noisy images for comparison
        original_image = raw_x[0].transpose(1, 2, 0)
        noisy_image = (raw_x[0] + raw_n[0]).transpose(1, 2, 0)
        original_image_uint8 = (np.clip(original_image, 0, 1) * 255 + 0.5).astype(np.uint8)
        noisy_image_uint8 = (np.clip(noisy_image, 0, 1) * 255 + 0.5).astype(np.uint8)

        # Save images
        cv2.imwrite(f'{output_dir}{i}_output.png', denoised_image_uint8)
        cv2.imwrite(f'{output_dir}{i}_input.png', noisy_image_uint8)
        print(cv2.PSNR(denoised_image_uint8, original_image_uint8))
        # Calculate PSNR and update metrics
        sum_psnr += cv2.PSNR(denoised_image_uint8, original_image_uint8)

    # Compute average metrics
    avg_psnr = sum_psnr / test_data_size

    # Print and log results
    result_message = f"Test PSNR (Color): {avg_psnr:.2f}"
    print(result_message)
    fout.write(result_message + "\n")
    fout.flush()


def main(fout):
    #_/_/_/ load dataset _/_/_/ 
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH, 
        TESTING_DATA_PATH, 
        IMAGE_DIR_PATH, 
        CROP_SIZE)
 
    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE,1,CROP_SIZE,CROP_SIZE), MOVE_RANGE)
 
    # load myfcn model
    model = MyFcn(N_ACTIONS)
 
    #_/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    chainer.serializers.load_npz('/content/pixelRL/denoise_with_convGRU_and_RMC/model/pretrained_15.npz', agent.model)
    agent.act_deterministically = True
    agent.model.to_gpu()

    #_/_/_/ testing _/_/_/
    test(mini_batch_loader, agent, fout)
    
     
 
if __name__ == '__main__':
    try:
        fout = open('testlog.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start)/60))
        print("{s}[h]".format(s=(end - start)/60/60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start)/60))
        fout.write("{s}[h]\n".format(s=(end - start)/60/60))
        fout.close()
    except Exception as error:
        print(error.message)
