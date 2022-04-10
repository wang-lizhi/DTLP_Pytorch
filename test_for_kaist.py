import os
import numpy as np
import torch
import utils
import spectral as spy
import scipy.io as scio
import train_for_cave.model as model

# parse command line input
opt = utils.parse_arg()

# set parameters
dataPath = opt.testset_path
model_path = opt.pretrained_path
save_path = './results/'
channel = opt.channel
patch_len = opt.blocksize
stride = 24

# Set GPU
opt.cuda = opt.gpuid>=0
if opt.cuda:
    torch.cuda.set_device(opt.gpuid)
else:
    raise NotImplementedError

# save the test results for each image
psnr_list = []
ssim_list = []
sam_list = []

# extract mask
mask_mat = scio.loadmat('./mask.mat')
mask = mask_mat['mask']  #mask ∈ [0, 1]
# mask = np.round(mask_mat['mask'])  #mask ∈ {0, 1}
mask = np.expand_dims(mask, axis=2)
mask = np.tile(mask, [1, 1, channel])

# load the model
model = model.prepare_model(opt)
model.load_state_dict(torch.load(model_path,  map_location='cuda:0') )

# get the list of testset
pics = os.listdir(dataPath)
for pic_name in pics:
    input_path = dataPath + pic_name 
    result_path = save_path + pic_name[:-4] + '/'
    print(pic_name)

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    # extract data
    mat_data = scio.loadmat(input_path)
    img = mat_data['img']
    gt = img

    #roll the  image to be tested
    img_roll = np.zeros(img.shape, dtype='float32')
    for ch in range(channel):
        img_roll[:,:,ch] = np.roll(img[:, :, ch], shift = -ch, axis = 0)

    # divide into patches
    patch_list = []
    mask_list = []
    row_max = (img_roll.shape[0] - patch_len)//stride + 1
    col_max = (img_roll.shape[1] - patch_len)//stride + 1
    for i in range(row_max + 1):    #row
        if i < row_max :
            for j in range(col_max + 1):    #col
                if j < col_max :
                    patch_list.append(img_roll[ stride * i : stride * i + patch_len, stride * j : stride * j + patch_len, :])
                    mask_list.append(mask[ stride * i : stride * i + patch_len, stride * j : stride * j + patch_len, :])
                else:
                    patch_list.append(img_roll[ stride * i : stride * i + patch_len, img_roll.shape[1] - patch_len : img_roll.shape[1], :])
                    mask_list.append(mask[ stride * i : stride * i + patch_len, img_roll.shape[1] - patch_len : img_roll.shape[1], :])
        else:
            for j in range(col_max + 1):    #col
                if j < col_max :
                    patch_list.append(img_roll[ img_roll.shape[0] - patch_len : img_roll.shape[0], stride * j : stride * j + patch_len, :])
                    mask_list.append(mask[ img_roll.shape[0] - patch_len : img_roll.shape[0], stride * j : stride * j + patch_len, :])
                else:
                    patch_list.append(img_roll[ img_roll.shape[0] - patch_len : img_roll.shape[0], img_roll.shape[1] - patch_len : img_roll.shape[1], :])
                    mask_list.append(mask[ img_roll.shape[0] - patch_len : img_roll.shape[0], img_roll.shape[1] - patch_len : img_roll.shape[1], :])

    input_data = torch.FloatTensor(patch_list).unsqueeze(0)
    input_mask = torch.FloatTensor(mask_list).unsqueeze(0)
    input_tensor = torch.cat((input_data, input_mask), 0).transpose(1, 0)

    # model interface
    pred_list = []
    psnr_cache = []
    model.eval()
    if opt.cuda:
        model.cuda()

    loader = torch.utils.data.DataLoader(input_tensor, 
                                            batch_size = 1, 
                                            shuffle=False, )

    for batch_idx, batch in enumerate(loader):
        data = batch[:, 0, :, :, :]
        mask_in = batch[0, 1, :, :, 0]
        with torch.no_grad():
            if opt.cuda:
                data = data.cuda()
            hsi_pred = model(data, mask_in) 
            hsi_pred, data = hsi_pred.cpu(), data.cpu()
            pred_list.append(np.array(hsi_pred))
            data = np.array(data)
            hsi_pred = np.array(hsi_pred)
            #calculate psnr
            for i in range(len(data)):
                psnr_cache.append(utils.Cal_PSNR_by_default(data[i],hsi_pred[i]))

    # calculate the average PSNR of the total validset
    patch_PSNR = sum(psnr_cache)/len(psnr_cache)
    print('The average PSNR of patches is:' + str(patch_PSNR))

    # get the prediction
    predict_list = np.concatenate(pred_list, 0)

    # joint the patches and remove the image boundaries
    row_list = []
    for i in range(row_max + 1):    #row
        row_data = []
        row_data.append(predict_list[ i * (col_max + 1) + 0][:, 0:24, :])
        for a in range(8):
            row_data.append((predict_list[ i * (col_max + 1) + a][:, 24:48, :] + predict_list[ i * (col_max + 1) + a +1][:, 0:24, :]) / 2)
        row_data.append((predict_list[ i * (col_max + 1) + 8][:, 24:48, :] + predict_list[ i * (col_max + 1) + 9][:, 8:32, :]) / 2)
        row_data.append(predict_list[ i * (col_max + 1) + 9][:, 32:48, :] )
        row_data_arr = np.concatenate(row_data, 1)
        row_list.append(row_data_arr)

    half_row_list = []
    half_row_list.append(row_list[0][0:stride])
    for i in range(8):
        half_row_list.append((row_list[i][stride::] + row_list[i + 1][0:stride])/2)
    half_row_list.append((row_list[8][stride::] + row_list[9][8:32]) / 2)
    half_row_list.append(row_list[9][32::])

    predict_img = np.concatenate(half_row_list, 0)

    #unroll
    output = np.zeros(predict_img.shape, dtype='float32')
    for ch in range(channel):
        output[:,:,ch] = np.roll(predict_img[:, :, ch], shift = ch, axis = 0)

    gt = gt.astype('float32')
    output = output.astype('float32')

    # calculate the psnr
    PSNR = utils.Cal_PSNR_by_default(gt, output)
    psnr_list.append(PSNR)
    print('PSNR:' + str(PSNR))

    #calculate the ssim
    SSIM = utils.Cal_SSIM(gt, output)
    ssim_list.append(SSIM)
    print('SSIM:' + str(SSIM))

    #calculate the sam
    SAM = utils.Cal_SAM(gt, output)
    sam_list.append(SAM)
    print('SAM:' + str(SAM))

    # save rgb images
    hsi2rgb_mask = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3.08373333333324e-05,0.000808629333333337,0.00231250000000000,0.00676812333333329,0.0110854920000000,0.0123992618666667,0.0166835344000001,0.0211791360000000,0.0200000000000000,0.0188145920000000,0.0171802530000000],
                                        [0.00133000000000000,0.00176000000000000,0.00219838700000000,0.00263067733333333,0.00324473600000000,0.00424691666666667,0.00530156800000000,0.00625349342857142,0.00804886857142856,0.0113580952380952,0.0124558190000000,0.0129867990000000,0.0142776960000000,0.0153179610000000,0.0159507916666667,0.0165760625000000,0.0181894760000000,0.0196792568888889,0.0136953208888889,0.0114479166666667,0.0100280153333333,0.00746627600000002,0.00280633600000001,0.00100000000000000,0.00100000000000000,0.00100000000000000,0.00100000000000000,0.00100000000000000], 
                                        [0.0291752640000000,0.0269502720000000,0.0255224880000000,0.0247204083809524,0.0234726400000000,0.0211318750000000,0.0178115968000000,0.0126948450571429,0.00834112000000001,0.00500800000000000,0.00195278080000001,7.16800000000704e-07,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    for i in range(3):
        sum_row = sum(hsi2rgb_mask[i])
        for j in range(28):
            hsi2rgb_mask[i][j] = hsi2rgb_mask[i][j] / sum_row
    
    gt_rgb = np.matmul(gt, np.transpose(hsi2rgb_mask, [1, 0]))
    output_rgb = np.matmul(output, np.transpose(hsi2rgb_mask, [1, 0]))

    spy.save_rgb(result_path + pic_name[:-4] + '_output.png', output_rgb)
    spy.save_rgb(result_path + pic_name[:-4] + '_gt.png', gt_rgb)


# calculate the average results
psnr_avg = sum(psnr_list)/len(psnr_list)
print('psnr_average:  ' + str(psnr_avg))

ssim_avg = sum(ssim_list)/len(ssim_list)
print('ssim_average:  ' + str(ssim_avg))

sam_avg = sum(sam_list)/len(sam_list)
print('sam_average:  ' + str(sam_avg))