import os
import sys
import argparse
import pathlib
import yaml
import subprocess

import numpy as np
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import pandas as pd

# from validate_ICME import validate
# from power import extract_power

serial_dict = {'ICMR':'0123456789ABCDEF'}
test_batch = 1  # don't change it for effective evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation performacne by benchmark model with NNAPI',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_path', type=str, default=None, help='path to tflite model')
    parser.add_argument('--test_data', type=str, default='datasets/ICME2022_Training_Dataset/images_real_world', help='path to dir of test images')
    parser.add_argument('--label', type=str,     default='datasets/ICME2022_Training_Dataset/labels_real_world', help='path to dir of test images')
    args = parser.parse_args()

    return args


def extract_power(path_to_csv):
    power_col = 3
    invaild_power_lower_bound = 100.    # mW

    df = pd.read_csv(path_to_csv, header=None, index_col=0)
    end_time = df.tail(1).index.item()
    start_time = end_time - 30. if end_time > 30. else 0.
    start_index = df.index.get_loc(start_time, method='nearest')

    power = df[power_col].to_numpy()
    power = power[start_index:]
    power = power[power > invaild_power_lower_bound]

    return power.mean() if len(power) > 0 else 0


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds): # ?
            # print(lt.shape)
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def calculate_miou(args):  
    totalItemCount = 0
    totalMeanIoU = 0
    n_classes = 6
    output_batch = []
    label_batch = []

    # load trained model
    interpreter = tf.lite.Interpreter(args.file_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(output)
    #input()
    # check test data and labels
    assert os.path.isdir(args.label) and os.path.isdir(args.test_data), \
    f"File folder missing/invaild: '{args.label}' or '{args.test_data}' is wrong"
    test_images = sorted(os.listdir(args.test_data))
    test_labels = sorted(os.listdir(args.label))
    assert len(os.listdir(args.label)) == len(test_images) and len(os.listdir(args.label)) > 0,\
    f"# images in visualize and labels should be same"

    for idx, (img_name, label_name) in enumerate(zip(test_images, test_labels), 1):

        if os.path.isdir(img_name):            
            continue

        # get prediction of segmentation
        path = os.path.join(args.test_data, img_name)
        img = cv2.imread(path)
        assert img is not None, f'{os.path.join(args.test_data, img_name)} is NoneType'
        img = cv2.resize(img, (1920, 1080))
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        gt = cv2.imread(os.path.join(args.label, label_name), cv2.IMREAD_GRAYSCALE)
        assert gt is not None, f'{os.path.join(args.label, label_name)} is NoneType'
        gt = cv2.resize(gt, (1920, 1080))

        output_batch.append(output)
        label_batch.append(np.expand_dims(gt , 0))
        # totalItemCount += 1
        
        if idx % test_batch == 0:
            output_batch = np.concatenate(output_batch, axis=0)
            label_batch = np.concatenate(label_batch, axis=0)
            
            running_metrics = runningScore(n_classes)            
            assert output_batch.shape == label_batch.shape, f'mismatch shape of gt and output: {label_batch.shape} vs. {output_batch.shape}'
            # print('output_batch.shape: ',output_batch.shape)
            running_metrics.update(label_batch, output_batch)
            score, class_iou = running_metrics.get_scores()           
            totalMeanIoU += score.get("Mean IoU : \t")
            break

        print('mIOU progress: {:3}/{:3}'.format(idx, len(test_images)), end='\r')
    print ("\n\n*** Mean mean Iou: {:.3f} ***\n".format(totalMeanIoU))

    return totalMeanIoU


def run_on_phone(args):
    BENCHMARK_MODEL='/data/local/tmp/benchmark_model_plus_neuron_delegate_210721'
    ADB = 'adb -s %s'%(serial_dict['ICMR'])
    MODEL = pathlib.Path(args.file_path)
    path_power_csv = os.path.join(MODEL.parent, 'power.csv')
    if not os.path.exists(path_power_csv):
        os.system("touch "+path_power_csv)
        # print("create ", path_power_csv)
    
    os.system('%s root'%(ADB))
    os.system('%s wait-for-device'%(ADB))
    os.system('%s shell rm /sdcard/Android/data/com.mediatek.simplePM/files/Documents/simplePM/power.csv'%(ADB))
    # os.system('%s push ../power.csv /sdcard/Android/data/com.mediatek.simplePM/files/Documents/simplePM/'%(ADB))  #?power
    os.system('%s push %s /sdcard/Android/data/com.mediatek.simplePM/files/Documents/simplePM/'%(ADB, path_power_csv))    #  power
    os.system('%s shell settings put global stay_on_while_plugged_in 3'%(ADB)) 
    os.system('%s shell am start -n com.mediatek.simplePM/.MainActivity'%(ADB))
    os.system('%s shell "echo 0 1 > /proc/mtk_battery_cmd/current_cmd"'%(ADB))
    os.system('%s shell input tap 200 400'%(ADB)) 

    # upload model on the platform
    os.system('%s shell mkdir -p /data/local/tmp/MAI22/%s' % (ADB, MODEL.stem))
    os.system('%s push %s /data/local/tmp/MAI22' % (ADB, MODEL))

    # show log
    os.system('%s shell setprop debug.nn.vlog 1'%(ADB))

    # execute the model w/ BENCHMARK_MODEL
    os.system('%s shell chmod +x %s'%(ADB, BENCHMARK_MODEL))
    os.system('%s logcat -c'%ADB)
    
    instr = '%s shell %s --graph=/data/local/tmp/MAI22/%s \
                --min_secs=30 \
                --enable_op_profiling=true \
                --use_nnapi=true\
                --nnapi_allow_fp16=true \
                --nnapi_execution_preference=fast_single_answer \
                --profiling_output_csv_file=/data/local/tmp/MAI22/%s/output.csv'%(
                    ADB, BENCHMARK_MODEL, MODEL.name, MODEL.stem
                )
    output = subprocess.check_output(instr, shell=True)
    output = output.decode("utf-8")
    # output = output.split('\n')[-6]
    # output = output.split(' ')[-2]
    # latency = output.split('=')[-1]
    # print(f"\n*** avg latency: {latency} ***\n")
    ptr = output.find("at least 30")
    res = output[ptr:].split('\n')[1]
    res = res.split(' ')[-2]
    latency = res.split('=')[-1]
    print(f"\n*** avg latency: {latency} ***\n")

    # print(f"\n*** adb on phone:\n{output}\n ***\n")
    

    # os.system(instr)

    os.system('%s shell input tap 200 800'%(ADB))
    os.system('%s pull /sdcard/Android/data/com.mediatek.simplePM/files/Documents/simplePM/power.csv %s'%(ADB, MODEL.parent))
    power = extract_power(os.path.join(MODEL.parent, 'power.csv'))
    print("\n*** avg power: {:.2f} ***\n".format(power))

    # os.system('%s logcat -d > %s/log.csv'%(adbCmd, output_dir))    # for power?adbCmd isn't defined.
    os.system('%s logcat -d > %s/log.csv'%(ADB, MODEL.parent))

    # os.system('%s pull /data/local/tmp/MAI22/%s/output.csv %s'%
    #             (ADB, MODEL.stem, MODEL.parent))                  # mark
    
    os.system('%s shell rm -f /data/local/tmp/MAI22/%s'%(ADB, MODEL.name))
    os.system('%s shell rm -rR /data/local/tmp/MAI22/%s'%(ADB, MODEL.stem))
    
    os.system('%s shell "echo 0 0 > /proc/mtk_battery_cmd/current_cmd"'%(ADB))

    # cwd = os.getcwd()
    # group = cwd.split('/')[3]
    # os.system('cp %s/log.csv ../ml2022_final/group/%s/'%(MODEL.parent, group))
    # os.system('cp %s/output.csv ../ml2022_final/group/%s/'%(MODEL.parent, group))
    # os.system('cp %s/power.csv ../ml2022_final/group/%s/'%(MODEL.parent, group))

    return latency, power


def record_on_board(result):
    group_number = os.path.expanduser('~').split('group')[-1]
    result['group'] = group_number
    print("\ngropu number: ", group_number)
    with open(f'../result/result_{group_number}.yaml', 'w') as f:
        yaml.dump(result, f)

    result_dir = '../result/'
    results = []
    for yml in os.listdir(result_dir):              
        try:
            with open(os.path.join(result_dir, yml), 'r') as f:
                res = yaml.safe_load(f)

                if res is None:
                    # print(os.path.join(result_dir, yml), 'None!')
                    continue

                if not (len(res) == 4 and isinstance(res['latency'], float) and \
                isinstance(res['power'], float) and isinstance(res['mIOU'], float) and \
                isinstance(res['group'], str)):
                    # print(len(res) == 4)
                    # print( isinstance(res['latency'], float))
                    # print(isinstance(res['power'], float))
                    # print(isinstance(res['mIOU'], float) )
                    # print(isinstance(res['group'], int))
                    print(os.path.join(result_dir, yml), 'wired!')
                    continue
                
                results.append(res)
        except:
            # print(os.path.join(result_dir, yml), 'fails!')
            continue
        
    results.sort(key=lambda yml: yml['latency'])
    rank_latency = pd.DataFrame(results)[['group', 'latency']]

    results.sort(key=lambda yml: yml['power'])
    rank_power = pd.DataFrame(results)[['group', 'power']]

    results.sort(key=lambda yml: yml['mIOU'], reverse=True)
    rank_miou = pd.DataFrame(results)[['group', 'mIOU']]

    print('*** latency ranking [ranking, group id, value]\n', rank_latency)
    print('\n*** power ranking   [ranking, group id, value]\n', rank_power)
    print('\n*** mIOU ranking    [ranking, group id, value]\n', rank_miou)
  
    
def main(args):
    #print('\n*** evaluate latency & power on the phone...\n')
    #latency, power = run_on_phone(args)

    print('\n*** calculating mIOU...\n')
    miou = calculate_miou(args)

    #print("*** overall results [ latency: {}, power: {:.3f}, mIOU: {:.3f} ] ***".format(latency, power, miou))
    #result = {"latency": float(latency), "power": float(power), "mIOU": float(miou)}
    #record_on_board(result)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
