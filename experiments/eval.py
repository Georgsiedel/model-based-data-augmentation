import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
module_path = os.path.abspath(os.path.dirname(__file__))

if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchmetrics.classification import MulticlassCalibrationError
import argparse
import importlib
from run_exp import device
import models
import eval_adversarial
import eval_corruptions
import data
import utils

parser = argparse.ArgumentParser(description='PyTorch Robustness Testing')
parser.add_argument('--runs', default=1, type=int, help='run number')
parser.add_argument('--experiment', default=0, type=int,
                    help='experiment number - each experiment is defined in module config{experiment}')
parser.add_argument('--batchsize', default=1000, type=int,
                    help='Images per batch - more means quicker training, but higher memory demand')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset to choose')
parser.add_argument('--validontest', type=utils.str2bool, nargs='?', const=True, default=True, help='True: Use full '
                    'training data and test data. False: 80:20 train:valiation split, validation also used for testing.')
parser.add_argument('--modeltype', default='wideresnet', type=str,
                    help='Modeltype to train, use either default WRN28 or model from pytorch models')
parser.add_argument('--modelparams', default={}, type=str, action=utils.str2dictAction, metavar='KEY=VALUE',
                    help='parameters for the chosen model')
parser.add_argument('--resize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Resize a model to 224x224 pixels, standard for models like transformers.')
parser.add_argument('--combine_test_corruptions', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to combine all testing noise values by drawing from the randomly')
parser.add_argument('--number_workers', default=0, type=int, help='How many workers are launched to parallelize data loading.')
parser.add_argument('--normalize', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to normalize input data to mean=0 and std=1')
parser.add_argument('--test_on_c', type=utils.str2bool, nargs='?', const=True, default=True,
                    help='Whether to test on corrupted benchmark sets C and C-bar')
parser.add_argument('--calculate_adv_distance', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial distance with adv_distance_params')
parser.add_argument('--adv_distance_params', default={'setsize': 500, 'nb_iters': 200, 'eps_iter': 0.0003, 'norm': 'np.inf',
                        "epsilon": 0.1, "clever": True, "clever_batches": 50, "clever_samples": 50},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')
parser.add_argument('--calculate_autoattack_robustness', type=utils.str2bool, nargs='?', const=False, default=False,
                    help='Whether to calculate adversarial accuracy with Autoattack with autoattack_params')
parser.add_argument('--autoattack_params', default={'setsize': 500, 'epsilon': 8/255, 'norm': 'Linf'},
                    type=str, action=utils.str2dictAction, metavar='KEY=VALUE', help='parameters for the trades loss function')

args = parser.parse_args()
configname = (f'experiments.configs.config{args.experiment}')
config = importlib.import_module(configname)
test_corruptions = config.test_corruptions

def compute_clean(testloader, model, num_classes):
    with torch.no_grad():
        correct = 0
        total = 0
        calibration_metric = MulticlassCalibrationError(num_classes=num_classes, n_bins=15, norm='l2')
        all_targets = torch.empty(0)
        all_targets_pred = torch.empty((0, num_classes))
        all_targets, all_targets_pred = all_targets.to(device), all_targets_pred.to(device)

        for batch_idx, (inputs, targets) in enumerate(testloader):

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            with torch.amp.autocast('cuda'):
                targets_pred = model(inputs)

            _, predicted = targets_pred.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets = torch.cat((all_targets, targets), 0)
            all_targets_pred = torch.cat((all_targets_pred, targets_pred), 0)

        acc = 100.*correct/total
        rmsce_clean = float(calibration_metric(all_targets_pred, all_targets).cpu())
        print("Clean Accuracy ", acc, "%, RMSCE Calibration Error: ", rmsce_clean)

        return acc, rmsce_clean

if __name__ == '__main__':
    Testtracker = utils.TestTracking(args.dataset, args.modeltype, args.experiment, args.runs,
                                args.combine_test_corruptions, args.test_on_c,
                                args.calculate_adv_distance, args.calculate_autoattack_robustness,
                                test_corruptions, args.adv_distance_params)

    for run in range(args.runs):
        # Load data
        Dataloader = data.DataLoading(dataset=args.dataset, validontest=args.validontest, generated_ratio=0.0, resize=args.resize, 
                                      run=run, number_workers=args.number_workers)
        Dataloader.create_transforms(train_aug_strat_orig='None', train_aug_strat_gen='None')
        Dataloader.load_base_data(test_only=True)
        workers = 0 if args.validontest else args.number_workers
        
        testloader = torch.utils.data.DataLoader(Dataloader.testset, batch_size=args.batchsize, pin_memory=True, num_workers=workers)
            
        Testtracker.initialize(run)

        # Load model
        model_class = getattr(models, args.modeltype)
        model = model_class(dataset=args.dataset, normalized=args.normalize, num_classes=Dataloader.num_classes,
                            factor=Dataloader.factor, **args.modelparams)
        model = model.to(device) #torch.nn.DataParallel(

        state_dict = torch.load(Testtracker.filename, weights_only=True)['model_state_dict']

        # Remove "module." prefix from keys
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = v

        # Load the modified state_dict into the original model
        model.load_state_dict(new_state_dict, strict=False)

        model.eval()

        # Clean Test Accuracy
        acc, rmsce = compute_clean(testloader, model, Dataloader.num_classes)
        Testtracker.track_results([acc, rmsce])

        if args.test_on_c == True:  # C-dataset robust accuracy
            subset = False if args.validontest else True
            subsetsize = None if args.validontest else 1000 #subset for quick validation runs

            testsets_c = Dataloader.load_data_c(subset=subset, subsetsize=subsetsize, valid_run=False)
            accs_c = eval_corruptions.compute_c_corruptions(args.dataset, testsets_c, model, args.batchsize,
                                                            Dataloader.num_classes, valid_run=False, workers=workers)
            Testtracker.track_results(accs_c)

        if args.calculate_adv_distance == True:  # adversarial distance calculation
            adv_acc, dist_sorted, mean_dist = eval_adversarial.compute_adv_distance(Dataloader.testset,
                                                            0, model, args.adv_distance_params)
            Testtracker.track_results(np.concatenate(([adv_acc], mean_dist)))
            Testtracker.save_adv_distance(dist_sorted, args.adv_distance_params)

        if args.calculate_autoattack_robustness == True:  # adversarial accuracy calculation
            adv_acc_aa, mean_dist_aa = eval_adversarial.compute_adv_acc(args.autoattack_params, Dataloader.testset,
                                                                        model, 0, args.batchsize)
            Testtracker.track_results([adv_acc_aa, mean_dist_aa])

        # Robust Accuracy on p-norm noise - either combined or separate noise types
        accs = eval_corruptions.select_p_corruptions(testloader, model, test_corruptions, args.dataset, args.combine_test_corruptions)
        Testtracker.track_results(accs)

        print(Testtracker.accs)

    Testtracker.create_report()