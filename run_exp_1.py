import os
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    import importlib

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" #prevents "CUDA error: unspecified launch failure" and is recommended for some illegal memory access errors #increases train time by ~5-15%

    for experiment in [139,145]:

        configname = (f'experiments.configs.config{experiment}')
        config = importlib.import_module(configname)

        print('Starting experiment #',experiment, 'on', config.dataset, 'dataset')
        runs = 1
        
        if experiment in [139,145]:
            resume = True
        else:
            resume = False

        for run in range(runs):
            print("Training run #",run)
            if config.combine_train_corruptions:
                print('Combined training')
                cmd0 = f"python experiments/train.py --resume={resume} --run={run} --experiment={experiment} --epochs=" \
                       f"{config.epochs} --learningrate={config.learningrate} --dataset={config.dataset} --validontest=" \
                       f"{config.validontest} --lrschedule={config.lrschedule} --lrparams=\"{config.lrparams}\" " \
                       f"--earlystop={config.earlystop} --earlystopPatience={config.earlystopPatience} --optimizer=" \
                       f"{config.optimizer} --optimizerparams=\"{config.optimizerparams}\" --modeltype=" \
                       f"{config.modeltype} --modelparams=\"{config.modelparams}\" --resize={config.resize} " \
                       f"--train_aug_strat_orig={config.train_aug_strat_orig} " \
                       f"--train_aug_strat_gen={config.train_aug_strat_gen} --loss=" \
                       f"{config.loss} --lossparams=\"{config.lossparams}\" --trades_loss={config.trades_loss} " \
                       f"--trades_lossparams=\"{config.trades_lossparams}\" --robust_loss={config.robust_loss} " \
                       f"--robust_lossparams=\"{config.robust_lossparams}\" --mixup=\"{config.mixup}\" --cutmix=" \
                       f"\"{config.cutmix}\" --manifold=\"{config.manifold}\" --combine_train_corruptions=" \
                       f"{config.combine_train_corruptions} --concurrent_combinations={config.concurrent_combinations}" \
                       f" --batchsize={config.batchsize} --number_workers={config.number_workers} " \
                       f"--RandomEraseProbability={config.RandomEraseProbability} --warmupepochs={config.warmupepochs}" \
                       f" --normalize={config.normalize} --pixel_factor={config.pixel_factor} --minibatchsize=" \
                       f"{config.minibatchsize} --validonc={config.validonc} --validonadv={config.validonadv} --swa=" \
                       f"\"{config.swa}\" --noise_sparsity={config.noise_sparsity} --noise_patch_scale=" \
                       f"\"{config.noise_patch_scale}\" --generated_ratio={config.generated_ratio} "
                os.system(cmd0)
            else:
                for id, (train_corruption) in enumerate(config.train_corruptions):
                    print("Separate corruption training:", train_corruption)
                    cmd0 = f"python experiments/train.py --resume={resume} --train_corruptions=\"{train_corruption}\"" \
                           f" --run={run} --experiment={experiment} --epochs={config.epochs} " \
                           f"--learningrate={config.learningrate} --dataset={config.dataset} --validontest=" \
                           f"{config.validontest} --lrschedule={config.lrschedule} --lrparams=\"{config.lrparams}\" " \
                           f"--earlystop={config.earlystop} --earlystopPatience={config.earlystopPatience} --optimizer=" \
                           f"{config.optimizer} --optimizerparams=\"{config.optimizerparams}\" --modeltype=" \
                           f"{config.modeltype} --modelparams=\"{config.modelparams}\" --resize={config.resize} " \
                           f"--train_aug_strat_orig={config.train_aug_strat_orig} " \
                            f"--train_aug_strat_gen={config.train_aug_strat_gen} --loss=" \
                           f"{config.loss} --lossparams=\"{config.lossparams}\" --trades_loss={config.trades_loss} " \
                           f"--trades_lossparams=\"{config.trades_lossparams}\" --robust_loss={config.robust_loss} " \
                           f"--robust_lossparams=\"{config.robust_lossparams}\" --mixup=\"{config.mixup}\" --cutmix=" \
                           f"\"{config.cutmix}\" --manifold=\"{config.manifold}\" --combine_train_corruptions=" \
                           f"{config.combine_train_corruptions} --concurrent_combinations={config.concurrent_combinations}" \
                           f" --batchsize={config.batchsize} --number_workers={config.number_workers} " \
                           f"--RandomEraseProbability={config.RandomEraseProbability} --warmupepochs={config.warmupepochs}" \
                           f" --normalize={config.normalize} --pixel_factor={config.pixel_factor} --minibatchsize=" \
                           f"{config.minibatchsize} --validonc={config.validonc} --validonadv={config.validonadv} --swa=" \
                           f"\"{config.swa}\" --noise_sparsity={config.noise_sparsity} --noise_patch_scale=" \
                           f"\"{config.noise_patch_scale}\" --generated_ratio={config.generated_ratio} "
                    os.system(cmd0)

        # Calculate accuracy and robust accuracy, evaluating each trained network on each corruption

        print('Beginning metric evaluation')
        cmdeval = "python experiments/eval.py --resume={} --experiment={} --runs={} --batchsize={} --dataset={} " \
                "--modeltype={} --modelparams=\"{}\" --resize={} --combine_test_corruptions={} --number_workers={} " \
                "--normalize={} --pixel_factor={} --test_on_c={} --calculate_adv_distance={} --adv_distance_params=\"{}\" " \
                "--calculate_autoattack_robustness={} --autoattack_params=\"{}\" --combine_train_corruptions={} " \
                .format(resume, experiment, runs, 1000, config.dataset, config.modeltype, config.modelparams,
                        config.resize, config.combine_test_corruptions, 0, config.normalize,
                        config.pixel_factor, config.test_on_c, config.calculate_adv_distance, config.adv_distance_params,
                        config.calculate_autoattack_robustness, config.autoattack_params, config.combine_train_corruptions)
        os.system(cmdeval)

