import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import os


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    checkpoint=None,
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    if checkpoint is not None:
        current_epoch = checkpoint['epoch'] + 1  # Get the saved epoch number
        metrics = checkpoint['metrics']
        best_valid_loss = checkpoint['loss']
    else:
        current_epoch = 1
        best_valid_loss = 1e10
        metrics = {
            "train": {
                "loss": [],
            },
            "val": {
                "loss": [],
            }
        }

    epoch_range = range(current_epoch, config["epochs"] + 1)
    for epoch_no in epoch_range:
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        metrics["train"]["loss"].append(avg_loss / batch_no)

        if valid_loader is not None and (epoch_no) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                torch.save({
                'epoch': epoch_no,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you want to save optimizer states as well
                'loss': best_valid_loss,
                'metrics': metrics,
                }, foldername + "/best_model.pth")

                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                
            metrics["val"]["loss"].append(avg_loss_valid / batch_no)

        if foldername != "":
            # Save model and epoch
            torch.save({
                'epoch': epoch_no,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # Optional, if you want to save optimizer states as well
                'loss': best_valid_loss,
                'metrics': metrics,
            }, output_path)
        


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def compute_mse_per_feature(predicted_means, c_target, eval_points):
    """
    Compute the MSE per feature and the total average MSE.

    Args:
        predicted_means (torch.Tensor): Predicted means of shape (B, L, K).
        c_target (torch.Tensor): Target values of shape (B, L, K).
        eval_points (torch.Tensor): Mask of shape (B, L, K) with 1 for validation points and 0 otherwise.

    Returns:
        mse_per_feature (torch.Tensor): MSE per feature (shape: K).
        total_average_mse (float): Total average MSE across all features.
    """
    # Compute the squared errors
    squared_errors = ((predicted_means - c_target) ** 2) * eval_points

    # Sum over batch (B) and timesteps (L) dimensions for each feature
    total_squared_error_per_feature = squared_errors.sum(dim=(0, 1))

    # Count the number of valid points for each feature
    total_eval_points_per_feature = eval_points.sum(dim=(0, 1))

    # Compute the MSE per feature
    mse_per_feature = total_squared_error_per_feature / total_eval_points_per_feature

    # Compute the total average MSE across all features
    total_average_mse = mse_per_feature.mean().item()

    return mse_per_feature, total_average_mse


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    num_features = 4
    with torch.no_grad():
        model.eval()
        mse_total = 0
        evalpoints_total = 0

        all_crps = []
        all_mse = []
        all_mse2 = []
        # all_target = []
        # all_evalpoint = []
        # all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                
                crps = torch.zeros(num_features)
                mse = torch.zeros(num_features)
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,T,K)
                c_target = c_target.permute(0, 2, 1)  # (B,T,K)
                eval_points = eval_points.permute(0, 2, 1) # (B,T,K)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1) # (B,T,K)
                # all_target.append(c_target.cpu())
                # all_evalpoint.append(eval_points.cpu())
                # all_generated_samples.append(samples.cpu())

                print('\nBatch: ', batch_no)
                # print('C-target shape: ', c_target.shape)
                # print('Eval_points shape: ', eval_points.shape)
                # print('Samples shape: ', samples.shape)
                # print('All Target shape: ', np.shape(all_target))
                # print('All Eval_points shape: ', np.shape(all_evalpoint))
                # print('All Samples shape: ', np.shape(all_generated_samples))

                for feature_idx in range(num_features):
                    crps[feature_idx] = calc_quantile_CRPS(c_target[:, :, feature_idx], samples[:, :, :, feature_idx], eval_points[:, :, feature_idx], mean_scaler, scaler)

                    tmp_mse = ((((samples_median.values[:, :, feature_idx] - c_target[:, :, feature_idx]) * eval_points[:, :, feature_idx]) ** 2) * (scaler ** 2)).sum().item()
                    tmp_eval_points = eval_points.sum().item()

                    mse[feature_idx] = tmp_mse / tmp_eval_points

                mse2, useless = compute_mse_per_feature(samples_median.values, c_target, eval_points)
                all_crps.append(crps)
                all_mse.append(mse)
                all_mse2.append(mse2.cpu())


                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)

                mse_total += mse_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mse_total": mse_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                if batch_no == 1:
                    break

            # all_target = torch.cat(all_target, dim=0)
            # all_evalpoint = torch.cat(all_evalpoint, dim=0)
            # all_generated_samples = torch.cat(all_generated_samples, dim=0)

            # print('After Concatenating')
            # print('All Target shape: ', np.shape(all_target))
            # print('All Eval_points shape: ', np.shape(all_evalpoint))
            # print('All Samples shape: ', np.shape(all_generated_samples))

            # crps = calc_quantile_CRPS(
            #     all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            # )

            # results = dict(
            #     mse=np.sqrt(mse_total / evalpoints_total),
            #     crps=crps,
            # )

            # print("MSE:", results['mse'])
            # print("CRPS:", results['crps'])

            print('CRPS per feature: ', np.mean(all_crps, axis=0))
            print('CRPS total: ', np.mean(all_crps))
            print('MSE per feature: ', np.mean(all_mse, axis=0))
            print('MSE total: ', np.mean(all_mse))
            print('MSE2 per feature: ', np.mean(all_mse2, axis=0))
            print('MSE2 total: ', np.mean(all_mse2))

        # return results

# import torch
# import numpy as np
# from tqdm import tqdm

# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
#     num_features = 4
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0
#         evalpoints_total = 0

#         all_crps = []
#         all_mse = []
#         all_mse2 = []
#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 print('Batch Number: ', batch_no)
#                 output = model.evaluate(test_batch, nsample)
#                 samples, c_target, eval_points, observed_points, observed_time = output

#                 samples = samples.permute(0, 1, 3, 2)  # (B, nsample, T, K)
#                 c_target = c_target.permute(0, 2, 1)  # (B, T, K)
#                 eval_points = eval_points.permute(0, 2, 1) # (B, T, K)

#                 samples_median = samples.median(dim=1).values # (B, T, K)
#                 batch_crps = torch.zeros(num_features)
#                 batch_mse = torch.zeros(num_features)

#                 print('\nBatch: ', batch_no)

#                 for feature_idx in range(num_features):
#                     batch_crps[feature_idx] = calc_quantile_CRPS_optimized(
#                         c_target[:, :, feature_idx],
#                         samples[:, :, :, feature_idx],
#                         eval_points[:, :, feature_idx],
#                         mean_scaler,
#                         scaler
#                     )

#                     tmp_mse_per_feature = (
#                         ((samples_median[:, :, feature_idx] - c_target[:, :, feature_idx]) * eval_points[:, :, feature_idx]) ** 2
#                     ).sum()
#                     batch_mse[feature_idx] = (tmp_mse_per_feature * (scaler ** 2)) / eval_points.sum()

#                 mse2, _ = compute_mse_per_feature(samples_median, c_target, eval_points)
#                 all_crps.append(batch_crps)
#                 all_mse.append(batch_mse)
#                 all_mse2.append(mse2.cpu())

#                 mse_current = (
#                     ((samples_median - c_target) * eval_points) ** 2
#                 ) * (scaler ** 2)

#                 mse_total += mse_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()

#                 it.set_postfix(
#                     ordered_dict={
#                         "mse_total": mse_total / evalpoints_total,
#                         "batch_no": batch_no,
#                     },
#                     refresh=True,
#                 )

#             print('CRPS per feature: ', np.mean(torch.stack(all_crps).cpu().numpy(), axis=0))
#             print('CRPS total: ', np.mean(torch.stack(all_crps).cpu().numpy()))
#             print('MSE per feature: ', np.mean(torch.stack(all_mse).cpu().numpy(), axis=0))
#             print('MSE total: ', np.mean(torch.stack(all_mse).cpu().numpy()))
#             print('MSE2 per feature: ', np.mean(torch.stack(all_mse2).cpu().numpy(), axis=0))
#             print('MSE2 total: ', np.mean(torch.stack(all_mse2).cpu().numpy()))

# def calc_quantile_CRPS_optimized(target, forecast, eval_points, mean_scaler, scaler):
#     target = target * scaler + mean_scaler
#     forecast = forecast * scaler + mean_scaler

#     quantiles = torch.arange(0.05, 1.0, 0.05, device=target.device)
#     denom = calc_denominator(target, eval_points)
#     CRPS = 0
#     q_pred = torch.quantile(forecast, quantiles[None, None, :], dim=1) # (B, T, num_quantiles)
#     for i in range(len(quantiles)):
#         q_loss = quantile_loss(target, q_pred[:, :, i], quantiles[i], eval_points)
#         CRPS += q_loss / denom
#     return CRPS.item() / len(quantiles)

# def quantile_loss(target, forecast, q, eval_points):
#     return (eval_points * (q * torch.relu(target - forecast) + (1 - q) * torch.relu(forecast - target))).sum()

# def calc_denominator(target, eval_points):
#     return eval_points.sum() # Assuming denominator is the same for all quantiles and features

    

def plot_test(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                ## CODE ADDED BY ME UNTIL BREAK FOR TESTING
                # Select the first example from the first batch
                max_features = 4
                first_sample_idx = 0
                generated_samples = samples[first_sample_idx].cpu()  # (nsample, L, K)
                target = c_target[first_sample_idx].cpu()   # (L, K)
                eval_mask = eval_points[first_sample_idx].cpu()   # (L, K)
                observed_mask = observed_points[first_sample_idx].cpu()   # (L, K)
                observed_time = observed_time[first_sample_idx].cpu()   # (L)

                # Median and standard deviation of generated samples
                predicted_values = generated_samples.median(dim=0).values  # (L, K)
                std_values = generated_samples.std(dim=0).numpy()  # (L, K)

                # Compute 2-sigma confidence interval bounds
                upper_bound = predicted_values + 2 * std_values  # (L, K)
                lower_bound = predicted_values - 2 * std_values  # (L, K)

                # Determine the number of features to plot
                num_features = min(target.size(1), max_features)

                # Set up subplots
                fig, axes = plt.subplots(num_features, 1, figsize=(10, 4 * num_features), sharex=True)
                if num_features == 1:  # Ensure axes is iterable if there's only one feature
                    axes = [axes]

                time_steps = torch.arange(target.size(0))

                for feature_idx in range(num_features):
                    ax = axes[feature_idx]
                    # ax.plot(
                    #     time_steps, 
                    #     target[:, feature_idx] * scaler, 
                    #     'g-', 
                    #     label=f"Ground Truth (Feature {feature_idx})", 
                    #     s=5,
                    #     alpha=0.6
                    # )

                    # indices = np.where(np.abs(predicted_values[:, feature_idx]) < 1)
                    # tmp = predicted_values[:, feature_idx]
                    # print('this many non wacko predicted', len(time_steps[indices]))
                    # ax.scatter(
                    #     time_steps[indices], 
                    #     tmp[indices],  
                    #     label=f"Predicted without big (Feature {feature_idx})", 
                    #     alpha=0.8,
                    #     s=3,
                    #     color='purple'
                    # )

                    # for tmp_sample in generated_samples:
                    #     big_tmp = tmp_sample[:, feature_idx]
                    #     ax.scatter(
                    #         time_steps[indices],
                    #         big_tmp[indices],
                    #         color='red',
                    #         alpha=0.2,
                    #         s=1
                    #     )

                    # Plot confidence interval as a shaded region
                    ax.fill_between(
                        time_steps[eval_mask[:, feature_idx].bool()],
                        lower_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                        upper_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                        color='red',
                        alpha=0.3,
                        label="2-Sigma Confidence Interval"
                    )

                    ax.scatter(
                        time_steps[eval_mask[:, feature_idx].bool()], 
                        predicted_values[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,  
                        label=f"Predicted Mean (Feature {feature_idx})",
                        s=3,
                        color='red'
                    )

                    ax.scatter(
                        time_steps[observed_mask[:, feature_idx].bool()],
                        target[observed_mask[:, feature_idx].bool(), feature_idx] * scaler,
                        c="blue",
                        label="Observed",
                        s=3,
                    )
                    # print('this many to eval', len(time_steps[eval_mask[:, feature_idx].bool()]))
                    ax.scatter(
                        time_steps[eval_mask[:, feature_idx].bool()],
                        target[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                        c="orange",
                        label="Masked",
                        s=3,
                    )
                    ax.set_title(f"Feature {feature_idx}")
                    ax.set_ylabel("Values")
                    ax.legend(loc="upper right", fontsize=8)
                    ax.grid(True)

                plt.xlabel("Time Steps")
                plt.tight_layout()
                plt.savefig('test.png')
                plt.show()

                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            crps = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data

import torch

import torch

def inspect_data(model, test_loader, example_idx, nsample):
    dataset = test_loader.dataset
    example = dataset[example_idx]
    single_example_loader = torch.utils.data.DataLoader([example], batch_size=1)

    with torch.no_grad():
        model.eval()

        for test_batch in single_example_loader:
            print("--- Test Batch Inspection ---")
            print("Test Batch Keys:", test_batch.keys()) #prints the keys of the dictionary.
            print("Observed Data Shape:", test_batch["observed_data"].shape)
            print("Observed Data:", test_batch["observed_data"])
            print("Observed Data Type:", test_batch["observed_data"].dtype)
            print("Observed Mask Shape:", test_batch["observed_mask"].shape)
            print("Observed Mask:", test_batch["observed_mask"])
            print("Observed Mask Type:", test_batch["observed_mask"].dtype)
            print("Timepoints Shape:", test_batch["timepoints"].shape)
            print("Timepoints:", test_batch["timepoints"])
            print("Timepoints Type:", test_batch["timepoints"].dtype)

            # Assuming your model's evaluate function can handle the dictionary.
            output = model.evaluate(test_batch, nsample)
            print("--- Model Output Inspection ---")
            print("Model Output Shape:", output.shape)
            print("Model Output Data:", output)
            print("Model Output Data Type:", output.dtype)

def plot_test_for_example(model, test_loader, example_idx, nsample=100, scaler=1, mean_scaler=0, foldername="", y_labels=None, save_path="diffusion_results.npz"):
    # inspect_data(model, test_loader, example_idx, nsample)
    # exit()
    # Access the dataset directly
    dataset = test_loader.dataset
    
    # Extract the specific example
    example = dataset[example_idx]

    # Wrap the single example in a DataLoader to keep the rest of the code consistent
    single_example_loader = torch.utils.data.DataLoader([example], batch_size=1)

    with torch.no_grad():
        model.eval()
        
        for test_batch in single_example_loader:
            output = model.evaluate(test_batch, nsample)

            # Your existing plotting code goes here, updated to handle the single example
            samples, c_target, eval_points, observed_points, observed_time = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)
            observed_points = observed_points.permute(0, 2, 1)

            # Since batch_size=1, select the only example (index 0 in the batch)
            generated_samples = samples[0].cpu()  # (nsample, L, K)
            target = c_target[0].cpu()  # (L, K)
            eval_mask = eval_points[0].cpu()  # (L, K)
            observed_mask = observed_points[0].cpu()  # (L, K)
            observed_time = observed_time[0].cpu()  # (L)

            # Compute median, standard deviation, and confidence intervals
            predicted_values = generated_samples.median(dim=0).values  # (L, K)
            std_values = generated_samples.std(dim=0)  # (L, K)
            upper_bound = predicted_values + 2 * std_values
            lower_bound = predicted_values - 2 * std_values

            # Determine the number of features to plot
            num_features = min(target.size(1), 4)

            time_steps = torch.arange(target.size(0))

            saved_data = {}  # Dictionary to store data for saving

            for feature_idx in range(num_features):
                # Extract observed and masked data
                obs_indices = time_steps[observed_mask[:, feature_idx].bool()].numpy()
                obs_y = target[observed_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler
                masked_indices = time_steps[eval_mask[:, feature_idx].bool()].numpy()
                masked_y = target[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler
                pred_means = predicted_values[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler
                pred_lower = lower_bound[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler
                pred_upper = upper_bound[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler

                # Save data
                key_label = y_labels[feature_idx] if y_labels else f"Channel_{feature_idx+1}"
                saved_data[f"{key_label}_train_x"] = obs_indices
                saved_data[f"{key_label}_train_y"] = obs_y
                saved_data[f"{key_label}_test_x"] = masked_indices
                saved_data[f"{key_label}_test_y"] = masked_y
                saved_data[f"{key_label}_predicted_means"] = pred_means
                saved_data[f"{key_label}_lower_bound"] = pred_lower
                saved_data[f"{key_label}_upper_bound"] = pred_upper

                if feature_idx == 0:
                    print('eval_mask[:, feature_idx].bool()', eval_mask[:, feature_idx].bool())
                    print('predicted_values[eval_mask[:, feature_idx].bool(), feature_idx].numpy()', predicted_values[eval_mask[:, feature_idx].bool(), feature_idx].numpy())
                    print('predicted_values[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler', predicted_values[eval_mask[:, feature_idx].bool(), feature_idx].numpy() * scaler)

                tmp_mse = (np.sum(np.square((target[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy() - (predicted_values[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy()))) / len((target[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy())
                print(f'MSE for key {y_labels[feature_idx]} is: {tmp_mse}')

            # Save the results
            np.savez(save_path, **saved_data)
            print(f"Results saved to {save_path}")

            # Plot results
            fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True, gridspec_kw={'hspace': 0})
            if num_features == 1:
                axes = [axes]

            for feature_idx in range(num_features):
                ax = axes[feature_idx]
                # Plot ground truth, predicted mean, and confidence intervals
                ax.scatter(
                    time_steps[observed_mask[:, feature_idx].bool()],
                    target[observed_mask[:, feature_idx].bool(), feature_idx] * scaler,
                    c="#E6C229",
                    label="Observed",
                    s=3,
                )
                ax.scatter(
                    time_steps[eval_mask[:, feature_idx].bool()],
                    target[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                    c="#1B998B",
                    label="Masked",
                    s=3,
                )
                ax.scatter(
                    time_steps[eval_mask[:, feature_idx].bool()],
                    predicted_values[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                    label=f"Predicted Mean",
                    s=3,
                    color="#DF2935",
                )
                ax.fill_between(
                    time_steps[eval_mask[:, feature_idx].bool()],
                    lower_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                    upper_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
                    color="#DF2935",
                    alpha=0.2,
                    label=r'2-$\sigma$',
                )

                if y_labels is not None:
                    ax.set_ylabel(y_labels[feature_idx])
                else:
                    ax.set_ylabel(f"Feature {feature_idx}")

                # ax.set_ylim(-3.5, 7.5)
                if feature_idx == 0:
                    ax.legend(loc="upper right")
                ax.grid(True)

            # Set x-axis label and overall title
            axes[-1].set_xlabel("Timesteps")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save or show plot
            if foldername:
                plt.savefig(f"{foldername}/example_{example_idx}.png")
            plt.show()

# def plot_test_for_example(model, test_loader, example_idx, nsample=100, scaler=1, mean_scaler=0, foldername="", y_labels=None):
    # # Access the dataset directly
    # dataset = test_loader.dataset
    
    # # Extract the specific example
    # example = dataset[example_idx]

    # # Wrap the single example in a DataLoader to keep the rest of the code consistent
    # single_example_loader = torch.utils.data.DataLoader([example], batch_size=1)

    # with torch.no_grad():
    #     model.eval()
        
    #     for test_batch in single_example_loader:
    #         output = model.evaluate(test_batch, nsample)

    #         # Your existing plotting code goes here, updated to handle the single example
    #         samples, c_target, eval_points, observed_points, observed_time = output
    #         samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
    #         c_target = c_target.permute(0, 2, 1)  # (B,L,K)
    #         eval_points = eval_points.permute(0, 2, 1)
    #         observed_points = observed_points.permute(0, 2, 1)

    #         # Since batch_size=1, select the only example (index 0 in the batch)
    #         generated_samples = samples[0].cpu()  # (nsample, L, K)
    #         target = c_target[0].cpu()  # (L, K)
    #         eval_mask = eval_points[0].cpu()  # (L, K)
    #         observed_mask = observed_points[0].cpu()  # (L, K)
    #         observed_time = observed_time[0].cpu()  # (L)

    #         # Compute median, standard deviation, and confidence intervals
    #         predicted_values = generated_samples.median(dim=0).values  # (L, K)
    #         std_values = generated_samples.std(dim=0)  # (L, K)
    #         upper_bound = predicted_values + 2 * std_values
    #         lower_bound = predicted_values - 2 * std_values

    #         # Determine the number of features to plot
    #         num_features = min(target.size(1), 4)

    #         # Plot results
    #         fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True, gridspec_kw={'hspace': 0})
    #         if num_features == 1:
    #             axes = [axes]

    #         time_steps = torch.arange(target.size(0))

    #         for feature_idx in range(num_features):
    #             ax = axes[feature_idx]
    #             # Plot ground truth, predicted mean, and confidence intervals
    #             ax.scatter(
    #                 time_steps[observed_mask[:, feature_idx].bool()],
    #                 target[observed_mask[:, feature_idx].bool(), feature_idx] * scaler,
    #                 c="#E6C229",
    #                 label="Observed",
    #                 s=3,
    #             )
    #             ax.scatter(
    #                 time_steps[eval_mask[:, feature_idx].bool()],
    #                 target[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
    #                 c="#1B998B",
    #                 label="Masked",
    #                 s=3,
    #             )
    #             ax.scatter(
    #                 time_steps[eval_mask[:, feature_idx].bool()],
    #                 predicted_values[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
    #                 label=f"Predicted Mean",
    #                 s=3,
    #                 color="#DF2935",
    #             )
    #             ax.fill_between(
    #                 time_steps[eval_mask[:, feature_idx].bool()],
    #                 lower_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
    #                 upper_bound[eval_mask[:, feature_idx].bool(), feature_idx] * scaler,
    #                 color="#DF2935",
    #                 alpha=0.2,
    #                 label=r'2-$\sigma$',
    #             )

    #             tmp_mse = (np.sum(np.square((target[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy() - (predicted_values[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy()))) / len((target[eval_mask[:, feature_idx].bool(), feature_idx]).cpu().numpy())
    #             print(f'MSE for key {y_labels[feature_idx]} is: {tmp_mse}')

    #             if y_labels is not None:
    #                 ax.set_ylabel(y_labels[feature_idx])
    #             else:
    #                 ax.set_ylabel(f"Feature {feature_idx}")

    #             ax.set_ylim(-3.5, 7.5)
    #             if feature_idx == 0:
    #                 ax.legend(loc="upper right")
    #             ax.grid(True)

    #         # Set x-axis label and overall title
    #         axes[-1].set_xlabel("Timesteps")
    #         plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    #         # Save or show plot
    #         if foldername:
    #             plt.savefig(f"{foldername}/example_{example_idx}.png")
    #         plt.show()



def backup(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    num_features = 4
    with torch.no_grad():
        model.eval()
        mse_total = 0
        evalpoints_total = 0

        all_target = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target.cpu())
                all_evalpoint.append(eval_points.cpu())
                all_generated_samples.append(samples.cpu())

                print('Batch,=: ', batch_no)
                print('C-target shape: ', c_target.shape)
                print('Eval_points shape: ', eval_points.shape)
                print('Samples shape: ', samples.shape)
                print('All Target shape: ', np.shape(all_target))
                print('All Eval_points shape: ', np.shape(all_evalpoint))
                print('All Samples shape: ', np.shape(all_generated_samples))

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)

                mse_total += mse_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "mse_total": mse_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

                if batch_no == 2:
                    break

            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            print('After Concatenating')
            print('All Target shape: ', np.shape(all_target))
            print('All Eval_points shape: ', np.shape(all_evalpoint))
            print('All Samples shape: ', np.shape(all_generated_samples))

            crps = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            results = dict(
                mse=np.sqrt(mse_total / evalpoints_total),
                crps=crps,
            )

            print("MSE:", results['mse'])
            print("CRPS:", results['crps'])

        return results