import sys
import logging
import math
import time
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy.special import logsumexp
from ema import EMA
from model import *
from utils import *
from diffusion_utils import *

plt.style.use('ggplot')


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        self.num_timesteps = config.diffusion.timesteps
        self.vis_step = config.diffusion.vis_step
        self.num_figs = config.diffusion.num_figs
        self.dataset_object = None

        betas = make_beta_schedule(schedule=config.diffusion.beta_schedule, num_timesteps=self.num_timesteps,
                                   start=config.diffusion.beta_start, end=config.diffusion.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if config.diffusion.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # initial prediction model as guided condition
        self.cond_pred_model = None
        if config.diffusion.conditioning_signal == "OLS":
            # use training data to compute OLS beta parameter
            _, dataset = get_dataset(args, config, test_set=False)
            train_x = dataset[:, :-config.model.y_dim]
            # concat a column of 1's for the bias term
            train_x = torch.cat((torch.ones(train_x.shape[0], 1), train_x), dim=1)
            train_y = dataset[:, -config.model.y_dim:]
            # OLS beta hat
            xtx = train_x.T.mm(train_x)
            if torch.det(xtx).cpu().detach().numpy() == 0:
                xtx_inv = torch.linalg.pinv(xtx)
                logging.info("Invert the matrix with Moore-Penrose inverse...\n")
            else:
                xtx_inv = torch.inverse(xtx)
                logging.info("Invert the invertible square matrix...\n")
            # xtx_inv = torch.linalg.pinv(xtx)
            self.cond_pred_model = (xtx_inv.mm(train_x.T.mm(train_y))).to(self.device)
            # OLS_RMSE = (train_y - train_x.mm(self.cond_pred_model.cpu())).square().mean().sqrt()
            # logging.info("Training data RMSE with OLS is {:.8f}.\n".format(OLS_RMSE))
            del dataset
            del train_x
            del train_y
            gc.collect()
        elif config.diffusion.conditioning_signal == "NN":
            self.cond_pred_model = DeterministicFeedForwardNeuralNetwork(
                dim_in=config.model.x_dim, dim_out=config.model.y_dim,
                hid_layers=config.diffusion.nonlinear_guidance.hid_layers,
                use_batchnorm=config.diffusion.nonlinear_guidance.use_batchnorm,
                negative_slope=config.diffusion.nonlinear_guidance.negative_slope,
                dropout_rate=config.diffusion.nonlinear_guidance.dropout_rate).to(self.device)
            self.aux_cost_function = nn.MSELoss()
        else:
            pass

    # Compute guiding prediction as diffusion condition
    def compute_guiding_prediction(self, x, method="OLS"):
        """
        Compute y_0_hat, to be used as the Gaussian mean at time step T.
        """
        if method == "OLS":
            x = torch.cat((torch.ones(x.shape[0], 1).to(x.device), x), dim=1)
            y_pred = x.mm(self.cond_pred_model)
        # elif method == "ZERO":
        #     y_pred = torch.zeros(x.shape[0], 1).to(x.device)
        elif method == "NN":
            y_pred = self.cond_pred_model(x)
        else:
            y_pred = None
        return y_pred

    def evaluate_guidance_model(self, dataset_object, dataset_loader):
        """
        Evaluate guidance model by reporting train or test set unnormalized y RMSE.
        """
        y_se_list = []
        for xy_0 in dataset_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, :-self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim:]
            y_batch_pred_mean = self.compute_guiding_prediction(
                x_batch, method=self.config.diffusion.conditioning_signal).cpu().detach().numpy()
            y_batch = y_batch.cpu().detach().numpy()
            if dataset_object.normalize_y:
                y_batch = dataset_object.scaler_y.inverse_transform(y_batch).astype(np.float32)
                y_batch_pred_mean = dataset_object.scaler_y.inverse_transform(y_batch_pred_mean).astype(np.float32)
            y_se = (y_batch_pred_mean - y_batch) ** 2
            if len(y_se_list) == 0:
                y_se_list = y_se
            else:
                y_se_list = np.concatenate([y_se_list, y_se], axis=0)
        y_rmse = np.sqrt(np.mean(y_se_list))
        return y_rmse

    def evaluate_guidance_model_on_both_train_and_test_set(self,
                                                           train_set_object, train_loader,
                                                           test_set_object, test_loader):
        y_train_rmse_aux_model = self.evaluate_guidance_model(train_set_object, train_loader)
        y_test_rmse_aux_model = self.evaluate_guidance_model(test_set_object, test_loader)
        logging.info(("{} guidance model un-normalized y RMSE " +
                      "\n\tof the training set and of the test set are " +
                      "\n\t{:.8f} and {:.8f}, respectively.").format(
            self.config.diffusion.conditioning_signal, y_train_rmse_aux_model, y_test_rmse_aux_model))

    def nonlinear_guidance_model_train_step(self, x_batch, y_batch, aux_optimizer):
        """
        One optimization step of the non-linear guidance model that predicts y_0_hat.
        """
        y_batch_pred = self.cond_pred_model(x_batch)
        aux_cost = self.aux_cost_function(y_batch_pred, y_batch)
        # update non-linear guidance model
        aux_optimizer.zero_grad()
        aux_cost.backward()
        aux_optimizer.step()
        return aux_cost.cpu().item()

    def nonlinear_guidance_model_train_loop_per_epoch(self, train_batch_loader, aux_optimizer, epoch):
        for xy_0 in train_batch_loader:
            xy_0 = xy_0.to(self.device)
            x_batch = xy_0[:, :-self.config.model.y_dim]
            y_batch = xy_0[:, -self.config.model.y_dim:]
            aux_loss = self.nonlinear_guidance_model_train_step(x_batch, y_batch, aux_optimizer)
        if epoch % self.config.diffusion.nonlinear_guidance.logging_interval == 0:
            logging.info(f"epoch: {epoch}, non-linear guidance model pre-training loss: {aux_loss}")

    def obtain_true_and_pred_y_t(self, cur_t, y_seq, y_T_mean, y_0):
        y_t_p_sample = y_seq[self.num_timesteps - cur_t].detach().cpu()
        y_t_true = q_sample(y_0, y_T_mean,
                            self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                            torch.tensor([cur_t - 1])).detach().cpu()
        return y_t_p_sample, y_t_true

    def compute_unnorm_y(self, cur_y, testing):
        if testing:
            y_mean = cur_y.cpu().reshape(-1, self.config.testing.n_z_samples).mean(1).reshape(-1, 1)
        else:
            y_mean = cur_y.cpu()
        if self.config.data.normalize_y:
            y_t_unnorm = self.dataset_object.scaler_y.inverse_transform(y_mean)
        else:
            y_t_unnorm = y_mean
        return y_t_unnorm

    def make_subplot_at_timestep_t(self, cur_t, cur_y, y_i, y_0, axs, ax_idx, prior=False, testing=True):
        # kl = (y_i - cur_y).square().mean()
        # kl_y0 = (y_0.cpu() - cur_y).square().mean()
        y_0_unnorm = self.compute_unnorm_y(y_0, testing)
        y_t_unnorm = self.compute_unnorm_y(cur_y, testing)
        kl_unnorm = ((y_0_unnorm - y_t_unnorm) ** 2).mean() ** 0.5
        axs[ax_idx].plot(cur_y, '.', label='pred', c='tab:blue')
        axs[ax_idx].plot(y_i, '.', label='true', c='tab:red')
        # axs[ax_idx].set_xlabel(
        #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0),
        #     fontsize=20)
        kl_unnorm_str = 'Unnormed RMSE: {:.2f}'.format(kl_unnorm)
        if prior:
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p({y}_\mathbf{prior})$\n' + kl_unnorm_str,
                                  fontsize=23)
            axs[ax_idx].legend()
        else:
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$',
                                  fontsize=23)
            axs[ax_idx].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$\n' + kl_unnorm_str,
                                  fontsize=23)

    def train(self):
        args = self.args
        config = self.config
        tb_logger = self.config.tb_logger
        # first obtain test set for pre-trained model evaluation
        logging.info("Test set info:")
        test_set_object, test_set = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            test_set,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        # obtain training set
        logging.info("Training set info:")
        dataset_object, dataset = get_dataset(args, config, test_set=False)
        self.dataset_object = dataset_object
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        # obtain training (as a subset of original one) and validation set for guidance model hyperparameter tuning
        if hasattr(config.diffusion.nonlinear_guidance, "apply_early_stopping") \
                and config.diffusion.nonlinear_guidance.apply_early_stopping:
            logging.info(("\nSplit original training set into training and validation set " +
                          "for f_phi hyperparameter tuning..."))
            logging.info("Validation set info:")
            val_set_object, val_set = get_dataset(args, config, test_set=True, validation=True)
            val_loader = data.DataLoader(
                val_set,
                batch_size=config.testing.batch_size,
                num_workers=config.data.num_workers,
            )
            logging.info("Training subset info:")
            train_subset_object, train_subset = get_dataset(args, config, test_set=False, validation=True)
            train_subset_loader = data.DataLoader(
                train_subset,
                batch_size=config.training.batch_size,
                shuffle=True,
                num_workers=config.data.num_workers,
            )

        model = ConditionalGuidedModel(config)
        model = model.to(self.device)
        # evaluate f_phi(x) on both training and test set
        if config.diffusion.conditioning_signal == "NN":
            logging.info("\nBefore pre-training:")
        self.evaluate_guidance_model_on_both_train_and_test_set(dataset_object, train_loader,
                                                                test_set_object, test_loader)

        optimizer = get_optimizer(self.config.optim, model.parameters())
        # apply an auxiliary optimizer for the NN guidance model that predicts y_0_hat
        if config.diffusion.conditioning_signal == "NN":
            aux_optimizer = get_optimizer(self.config.aux_optim, self.cond_pred_model.parameters())

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        # pre-train the non-linear guidance model
        if config.diffusion.conditioning_signal == "NN" and config.diffusion.nonlinear_guidance.pre_train:
            n_guidance_model_pretrain_epochs = config.diffusion.nonlinear_guidance.n_pretrain_epochs
            self.cond_pred_model.train()
            if hasattr(config.diffusion.nonlinear_guidance, "apply_early_stopping") \
                    and config.diffusion.nonlinear_guidance.apply_early_stopping:
                early_stopper = EarlyStopping(patience=config.diffusion.nonlinear_guidance.patience,
                                              delta=config.diffusion.nonlinear_guidance.delta)
                train_val_start_time = time.time()
                for epoch in range(config.diffusion.nonlinear_guidance.n_pretrain_max_epochs):
                    self.nonlinear_guidance_model_train_loop_per_epoch(train_subset_loader, aux_optimizer, epoch)
                    y_val_rmse_aux_model = self.evaluate_guidance_model(val_set_object, val_loader)
                    val_cost = y_val_rmse_aux_model
                    early_stopper(val_cost=val_cost, epoch=epoch)
                    if early_stopper.early_stop:
                        print(("Obtained best performance on validation set after Epoch {}; " +
                               "early stopping at Epoch {}.").format(
                            early_stopper.best_epoch, epoch))
                        break
                train_val_end_time = time.time()
                logging.info(("Tuning for number of epochs to train non-linear guidance model " +
                              "took {:.4f} minutes.").format(
                    (train_val_end_time - train_val_start_time) / 60))
                logging.info("\nAfter tuning for best total epochs, on training sebset and validation set:")
                self.evaluate_guidance_model_on_both_train_and_test_set(train_subset_object, train_subset_loader,
                                                                        val_set_object, val_loader)
                # reset guidance model weights for re-training on original training set
                logging.info("\nReset guidance model weights...")
                for layer in self.cond_pred_model.network.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
                logging.info("\nRe-training the guidance model on original training set with {} epochs...".format(
                    early_stopper.best_epoch
                ))
                n_guidance_model_pretrain_epochs = early_stopper.best_epoch
                aux_optimizer = get_optimizer(self.config.aux_optim, self.cond_pred_model.parameters())
            pretrain_start_time = time.time()
            for epoch in range(n_guidance_model_pretrain_epochs):
                self.nonlinear_guidance_model_train_loop_per_epoch(train_loader, aux_optimizer, epoch)
            pretrain_end_time = time.time()
            logging.info("Pre-training of non-linear guidance model took {:.4f} minutes.".format(
                (pretrain_end_time - pretrain_start_time) / 60))
            logging.info("\nAfter pre-training:")
            self.evaluate_guidance_model_on_both_train_and_test_set(dataset_object, train_loader,
                                                                    test_set_object, test_loader)
            # save auxiliary model
            aux_states = [
                self.cond_pred_model.state_dict(),
                aux_optimizer.state_dict(),
            ]
            torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

        # train diffusion model
        if not self.args.train_guidance_only:
            start_epoch, step = 0, 0
            if self.args.resume_training:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                    map_location=self.device)
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])
                # load auxiliary model
                if config.diffusion.conditioning_signal == "NN":
                    aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                            map_location=self.device)
                    self.cond_pred_model.load_state_dict(aux_states[0])
                    aux_optimizer.load_state_dict(aux_states[1])

            if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                logging.info("Prior distribution at timestep T has a mean of 0.")
            for epoch in range(start_epoch, self.config.training.n_epochs):
                data_start = time.time()
                data_time = 0
                for i, xy_0 in enumerate(train_loader):
                    n = xy_0.size(0)
                    data_time += time.time() - data_start
                    model.train()
                    step += 1

                    # antithetic sampling -- low (inclusive) and high (exclusive)
                    t = torch.randint(
                        low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                    ).to(self.device)
                    t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]

                    # noise estimation loss
                    xy_0 = xy_0.to(self.device)
                    x_batch = xy_0[:, :-config.model.y_dim]
                    y_batch = xy_0[:, -config.model.y_dim:]  # shape: (batch_size, 1)
                    y_0_hat_batch = self.compute_guiding_prediction(x_batch,
                                                                    method=config.diffusion.conditioning_signal)
                    y_T_mean = y_0_hat_batch
                    if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                        y_T_mean = torch.zeros(y_batch.shape[0], 1).to(y_batch.device)
                    e = torch.randn_like(y_batch).to(y_batch.device)
                    y_t_batch = q_sample(y_batch, y_T_mean,
                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt, t, noise=e)
                    # output = model(x_batch, y_t_batch, y_T_mean, t)
                    output = model(x_batch, y_t_batch, y_0_hat_batch, t)

                    loss = (e - output).square().mean()  # use the same noise sample e during training to compute loss

                    tb_logger.add_scalar("loss", loss, global_step=step)

                    if step % self.config.training.logging_freq == 0 or step == 1:
                        logging.info(
                            (f"epoch: {epoch}, step: {step}, Noise Estimation loss: {loss.item()}, " +
                             f"data time: {data_time / (i + 1)}")
                        )

                    # optimize diffusion model that predicts eps_theta
                    optimizer.zero_grad()
                    loss.backward()
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                    if self.config.model.ema:
                        ema_helper.update(model)

                    # optimize non-linear guidance model
                    if config.diffusion.conditioning_signal == "NN" and config.diffusion.nonlinear_guidance.joint_train:
                        self.cond_pred_model.train()
                        aux_loss = self.nonlinear_guidance_model_train_step(x_batch, y_batch, aux_optimizer)
                        if step % self.config.training.logging_freq == 0 or step == 1:
                            logging.info(
                                f"meanwhile, non-linear guidance model joint-training loss: {aux_loss}"
                            )

                    # save diffusion model
                    if step % self.config.training.snapshot_freq == 0 or step == 1:
                        states = [
                            model.state_dict(),
                            optimizer.state_dict(),
                            epoch,
                            step,
                        ]
                        if self.config.model.ema:
                            states.append(ema_helper.state_dict())

                        if step > 1:  # skip saving the initial ckpt
                            torch.save(
                                states,
                                os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                            )
                        torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                        # save auxiliary model
                        if hasattr(config.diffusion.nonlinear_guidance, "joint_train"):
                            if config.diffusion.nonlinear_guidance.joint_train:
                                assert config.diffusion.conditioning_signal == "NN"
                                aux_states = [
                                    self.cond_pred_model.state_dict(),
                                    aux_optimizer.state_dict(),
                                ]
                                if step > 1:  # skip saving the initial ckpt
                                    torch.save(
                                        aux_states,
                                        os.path.join(self.args.log_path, "aux_ckpt_{}.pth".format(step)),
                                    )
                                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))

                    if step % self.config.training.validation_freq == 0 or step == 1:
                        if config.data.dataset == "uci":  # plot UCI prediction and ground truth
                            with torch.no_grad():
                                y_p_seq = p_sample_loop(model, x_batch, y_batch, y_T_mean, self.num_timesteps,
                                                        self.alphas, self.one_minus_alphas_bar_sqrt)
                                fig, axs = plt.subplots(1, (self.num_figs + 1),
                                                        figsize=((self.num_figs + 1) * 8.5, 8.5), clear=True)
                                # plot at timestep 1
                                cur_t = 1
                                cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean, y_batch)
                                self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_batch, axs, 0,
                                                                testing=False)
                                # plot at vis_step interval
                                for j in range(1, self.num_figs):
                                    cur_t = j * self.vis_step
                                    cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean, y_batch)
                                    self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_batch, axs, j,
                                                                    testing=False)
                                # plot at timestep T
                                cur_t = self.num_timesteps
                                cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean, y_batch)
                                self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_batch, axs, self.num_figs,
                                                                prior=True, testing=False)
                                ax_list = [axs[j] for j in range(self.num_figs + 1)]
                                ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
                                ax_list[0].get_shared_y_axes().join(ax_list[0], *ax_list)
                                tb_logger.add_figure('samples', fig, step)
                                fig.savefig(
                                    os.path.join(args.im_path, 'samples_T{}_{}.png'.format(self.num_timesteps, step)))
                            plt.close('all')
                        else:  # visualization for toy data (where x is 1-D) during training
                            with torch.no_grad():
                                # plot q samples
                                if epoch == start_epoch:
                                    fig, axs = plt.subplots(1, self.num_figs + 1,
                                                            figsize=((self.num_figs + 1) * 8.5, 8.5), clear=True)
                                    # q samples at timestep 1
                                    y_1 = q_sample(y_batch, y_T_mean,
                                                   self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                                                   torch.tensor([0]).to(self.device)
                                                   ).detach().cpu()
                                    axs[0].scatter(x_batch.detach().cpu(), y_1, s=10, c='tab:red')
                                    axs[0].set_title(
                                        '$q(\mathbf{y}_{' + str(1) + '})$',
                                        fontsize=23)
                                    y_q_seq = []
                                    for j in range(1, self.num_figs):
                                        cur_t = j * self.vis_step
                                        cur_y = q_sample(y_batch, y_T_mean,
                                                         self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                                                         torch.tensor([cur_t - 1]).to(self.device)
                                                         ).detach().cpu()
                                        y_q_seq.append(cur_y)
                                        axs[j].scatter(x_batch.detach().cpu(), cur_y, s=10, c='tab:red')

                                        axs[j].set_title('$q(\mathbf{y}_{' + str(cur_t) + '})$',
                                                         fontsize=23)
                                    # q samples at timestep T
                                    y_T = q_sample(y_batch, y_T_mean,
                                                   self.alphas_bar_sqrt, self.one_minus_alphas_bar_sqrt,
                                                   torch.tensor([self.num_timesteps - 1]).to(self.device)
                                                   ).detach().cpu()
                                    axs[self.num_figs].scatter(x_batch.detach().cpu(), y_T, s=10, c='tab:red')
                                    axs[self.num_figs].set_title(
                                        '$q(\mathbf{y}_{' + str(self.num_timesteps) + '})$', fontsize=23)
                                    ax_list = [axs[j] for j in range(self.num_figs + 1)]
                                    ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
                                    ax_list[0].get_shared_y_axes().join(ax_list[0], *ax_list)
                                    if config.testing.squared_plot:
                                        for j in range(self.num_figs + 1):
                                            axs[j].set(aspect='equal', adjustable='box')
                                    tb_logger.add_figure('data', fig, step)
                                    fig.savefig(
                                        os.path.join(args.im_path,
                                                     'q_samples_T{}_{}.png'.format(self.num_timesteps, step)))

                                # plot p samples
                                fig, axs = plt.subplots(1, self.num_figs + 1,
                                                        figsize=((self.num_figs + 1) * 8.5, 8.5), clear=True)
                                y_p_seq = p_sample_loop(model, x_batch, y_0_hat_batch, y_T_mean, self.num_timesteps,
                                                        self.alphas, self.one_minus_alphas_bar_sqrt)
                                # p samples at timestep 1
                                cur_y = y_p_seq[self.num_timesteps - 1].detach().cpu()
                                axs[0].scatter(x_batch.detach().cpu(), cur_y, s=10, c='tab:blue')
                                axs[0].set_title('$p({z}_1)$', fontsize=23)
                                # kl = kld(y_1, cur_y)
                                # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                                axs[0].set_title('$p(\mathbf{y}_{1})$', fontsize=23)
                                # axs[0].set_xlabel(
                                #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(
                                #         kl, kl_y0), fontsize=20)
                                for j in range(1, self.num_figs):
                                    cur_t = j * self.vis_step
                                    cur_y = y_p_seq[self.num_timesteps - cur_t].detach().cpu()
                                    # kl = kld(y_q_seq[j-1].detach().cpu(), cur_y)
                                    # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                                    axs[j].scatter(x_batch.detach().cpu(), cur_y, s=10, c='tab:blue')
                                    axs[j].set_title('$p(\mathbf{y}_{' + str(cur_t) + '})$', fontsize=23)
                                    # axs[j].set_xlabel(
                                    #     'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(
                                    #         kl, kl_y0), fontsize=20)
                                # p samples at timestep T
                                cur_y = y_p_seq[0].detach().cpu()
                                axs[self.num_figs].scatter(x_batch.detach().cpu(), cur_y, s=10, c='tab:blue')
                                axs[self.num_figs].set_title('$p({z}_\mathbf{prior})$', fontsize=23)
                                # kl = kld(y_T, cur_y)
                                # kl_y0 = kld(y_batch.detach().cpu(), cur_y)
                                # axs[self.num_figs].set_xlabel(
                                #     'KL($q(y_t)||p(z)$)={:.2f}\nKL($q(y_0)||p(z)$)={:.2f}'.format(
                                #         kl, kl_y0), fontsize=20)
                                if step > 1:
                                    ax_list = [axs[j] for j in range(self.num_figs + 1)]
                                    ax_list[0].get_shared_x_axes().join(ax_list[0], *ax_list)
                                    ax_list[0].get_shared_y_axes().join(ax_list[0], *ax_list)
                                    # define custom 'xlim' and 'ylim' values
                                    # custom_xlim = axs[0].get_xlim()
                                    # custom_ylim = axs[0].get_ylim()
                                    # plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)
                                    if config.testing.squared_plot:
                                        for j in range(self.num_figs + 1):
                                            axs[j].set(aspect='equal', adjustable='box')
                                tb_logger.add_figure('samples', fig, step)
                                fig.savefig(
                                    os.path.join(args.im_path, 'p_samples_T{}_{}.png'.format(self.num_timesteps, step)))
                            plt.close('all')

                    data_start = time.time()
            plt.close('all')
            # save the model after training is finished
            states = [
                model.state_dict(),
                optimizer.state_dict(),
                epoch,
                step,
            ]
            if self.config.model.ema:
                states.append(ema_helper.state_dict())
            torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            # save auxiliary model after training is finished
            if config.diffusion.conditioning_signal == "NN":
                aux_states = [
                    self.cond_pred_model.state_dict(),
                    aux_optimizer.state_dict(),
                ]
                torch.save(aux_states, os.path.join(self.args.log_path, "aux_ckpt.pth"))
                # report training set RMSE if applied joint training
                if config.diffusion.nonlinear_guidance.joint_train:
                    y_rmse_aux_model = self.evaluate_guidance_model(dataset_object, train_loader)
                    logging.info(
                        "After joint-training, non-linear guidance model unnormalized y RMSE is {:.8f}.".format(
                            y_rmse_aux_model))

    # Currently only used for initial toy data
    def sample(self):
        model = ConditionalGuidedModel(self.config)

        if getattr(self.config.sampling, "ckpt_id", None) is None:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"),
                                map_location=self.device)
            ckpt_id = 'last'
        else:
            states = torch.load(os.path.join(self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.sampling.ckpt_id
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()

        # load auxiliary model
        if config.diffusion.conditioning_signal == "NN":
            aux_states = torch.load(os.path.join(self.args.log_path, "aux_ckpt.pth"),
                                    map_location=self.device)
            self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()

        if self.config.data.dataset == 'swiss_roll':
            xy_0 = SwissRoll().sample(self.config.sampling.sampling_size)
        elif self.config.data.dataset == 'moons':
            xy_0 = Moons().sample(self.config.sampling.sampling_size)
        elif self.config.data.dataset == '8gaussians':
            xy_0 = Gaussians().sample(self.config.sampling.sampling_size - self.config.sampling.sampling_size % 8,
                                      mode=8)
        elif self.config.data.dataset == '25gaussians':
            xy_0 = Gaussians().sample(self.config.sampling.sampling_size - self.config.sampling.sampling_size % 25,
                                      mode=25)

        x_batch = xy_0[:, :-self.config.model.y_dim]
        y_batch = xy_0[:, -self.config.model.y_dim:]
        y_0_hat_batch = self.compute_guiding_prediction(x_batch.to(self.device),
                                                        method=self.config.diffusion.conditioning_signal)
        y_T_mean = y_0_hat_batch
        if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
            y_T_mean = torch.zeros(y_batch.shape[0], 1).to(self.device)

        y_q_seq = []
        with torch.no_grad():
            fig, axs = plt.subplots(1, self.num_figs,
                                    figsize=(self.num_figs * 8.5, 8), clear=True)
            for i in range(self.num_figs - 1):
                cur_y = q_sample(y_batch, y_T_mean,
                                 self.alphas_bar_sqrt.cpu(), self.one_minus_alphas_bar_sqrt.cpu(),
                                 torch.tensor([i * self.vis_step])).detach().cpu()
                y_q_seq.append(cur_y)
                axs[i].scatter(x_batch, cur_y, s=10)
                axs[i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=23)
                axs[i].tick_params(axis='x', labelsize=20)
                axs[i].tick_params(axis='y', labelsize=20)
            cur_y = q_sample(y_batch, y_T_mean.cpu(),
                             self.alphas_bar_sqrt.cpu(), self.one_minus_alphas_bar_sqrt.cpu(),
                             torch.tensor([self.num_timesteps - 1])).detach().cpu()
            y_q_seq.append(cur_y)
            axs[self.num_figs - 1].scatter(x_batch, cur_y, s=10)
            axs[self.num_figs - 1].set_title('$q(\mathbf{y}_{' + str(self.num_timesteps - 1) + '})$', fontsize=23)
            fig.savefig(
                os.path.join(self.args.im_path, 'diffusion_samples_T{}_{}.png'.format(self.num_timesteps, ckpt_id)))

            y_p_seq = p_sample_loop(model, x_batch.to(self.device), y_0_hat_batch, y_T_mean,
                                    self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt)
            fig, axs = plt.subplots(2, self.num_figs,
                                    figsize=(self.num_figs * 8.5, 8 * 2.2), clear=True)
            for i in range(self.num_figs - 1):
                cur_y = y_p_seq[self.num_timesteps - i * self.vis_step - 1].detach().cpu()
                kl = kld(y_q_seq[i].detach().cpu(), cur_y)
                kl_y0 = kld(y_q_seq[0].detach().cpu(), cur_y)
                axs[0, i].scatter(x_batch, y_q_seq[i], s=10)
                axs[0, i].tick_params(axis='x', labelsize=20)
                axs[0, i].tick_params(axis='y', labelsize=20)
                axs[0, i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=23)
                axs[1, i].scatter(x_batch, cur_y, s=10)
                axs[1, i].set_title('$p_{\\theta}(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=23)
                axs[1, i].set_xlabel(
                    'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0), fontsize=20)
                axs[1, i].tick_params(axis='x', labelsize=20)
                axs[1, i].tick_params(axis='y', labelsize=20)
            cur_y = y_p_seq[0].detach().cpu()
            kl = kld(y_q_seq[-1].detach().cpu(), cur_y)
            kl_y0 = kld(y_q_seq[0].detach().cpu(), cur_y)
            axs[0, self.num_figs - 1].scatter(x_batch, y_q_seq[-1], s=10)
            axs[0, self.num_figs - 1].tick_params(axis='x', labelsize=20)
            axs[0, self.num_figs - 1].tick_params(axis='y', labelsize=20)
            axs[0, self.num_figs - 1].set_title('$q(\mathbf{y}_{' + str(self.num_timesteps) + '})$', fontsize=23)
            axs[1, self.num_figs - 1].scatter(x_batch, cur_y, s=10)
            axs[1, self.num_figs - 1].set_title('$p_{\\theta}(\mathbf{y}_{' + str(self.num_timesteps) + '})$',
                                                fontsize=23)
            axs[1, self.num_figs - 1].set_xlabel(
                'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0), fontsize=20)
            axs[1, self.num_figs - 1].tick_params(axis='x', labelsize=20)
            axs[1, self.num_figs - 1].tick_params(axis='y', labelsize=20)
            fig.savefig(
                os.path.join(self.args.im_path, 'generated_samples_T{}_{}.png'.format(self.num_timesteps, ckpt_id)))

            fig, axs = plt.subplots(2, self.num_figs,
                                    figsize=(self.num_figs * 8.5, 8 * 2.2), clear=True)
            for i in range(self.num_figs - 1):
                cur_y = y_p_seq[self.num_timesteps - i * self.vis_step - 1].detach().cpu()
                kl = kld(y_q_seq[i].detach().cpu(), cur_y)
                kl_y0 = kld(y_q_seq[0].detach().cpu(), cur_y)
                heatmap_x, _, _ = np.histogram2d(x_batch.numpy(), y_q_seq[i].numpy(), bins=100)
                axs[0, i].imshow(heatmap_x.T)
                axs[0, i].set_title('$q(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=23)
                axs[0, i].axis('off')
                axs[0, i].tick_params(axis='x', labelsize=20)
                axs[0, i].tick_params(axis='y', labelsize=20)
                heatmap, _, _ = np.histogram2d(x_batch.numpy(), cur_y.numpy(), bins=100)
                axs[1, i].imshow(heatmap.T)
                axs[1, i].set_title('$p_{\\theta}(\mathbf{y}_{' + str(i * self.vis_step) + '})$', fontsize=23)
                axs[1, i].axis('off')
                axs[1, i].set_xlabel(
                    'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0), fontsize=20)
                axs[1, i].tick_params(axis='x', labelsize=20)
                axs[1, i].tick_params(axis='y', labelsize=20)
            cur_y = y_p_seq[0].detach().cpu()
            kl = kld(y_q_seq[-1].detach().cpu(), cur_y)
            kl_y0 = kld(y_q_seq[0].detach().cpu(), cur_y)
            heatmap_x, _, _ = np.histogram2d(x_batch.numpy(), y_q_seq[-1].numpy(), bins=100)
            axs[0, self.num_figs - 1].imshow(heatmap_x.T)
            axs[0, self.num_figs - 1].set_title('$q(\mathbf{y}_{' + str(self.num_timesteps) + '})$', fontsize=23)
            axs[0, self.num_figs - 1].axis('off')
            axs[0, self.num_figs - 1].tick_params(axis='x', labelsize=20)
            axs[0, self.num_figs - 1].tick_params(axis='y', labelsize=20)
            heatmap, _, _ = np.histogram2d(x_batch.numpy(), cur_y.numpy(), bins=100)
            axs[1, self.num_figs - 1].imshow(heatmap.T)
            axs[1, self.num_figs - 1].set_title('$p_{\\theta}(\mathbf{y}_{' + str(self.num_timesteps) + '})$',
                                                fontsize=23)
            axs[1, self.num_figs - 1].axis('off')
            axs[1, self.num_figs - 1].set_xlabel(
                'KL($q(y_t)||p(y_t)$)={:.2f}\nKL($q(y_0)||p(y_t)$)={:.2f}'.format(kl, kl_y0), fontsize=20)
            axs[1, self.num_figs - 1].tick_params(axis='x', labelsize=20)
            axs[1, self.num_figs - 1].tick_params(axis='y', labelsize=20)
            fig.savefig(os.path.join(self.args.im_path,
                                     'generated_distribution_T{}_{}.png'.format(self.num_timesteps, ckpt_id)))
        plt.close('all')

    def test(self):
        """
        Evaluate model on regression tasks on test set.
        """

        #####################################################################################################
        ########################## local functions within the class function scope ##########################
        def compute_prediction_SE(config, dataset_object, y_batch, generated_y, return_pred_mean=False):
            """
            generated_y: has a shape of (current_batch_size, n_z_samples, dim_y)
            """
            low, high = config.testing.trimmed_mean_range
            y_true = y_batch.cpu().detach().numpy()
            y_pred_mean = None  # to be used to compute RMSE
            if low == 50 and high == 50:
                y_pred_mean = np.median(generated_y, axis=1)  # use median of samples as the mean prediction
            else:  # compute trimmed mean (i.e. discarding certain parts of the samples at both ends)
                generated_y.sort(axis=1)
                low_idx = int(low / 100 * config.testing.n_z_samples)
                high_idx = int(high / 100 * config.testing.n_z_samples)
                y_pred_mean = (generated_y[:, low_idx:high_idx]).mean(axis=1)
            if dataset_object.normalize_y:
                y_true = dataset_object.scaler_y.inverse_transform(y_true).astype(np.float32)
                y_pred_mean = dataset_object.scaler_y.inverse_transform(y_pred_mean).astype(np.float32)
            if return_pred_mean:
                return y_pred_mean
            else:
                y_se = (y_pred_mean - y_true) ** 2
                return y_se

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y, verbose=True):
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])
            if verbose:
                y_true_below_0, y_true_above_100 = y_true_quantile_bin_count[0], \
                                                   y_true_quantile_bin_count[-1]
                logging.info(("We have {} true y smaller than min of generated y, " + \
                              "and {} greater than max of generated y.").format(y_true_below_0, y_true_above_100))
            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])
            coverage = y_in_range.mean()
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        def store_gen_y_at_step_t(config, current_batch_size, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = self.num_timesteps - idx
            gen_y = y_tile_seq[idx].reshape(current_batch_size,
                                            config.testing.n_z_samples,
                                            config.model.y_dim).cpu().numpy()
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def store_y_se_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            current_t = self.num_timesteps - idx
            # compute sqaured error in each batch
            y_se = compute_prediction_SE(config=config, dataset_object=dataset_object,
                                         y_batch=y_batch, generated_y=gen_y)
            if len(y_se_by_batch_list[current_t]) == 0:
                y_se_by_batch_list[current_t] = y_se
            else:
                y_se_by_batch_list[current_t] = np.concatenate([y_se_by_batch_list[current_t], y_se], axis=0)

        def set_NLL_global_precision(test_var=True):
            if test_var:
                # compute test set sample variance
                if dataset_object.normalize_y:
                    y_test_unnorm = dataset_object.scaler_y.inverse_transform(dataset_object.y_test).astype(np.float32)
                else:
                    y_test_unnorm = dataset_object.y_test
                y_test_unnorm = y_test_unnorm if type(y_test_unnorm) is torch.Tensor \
                    else torch.from_numpy(y_test_unnorm)
                self.tau = 1 / (y_test_unnorm.var(unbiased=True).item())
            else:
                self.tau = 1

        def compute_batch_NLL(config, dataset_object, y_batch, generated_y):
            """
            generated_y: has a shape of (current_batch_size, n_z_samples, dim_y)

            NLL computation implementation from MC dropout repo
                https://github.com/yaringal/DropoutUncertaintyExps/blob/master/net/net.py,
                directly from MC Dropout paper Eq. (8).
            """
            y_true = y_batch.cpu().detach().numpy()
            if dataset_object.normalize_y:
                # unnormalize true y
                y_true = dataset_object.scaler_y.inverse_transform(y_true).astype(np.float32)
                # unnormalize generated y
                batch_size = generated_y.shape[0]
                generated_y = generated_y.reshape(batch_size * config.testing.n_z_samples, config.model.y_dim)
                generated_y = dataset_object.scaler_y.inverse_transform(generated_y).astype(np.float32).reshape(
                    batch_size, config.testing.n_z_samples, config.model.y_dim)
            generated_y = generated_y.swapaxes(0, 1)
            # obtain precision value and compute test batch NLL
            if self.tau is not None:
                tau = self.tau
            else:
                gen_y_var = torch.from_numpy(generated_y).var(dim=0, unbiased=True).numpy()
                tau = 1 / gen_y_var
            nll = -(logsumexp(-0.5 * tau * (y_true[None] - generated_y) ** 2., 0)
                    - np.log(config.testing.n_z_samples)
                    - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
            return nll

        def store_nll_at_step_t(config, idx, dataset_object, y_batch, gen_y):
            current_t = self.num_timesteps - idx
            # compute negative log-likelihood in each batch
            nll = compute_batch_NLL(config=config, dataset_object=dataset_object,
                                    y_batch=y_batch, generated_y=gen_y)
            if len(nll_by_batch_list[current_t]) == 0:
                nll_by_batch_list[current_t] = nll
            else:
                nll_by_batch_list[current_t] = np.concatenate([nll_by_batch_list[current_t], nll], axis=0)

        #####################################################################################################
        #####################################################################################################

        args = self.args
        config = self.config
        split = args.split
        log_path = os.path.join(self.args.log_path)
        dataset_object, dataset = get_dataset(args, config, test_set=True)
        test_loader = data.DataLoader(
            dataset,
            batch_size=config.testing.batch_size,
            num_workers=config.data.num_workers,
        )
        self.dataset_object = dataset_object
        # set global prevision value for NLL computation if needed
        if args.nll_global_var:
            set_NLL_global_precision(test_var=args.nll_test_var)

        model = ConditionalGuidedModel(self.config)
        if getattr(self.config.testing, "ckpt_id", None) is None:
            states = torch.load(os.path.join(log_path, "ckpt.pth"),
                                map_location=self.device)
            ckpt_id = 'last'
        else:
            states = torch.load(os.path.join(log_path, f"ckpt_{self.config.testing.ckpt_id}.pth"),
                                map_location=self.device)
            ckpt_id = self.config.testing.ckpt_id
        logging.info(f"Loading from: {log_path}/ckpt_{ckpt_id}.pth")
        model = model.to(self.device)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMA(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        else:
            ema_helper = None

        model.eval()

        # load auxiliary model
        if config.diffusion.conditioning_signal == "NN":
            aux_states = torch.load(os.path.join(log_path, "aux_ckpt.pth"),
                                    map_location=self.device)
            self.cond_pred_model.load_state_dict(aux_states[0], strict=True)
            self.cond_pred_model.eval()
        # report test set RMSE with guidance model
        y_rmse_aux_model = self.evaluate_guidance_model(dataset_object, test_loader)
        logging.info("Test set unnormalized y RMSE on trained {} guidance model is {:.8f}.".format(
            config.diffusion.conditioning_signal, y_rmse_aux_model))

        # sanity check
        logging.info("Sanity check of the checkpoint")
        if config.data.dataset == "uci":
            dataset_check = dataset_object.return_dataset(split="train")
        else:
            dataset_check = dataset_object.train_dataset
        dataset_check = dataset_check[:50]  # use the first 50 samples for sanity check
        x_check, y_check = dataset_check[:, :-config.model.y_dim], dataset_check[:, -config.model.y_dim:]
        y_check = y_check.to(self.device)
        y_0_hat_check = self.compute_guiding_prediction(x_check.to(self.device),
                                                        method=config.diffusion.conditioning_signal)

        y_T_mean_check = y_0_hat_check
        if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
            y_T_mean_check = torch.zeros(y_check.shape[0], 1).to(self.device)
        with torch.no_grad():
            y_p_seq = p_sample_loop(model, x_check.to(self.device), y_0_hat_check, y_T_mean_check,
                                    self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt)
            fig, axs = plt.subplots(1, (self.num_figs + 1),
                                    figsize=((self.num_figs + 1) * 8.5, 8.5), clear=True)
            # plot at timestep 1
            cur_t = 1
            cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean_check, y_check)
            self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_check, axs, 0,
                                            testing=False)
            # plot at vis_step interval
            for i in range(1, self.num_figs):
                cur_t = i * self.vis_step
                cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean_check, y_check)
                self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_check, axs, i,
                                                testing=False)
            # plot at timestep T
            cur_t = self.num_timesteps
            cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_p_seq, y_T_mean_check, y_check)
            self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_check, axs, self.num_figs,
                                            prior=True, testing=False)
            fig.savefig(os.path.join(args.im_path, 'sanity_check.pdf'))
            plt.close('all')

        if config.testing.compute_metric_all_steps:
            logging.info("\nWe compute RMSE, QICE, PICP and NLL for all steps.\n")
        else:
            mean_idx = self.num_timesteps - config.testing.mean_t
            coverage_idx = self.num_timesteps - config.testing.coverage_t
            nll_idx = self.num_timesteps - config.testing.nll_t
            logging.info(("\nWe pick t={} to compute y mean metric RMSE, " +
                          "and t={} to compute true y coverage metric QICE and PICP.\n").format(
                config.testing.mean_t, config.testing.coverage_t))

        with torch.no_grad():
            true_x_by_batch_list = []
            true_x_tile_by_batch_list = []
            true_y_by_batch_list = []
            gen_y_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            y_se_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]
            nll_by_batch_list = [[] for _ in range(self.num_timesteps + 1)]

            for step, xy_batch in enumerate(test_loader):
                # minibatch_start = time.time()
                xy_0 = xy_batch.to(self.device)
                current_batch_size = xy_0.shape[0]
                x_batch = xy_0[:, :-config.model.y_dim]
                y_batch = xy_0[:, -config.model.y_dim:]
                # compute y_0_hat as the initial prediction to guide the reverse diffusion process
                y_0_hat_batch = self.compute_guiding_prediction(x_batch,
                                                                method=config.diffusion.conditioning_signal)
                true_y_by_batch_list.append(y_batch.cpu().numpy())
                if config.testing.make_plot and config.data.dataset != "uci":
                    true_x_by_batch_list.append(x_batch.cpu().numpy())
                # obtain y samples through reverse diffusion -- some pytorch version might not have torch.tile
                y_0_tile = (y_batch.repeat(config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                    self.device).flatten(0, 1)
                y_0_hat_tile = (y_0_hat_batch.repeat(config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                    self.device).flatten(0, 1)
                y_T_mean_tile = y_0_hat_tile
                if config.diffusion.noise_prior:  # apply 0 instead of f_phi(x) as prior mean
                    y_T_mean_tile = torch.zeros(y_0_hat_tile.shape).to(self.device)
                x_tile = (x_batch.repeat(config.testing.n_z_samples, 1, 1).transpose(0, 1)).to(
                    self.device).flatten(0, 1)
                n_samples_gen_y_for_plot = 2
                if config.testing.plot_gen:
                    x_repeated = (x_batch.repeat(n_samples_gen_y_for_plot, 1, 1).transpose(0, 1)).to(
                        self.device).flatten(0, 1)
                    true_x_tile_by_batch_list.append(x_repeated.cpu().numpy())
                # generate samples from all time steps for the current mini-batch
                minibatch_sample_start = time.time()
                y_tile_seq = p_sample_loop(model, x_tile, y_0_hat_tile, y_T_mean_tile, self.num_timesteps,
                                           self.alphas, self.one_minus_alphas_bar_sqrt)
                minibatch_sample_end = time.time()
                logging.info("Minibatch {} sampling took {:.4f} seconds.".format(
                    step, (minibatch_sample_end - minibatch_sample_start)))
                # obtain generated y and compute squared error at all time steps or a particular time step
                if config.testing.compute_metric_all_steps:
                    for idx in range(self.num_timesteps + 1):
                        gen_y = store_gen_y_at_step_t(config=config, current_batch_size=current_batch_size,
                                                      idx=idx, y_tile_seq=y_tile_seq)
                        store_y_se_at_step_t(config=config, idx=idx,
                                             dataset_object=dataset_object,
                                             y_batch=y_batch, gen_y=gen_y)
                        store_nll_at_step_t(config=config, idx=idx,
                                            dataset_object=dataset_object,
                                            y_batch=y_batch, gen_y=gen_y)
                else:
                    # store generated y at certain step for RMSE and for QICE computation
                    gen_y = store_gen_y_at_step_t(config=config, current_batch_size=current_batch_size,
                                                  idx=mean_idx, y_tile_seq=y_tile_seq)
                    store_y_se_at_step_t(config=config, idx=mean_idx,
                                         dataset_object=dataset_object,
                                         y_batch=y_batch, gen_y=gen_y)
                    if coverage_idx != mean_idx:
                        _ = store_gen_y_at_step_t(config=config, current_batch_size=current_batch_size,
                                                  idx=coverage_idx, y_tile_seq=y_tile_seq)
                    if nll_idx != mean_idx and nll_idx != coverage_idx:
                        _ = store_gen_y_at_step_t(config=config, current_batch_size=current_batch_size,
                                                  idx=nll_idx, y_tile_seq=y_tile_seq)
                    store_nll_at_step_t(config=config, idx=nll_idx,
                                        dataset_object=dataset_object,
                                        y_batch=y_batch, gen_y=gen_y)

                # make plot at particular mini-batches
                if step % config.testing.plot_freq == 0:  # plot for every plot_freq-th mini-batch
                    fig, axs = plt.subplots(1, self.num_figs + 1,
                                            figsize=((self.num_figs + 1) * 8.5, 8.5), clear=True)
                    # plot at timestep 1
                    cur_t = 1
                    cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                    self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_0_tile, axs, 0)
                    # plot at vis_step interval
                    for i in range(1, self.num_figs):
                        cur_t = i * self.vis_step
                        cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                        self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_0_tile, axs, i)
                    # plot at timestep T
                    cur_t = self.num_timesteps
                    cur_y, y_i = self.obtain_true_and_pred_y_t(cur_t, y_tile_seq, y_T_mean_tile, y_0_tile)
                    self.make_subplot_at_timestep_t(cur_t, cur_y, y_i, y_0_tile, axs, self.num_figs, prior=True)
                    fig.savefig(os.path.join(args.im_path, 'samples_T{}_{}.png'.format(self.num_timesteps, step)))
                    plt.close('all')

        ################## compute metrics on test set ##################
        all_true_y = np.concatenate(true_y_by_batch_list, axis=0)
        if config.testing.make_plot and config.data.dataset != "uci":
            all_true_x = np.concatenate(true_x_by_batch_list, axis=0)
        if config.testing.plot_gen:
            all_true_x_tile = np.concatenate(true_x_tile_by_batch_list, axis=0)
        y_rmse_all_steps_list = []
        y_qice_all_steps_list = []
        y_picp_all_steps_list = []
        y_nll_all_steps_list = []

        if config.testing.compute_metric_all_steps:
            for idx in range(self.num_timesteps + 1):
                current_t = self.num_timesteps - idx
                # compute RMSE
                y_rmse = np.sqrt(np.mean(y_se_by_batch_list[current_t]))
                y_rmse_all_steps_list.append(y_rmse)
                # compute QICE
                all_gen_y = gen_y_by_batch_list[current_t]
                y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
                    config=config, dataset_object=dataset_object,
                    all_true_y=all_true_y, all_generated_y=all_gen_y, verbose=False)
                y_qice_all_steps_list.append(qice_coverage_ratio)
                # compute PICP
                coverage, _, _ = compute_PICP(config=config, y_true=y_true, all_gen_y=all_gen_y)
                y_picp_all_steps_list.append(coverage)
                # compute NLL
                y_nll = np.mean(nll_by_batch_list[current_t])
                y_nll_all_steps_list.append(y_nll)
            # make plot for metrics across all timesteps during reverse diffusion
            n_metrics = 4
            fig, axs = plt.subplots(n_metrics, 1, figsize=(8.5, n_metrics * 3), clear=True)  # W x H
            plt.subplots_adjust(hspace=0.5)
            xticks = np.arange(0, self.num_timesteps + 1, config.diffusion.vis_step)
            # RMSE
            axs[0].plot(y_rmse_all_steps_list)
            # axs[0].set_title('y RMSE across All Timesteps', fontsize=18)
            axs[0].set_xlabel('timestep', fontsize=12)
            axs[0].set_xticks(xticks)
            axs[0].set_xticklabels(xticks[::-1])
            axs[0].set_ylabel('y RMSE', fontsize=12)
            # QICE
            axs[1].plot(y_qice_all_steps_list)
            # axs[1].set_title('y QICE across All Timesteps', fontsize=18)
            axs[1].set_xlabel('timestep', fontsize=12)
            axs[1].set_xticks(xticks)
            axs[1].set_xticklabels(xticks[::-1])
            axs[1].set_ylabel('y QICE', fontsize=12)
            # PICP
            picp_ideal = (config.testing.PICP_range[1] - config.testing.PICP_range[0]) / 100
            axs[2].plot(y_picp_all_steps_list)
            axs[2].axhline(y=picp_ideal, c='b')
            # axs[2].set_title('y PICP across All Timesteps', fontsize=18)
            axs[2].set_xlabel('timestep', fontsize=12)
            axs[2].set_xticks(xticks)
            axs[2].set_xticklabels(xticks[::-1])
            axs[2].set_ylabel('y PICP', fontsize=12)
            # NLL
            axs[3].plot(y_nll_all_steps_list)
            # axs[3].set_title('y NLL across All Timesteps', fontsize=18)
            axs[3].set_xlabel('timestep', fontsize=12)
            axs[3].set_xticks(xticks)
            axs[3].set_xticklabels(xticks[::-1])
            axs[3].set_ylabel('y NLL', fontsize=12)
            # fig.suptitle('y Metrics across All Timesteps')
            fig.savefig(os.path.join(args.im_path, 'metrics_all_timesteps.pdf'))
        else:
            # compute RMSE
            y_rmse = np.sqrt(np.mean(y_se_by_batch_list[config.testing.mean_t]))
            y_rmse_all_steps_list.append(y_rmse)
            # compute QICE -- a cover metric
            all_gen_y = gen_y_by_batch_list[config.testing.coverage_t]
            y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
                config=config, dataset_object=dataset_object,
                all_true_y=all_true_y, all_generated_y=all_gen_y, verbose=True)
            y_qice_all_steps_list.append(qice_coverage_ratio)
            logging.info("\nWe generated {} y's given each x.".format(config.testing.n_z_samples))
            logging.info(("\nRMSE between true mean y and the mean of generated y given each x is " +
                          "{:.8f};\nQICE between true y coverage ratio by each generated y " +
                          "quantile interval and optimal ratio is {:.8f}.").format(y_rmse, qice_coverage_ratio))
            # compute PICP -- another coverage metric
            coverage, low, high = compute_PICP(config=config, y_true=y_true, all_gen_y=all_gen_y)
            y_picp_all_steps_list.append(coverage)
            logging.info(("There are {:.4f}% of true test y in the range of " +
                          "the computed {:.0f}% credible interval.").format(100 * coverage, high - low))
            # compute NLL
            y_nll = np.mean(nll_by_batch_list[config.testing.nll_t])
            y_nll_all_steps_list.append(y_nll)
            logging.info("\nNegative Log-Likelihood on test set is {:.8f}.".format(y_nll))

        logging.info(f"y RMSE at all steps: {y_rmse_all_steps_list}.\n")
        logging.info(f"y QICE at all steps: {y_qice_all_steps_list}.\n")
        logging.info(f"y PICP at all steps: {y_picp_all_steps_list}.\n\n")
        logging.info(f"y NLL at all steps: {y_nll_all_steps_list}.\n\n")

        # make plots for true vs. generated distribution comparison
        if config.testing.make_plot:
            assert config.data.dataset != "uci"
            all_gen_y = gen_y_by_batch_list[config.testing.vis_t]
            # compute QICE
            y_true_ratio_by_bin, qice_coverage_ratio, y_true = compute_true_coverage_by_gen_QI(
                config=config, dataset_object=dataset_object,
                all_true_y=all_true_y, all_generated_y=all_gen_y,
                verbose=False)
            logging.info(("\nQICE between true y coverage ratio by each generated y " +
                          "quantile interval and optimal ratio is {:.8f}.").format(
                qice_coverage_ratio))
            # compute PICP and RMSE
            if config.data.no_multimodality:
                if not config.data.inverse_xy:
                    coverage, CI_y_pred, low, high = compute_PICP(config=config,
                                                                  y_true=y_true,
                                                                  all_gen_y=all_gen_y,
                                                                  return_CI=True)
            # compute mean predicted y given each x
            y_pred_mean = compute_prediction_SE(config=config, dataset_object=dataset_object,
                                                y_batch=torch.from_numpy(all_true_y),
                                                generated_y=all_gen_y,
                                                return_pred_mean=True)

            # create plot
            logging.info("\nNow we start making the plot...")
            n_fig_rows, n_fig_cols = 3, 1
            fig, (ax1, ax2, ax3) = plt.subplots(n_fig_rows, n_fig_cols, clear=True)
            fig.set_figheight(config.testing.fig_size[0])
            fig.set_figwidth(config.testing.fig_size[1])
            # make individual plot to be organized into grid plot for the paper
            fig_1, ax_1 = plt.subplots(1, 1, clear=True)
            fig_1.set_figheight(config.testing.one_fig_size[0])
            fig_1.set_figwidth(config.testing.one_fig_size[1])
            fig_2, ax_2 = plt.subplots(1, 1, clear=True)
            fig_2.set_figheight(config.testing.one_fig_size[0])
            fig_2.set_figwidth(config.testing.one_fig_size[1])
            fig_3, ax_3 = plt.subplots(1, 1, clear=True)
            fig_3.set_figheight(config.testing.one_fig_size[0])
            fig_3.set_figwidth(config.testing.one_fig_size[1])
            # un-normalize y to its original scale for plotting
            if dataset_object.normalize_y:
                all_true_y = dataset_object.scaler_y.inverse_transform(all_true_y).astype(np.float32)
                all_gen_y = all_gen_y.reshape(dataset_object.test_n_samples * config.testing.n_z_samples,
                                              config.model.y_dim)
                all_gen_y = dataset_object.scaler_y.inverse_transform(all_gen_y).astype(np.float32).reshape(
                    dataset_object.test_n_samples,
                    config.testing.n_z_samples,
                    config.model.y_dim)

            ################## make first plot (only for toy data with 1D x) ##################
            if config.testing.squared_plot:
                ax1.set(aspect='equal', adjustable='box')
            if config.data.inverse_xy:
                x_noiseless_mean = compute_y_noiseless_mean(dataset_object,
                                                            torch.from_numpy(all_true_y),
                                                            config.data.true_function)
                if config.model.y_dim == 1:
                    sorted_idx_ = np.argsort(all_true_y, axis=0).squeeze()
                    ax1.plot(x_noiseless_mean[sorted_idx_], all_true_y[sorted_idx_],
                             c='orange', alpha=1, label='true-noiseless')
            if config.data.no_multimodality:
                logging.info("\nThe toy dataset doesn't contain multimodality.")
                if not config.data.inverse_xy:
                    y_rmse = np.sqrt(np.mean((y_pred_mean - all_true_y) ** 2))
                    logging.info(("\nRMSE between true y and the mean of generated y given each x is " +
                                  "{:.8f}.").format(y_rmse))
                    # obtain noiseless mean with ground truth data generation function
                    y_noiseless_mean = compute_y_noiseless_mean(dataset_object,
                                                                torch.from_numpy(all_true_x),
                                                                config.data.true_function)
                    logging.info(("\nRMSE between true expected y and the mean of generated y given each x is " +
                                  "{:.8f}.").format(np.sqrt(np.mean((y_pred_mean - y_noiseless_mean) ** 2))))
            n_true_x_for_plot_scale = 2
            if config.testing.plot_true:
                n_total_true = all_true_x.shape[0]
                true_sampled_idx = np.random.choice(
                    np.arange(n_total_true), size=n_total_true // n_true_x_for_plot_scale, replace=False)
                ax1.scatter(all_true_x[true_sampled_idx], all_true_y[true_sampled_idx],
                            s=2, c='r', marker="o", alpha=0.5, label='true')
                ax_1.scatter(all_true_x[true_sampled_idx], all_true_y[true_sampled_idx],
                             s=2, c='r', marker="o", alpha=0.5, label='true')
            if config.testing.plot_gen:
                # if sample the same idx for each test x, the sampled generated y tend to
                # follow a smooth trace instead of scattered randomly
                samp_idx = np.random.randint(low=0, high=config.testing.n_z_samples,
                                             size=(dataset_object.test_n_samples,
                                                   n_samples_gen_y_for_plot,
                                                   config.model.y_dim))
                all_gen_y_ = np.take_along_axis(all_gen_y, indices=samp_idx, axis=1)
                if len(all_gen_y_.shape) == 3:
                    all_gen_y_ = all_gen_y_.reshape(dataset_object.test_n_samples * n_samples_gen_y_for_plot,
                                                    config.model.y_dim)
                n_total_samples = all_true_x_tile.shape[0]
                gen_sampled_idx = np.random.choice(
                    np.arange(n_total_samples),
                    size=n_total_samples // (n_true_x_for_plot_scale * n_samples_gen_y_for_plot), replace=False)
                ax1.scatter(all_true_x_tile[gen_sampled_idx], all_gen_y_[gen_sampled_idx],
                            s=2, c='b', marker="^", alpha=0.5, label='generated')
                ax_1.scatter(all_true_x_tile[gen_sampled_idx], all_gen_y_[gen_sampled_idx],
                             s=2, c='b', marker="^", alpha=0.5, label='generated')
            if config.data.no_multimodality:
                if not config.data.inverse_xy:
                    logging.info("\nWe generated {} y's given each x.".format(config.testing.n_z_samples))
                    logging.info(("There are {:.4f}% of true test y in the range of " +
                                  "the computed {:.0f}% credible interval.").format(
                        100 * coverage, high - low))
                    # make sure input to x argument is properly sorted
                    if config.data.normalize_y:
                        CI_y_pred_lower = dataset_object.scaler_y.inverse_transform(
                            CI_y_pred[0].reshape(-1, 1)).flatten()
                        CI_y_pred_higher = dataset_object.scaler_y.inverse_transform(
                            CI_y_pred[1].reshape(-1, 1)).flatten()
                    else:
                        CI_y_pred_lower = CI_y_pred[0]
                        CI_y_pred_higher = CI_y_pred[1]
                    ax1.fill_between(x=all_true_x.squeeze(),
                                     y1=CI_y_pred_lower,
                                     y2=CI_y_pred_higher,
                                     facecolor='grey',
                                     alpha=0.6)
                    ax_1.fill_between(x=all_true_x.squeeze(),
                                      y1=CI_y_pred_lower,
                                      y2=CI_y_pred_higher,
                                      facecolor='grey',
                                      alpha=0.6)
            ax_1.legend(loc='best')
            ax1.legend(loc='best')
            ax1.set_title('True vs. Generated y Given x')
            ax_1.legend(loc='best')
            ax_1.set_xlabel('$x$', fontsize=10)
            ax_1.set_ylabel('$y$', fontsize=10)
            fig_1.savefig(os.path.join(args.im_path, 'gen_vs_true_scatter.png'), dpi=1200, bbox_inches='tight')

            ################## make second plot ##################
            n_bins = config.testing.n_bins
            optimal_ratio = 1 / n_bins
            all_bins = np.arange(n_bins) + 1

            ax2.bar(all_bins, y_true_ratio_by_bin, label='quantile coverage')
            ax2.hlines(optimal_ratio, all_bins[0] - 1, all_bins[-1] + 1, colors='r', label='optimal ratio')
            ax2.set_xticks(all_bins[::2])
            ax2.set_xlim(all_bins[0] - 1, all_bins[-1] + 1)
            ax2.set_ylim([0, 0.5])
            ax2.set_title('Ratio of True y \nin Each Quantile Interval of Generated y')
            ax2.legend(loc='best')
            ax_2.bar(all_bins, y_true_ratio_by_bin, label='quantile coverage')
            ax_2.hlines(optimal_ratio, all_bins[0] - 1, all_bins[-1] + 1, colors='r', label='optimal ratio')
            ax_2.set_xticks(all_bins[::2])
            ax_2.set_xlim(all_bins[0] - 1, all_bins[-1] + 1)
            ax_2.set_ylim([0, 0.5])
            ax_2.legend(loc='best')
            fig_2.savefig(os.path.join(args.im_path, 'quantile_interval_coverage.png'), dpi=1200)

            ################## make third plot ##################
            bins_norm_on_y = all_bins * y_true_ratio_by_bin / optimal_ratio
            ax3.set(aspect='equal', adjustable='box')
            ax3.scatter(all_bins, bins_norm_on_y)
            ax3.plot([-1, all_bins[-1] + 1], [-1, all_bins[-1] + 1], c='orange')
            ax3.set_xticks(all_bins[::2])
            ax3.set_yticks(all_bins[::2])
            ax3.set_xlim([-1, all_bins[-1] + 1])
            ax3.set_ylim([-1, all_bins[-1] + 1])
            ax3.set_title('Ratio of True y vs. Optimal Ratio \nin Each Quantile Interval of Generated y')
            ax_3.set(aspect='equal', adjustable='box')
            ax_3.scatter(all_bins, bins_norm_on_y)
            ax_3.plot([-1, all_bins[-1] + 1], [-1, all_bins[-1] + 1], c='orange')
            ax_3.set_xticks(all_bins[::2])
            ax_3.set_yticks(all_bins[::2])
            ax_3.set_xlim([-1, all_bins[-1] + 1])
            ax_3.set_ylim([-1, all_bins[-1] + 1])
            fig_3.savefig(os.path.join(args.im_path, 'quantile_interval_coverage_true_vs_optimal.png'), dpi=1200)

            fig.tight_layout()
            fig.savefig(os.path.join(args.im_path, 'gen_vs_true_distribution_vis.png'), dpi=1200)

        # clear the memory
        plt.close('all')
        del true_y_by_batch_list
        if config.testing.make_plot and config.data.dataset != "uci":
            del all_true_x
        if config.testing.plot_gen:
            del all_true_x_tile
        del gen_y_by_batch_list
        del y_se_by_batch_list
        gc.collect()

        return y_rmse_all_steps_list, y_qice_all_steps_list, y_picp_all_steps_list, y_nll_all_steps_list
