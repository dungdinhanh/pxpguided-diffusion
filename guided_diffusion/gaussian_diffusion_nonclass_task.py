"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""


from guided_diffusion.gaussian_diffusion_mlt import *
from guided_diffusion.respace import *


class GaussianDiffusionClassFree(GaussianDiffusionMLT2):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFree, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)

    def p_sample(
            self,
            model,
            x,
            t,
            w_cond=0.5,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        self.t = t
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, w_cond, clip_denoised, denoised_fn, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def process_xstart(self, x, denoised_fn=None, clip_denoised=True):
        if denoised_fn is not None:
            x = denoised_fn(x)
        if clip_denoised:
            return x.clamp(-1, 1)
        return x

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        eps_gen = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        new_eps = (1+w) * eps_gen_cond - w * eps_gen
        new_predxstart = self.process_xstart(self._predict_xstart_from_eps(x, t, new_eps), denoised_fn, clip_denoised)
        new_mean, _, _ = self.q_posterior_mean_variance(new_predxstart, x, t)

        del eps_gen, eps_gen_cond, out_cond, new_eps, new_predxstart
        return new_mean

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_mtl(self._wrap_model(cond_fn), *args, **kwargs)

    def p_sample_loop(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            w_cond=0.5,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                w_cond=w_cond
        ):
            final = sample
        return final["sample"]


    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            w_cond=0.5
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    w_cond=w_cond
                )
                yield out
                img = out["sample"]

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


class GaussianDiffusionClassFreeMLT2(GaussianDiffusionClassFree):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFreeMLT2, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                           model_mean_type=model_mean_type,
                                                           model_var_type=model_var_type,
                                                           loss_type=loss_type,
                                                           rescale_timesteps=rescale_timesteps)


    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        eps_gen =  self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])

        delta_gen_cond = -w * (eps_gen_cond - eps_gen) * _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        gradient_cond = out_cond['pred_xstart']

        new_gradient_cond = self.project_conflict(gradient_cond.clone(), delta_gen_cond.clone(), gradient_cond.shape)
        new_delta_gen_cond = self.project_conflict(delta_gen_cond.clone(), gradient_cond.clone(), gradient_cond.shape)

        # new_eps = new_gradient_cond + new_delta_gen_cond
        new_predxstart = self.process_xstart(new_gradient_cond + new_delta_gen_cond, denoised_fn=denoised_fn, clip_denoised=clip_denoised)

        new_mean, _, _ = self.q_posterior_mean_variance(new_predxstart, x, t)
        # del eps_gen, eps_gen_cond, out_cond, new_eps, new_predxstart, new_eps_gen_cond, new_delta_gen_cond
        return new_mean

class GaussianDiffusionClassFreeMLT3(GaussianDiffusionClassFree):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFreeMLT3, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                           model_mean_type=model_mean_type,
                                                           model_var_type=model_var_type,
                                                           loss_type=loss_type,
                                                           rescale_timesteps=rescale_timesteps)


    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        # eps_gen_weight = - w * self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        # eps_gen_cond_weight = (1+w) * self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        #
        # # eps_gen_weight = - self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        # # eps_gen_cond_weight = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        # # new_eps = (1+w) * eps_gen_cond - w * eps_gen
        #
        # new_eps_gen_weight = self.project_conflict(eps_gen_weight.clone(), eps_gen_cond_weight.clone(), eps_gen_weight.shape)
        # new_eps_gen_cond_weight = self.project_conflict(eps_gen_cond_weight.clone(), eps_gen_weight.clone(), eps_gen_weight.shape)
        eps_gen = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        delta_gen_cond = w * (eps_gen_cond - eps_gen)

        new_eps_gen_cond = eps_gen_cond.clone()
        new_delta_gen_cond = self.project_conflict(delta_gen_cond.clone(), eps_gen_cond.clone(), eps_gen_cond.shape)

        new_eps = new_eps_gen_cond + new_delta_gen_cond
        # print("_______________________________________________")
        # print(eps_gen_cond.norm())
        # print(new_eps_gen_cond.norm())
        # print(delta_gen_cond.norm())
        # print(new_delta_gen_cond.norm())
        # print(new_eps.norm())
        # print("____________________________________________")
        new_predxstart = self.process_xstart(self._predict_xstart_from_eps(x, t, new_eps), denoised_fn, clip_denoised)
        new_mean, _, _ = self.q_posterior_mean_variance(new_predxstart, x, t)
        del eps_gen, eps_gen_cond, out_cond, new_eps, new_predxstart, new_eps_gen_cond, new_delta_gen_cond
        return new_mean


class GaussianDiffusionClassFreeMLT4(GaussianDiffusionClassFree):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFreeMLT4, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                           model_mean_type=model_mean_type,
                                                           model_var_type=model_var_type,
                                                           loss_type=loss_type,
                                                           rescale_timesteps=rescale_timesteps)


    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        # eps_gen_weight = - w * self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        # eps_gen_cond_weight = (1+w) * self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        #
        # # eps_gen_weight = - self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        # # eps_gen_cond_weight = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        # # new_eps = (1+w) * eps_gen_cond - w * eps_gen
        #
        # new_eps_gen_weight = self.project_conflict(eps_gen_weight.clone(), eps_gen_cond_weight.clone(), eps_gen_weight.shape)
        # new_eps_gen_cond_weight = self.project_conflict(eps_gen_cond_weight.clone(), eps_gen_weight.clone(), eps_gen_weight.shape)
        eps_gen = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        delta_gen_cond = w * (eps_gen_cond - eps_gen)

        new_eps_gen_cond = self.project_conflict(eps_gen_cond.clone(), delta_gen_cond.clone(), eps_gen_cond.shape)
        new_delta_gen_cond = delta_gen_cond.clone()

        new_eps = new_eps_gen_cond + new_delta_gen_cond
        # print("_______________________________________________")
        # print(eps_gen_cond.norm())
        # print(new_eps_gen_cond.norm())
        # print(delta_gen_cond.norm())
        # print(new_delta_gen_cond.norm())
        # print(new_eps.norm())
        # print("____________________________________________")
        new_predxstart = self.process_xstart(self._predict_xstart_from_eps(x, t, new_eps), denoised_fn, clip_denoised)
        new_mean, _, _ = self.q_posterior_mean_variance(new_predxstart, x, t)
        del eps_gen, eps_gen_cond, out_cond, new_eps, new_predxstart, new_eps_gen_cond, new_delta_gen_cond
        return new_mean


class GaussianDiffusionClassFreeMLT5(GaussianDiffusionClassFree):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFreeMLT5, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                           model_mean_type=model_mean_type,
                                                           model_var_type=model_var_type,
                                                           loss_type=loss_type,
                                                           rescale_timesteps=rescale_timesteps)


    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        eps_gen = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])
        delta_gen_cond = w * (eps_gen_cond - eps_gen)

        x0_direction_gen = _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * p_mean_var['pred_xstart']
        delta_gen_cond_direction = - _extract_into_tensor(self.posterior_mean_coef1, t, x.shape) *\
                                   _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * delta_gen_cond

        new_x0_direction_gen = self.project_conflict(x0_direction_gen.clone(), delta_gen_cond_direction.clone(),
                                                     x0_direction_gen.shape)
        new_delta_gen_cond_direction = self.project_conflict(delta_gen_cond_direction.clone(), x0_direction_gen.clone(),
                                                             x0_direction_gen.shape)

        new_mean, _, _ = out_cond['mean'].float() - x0_direction_gen + new_x0_direction_gen \
                         + new_delta_gen_cond_direction
        del eps_gen, eps_gen_cond, out_cond, new_x0_direction_gen, new_delta_gen_cond_direction, \
            x0_direction_gen, delta_gen_cond_direction
        return new_mean


class GaussianDiffusionClassFreeMLT6(GaussianDiffusionClassFree):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionClassFreeMLT6, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                           model_mean_type=model_mean_type,
                                                           model_var_type=model_var_type,
                                                           loss_type=loss_type,
                                                           rescale_timesteps=rescale_timesteps)
        self.den_weight = 1.0
        self.cond_weight = 1.0

    def project_conflict(self, grad1, grad2, shape, u_weight=1.0):
        new_grad1 = torch.flatten(grad1, start_dim=1)
        new_grad2 = torch.flatten(grad2, start_dim=1)

        # g1 * g2 --------------- (batchsize,)
        g_1_g_2 = torch.sum(new_grad1 * new_grad2, dim=1)
        g_1_g_2 = torch.clamp(g_1_g_2, max=0.0)

        # ||g2||^2 ----------------- (batchsize,)
        norm_g2 = new_grad2.norm(dim=1) **2
        if torch.any(norm_g2 == 0.0):
            return new_grad1.view(shape)

        # (g1 * g2)/||g2||^2 ------------------- (batchsize,)
        g12_o_normg2 = g_1_g_2/norm_g2
        g12_o_normg2 = torch.unsqueeze(g12_o_normg2, dim=1)
        # why zero has problem?
        # g1
        new_grad1 -= u_weight * ((g12_o_normg2) * new_grad2)
        new_grad1 = new_grad1.view(shape)
        return new_grad1

    def condition_mean_mtl(self, cond_fn, p_mean_var, x, t, w=0.5, clip_denoised=True, denoised_fn=None,
                           model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        out_cond = self.p_mean_variance(cond_fn, x, t, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                        model_kwargs=model_kwargs)

        eps_gen =  self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps_gen_cond = self._predict_eps_from_xstart(x, t, out_cond['pred_xstart'])

        delta_gen_cond = -w * (eps_gen_cond - eps_gen) * _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        gradient_cond = out_cond['pred_xstart']

        new_gradient_cond = self.project_conflict(gradient_cond.clone(), delta_gen_cond.clone(), gradient_cond.shape, self.den_weight)
        new_delta_gen_cond = self.project_conflict(delta_gen_cond.clone(), gradient_cond.clone(), gradient_cond.shape, self.cond_weight)

        # new_eps = new_gradient_cond + new_delta_gen_cond
        new_predxstart = self.process_xstart(new_gradient_cond + new_delta_gen_cond, denoised_fn=denoised_fn, clip_denoised=clip_denoised)

        new_mean, _, _ = self.q_posterior_mean_variance(new_predxstart, x, t)
        # del eps_gen, eps_gen_cond, out_cond, new_eps, new_predxstart, new_eps_gen_cond, new_delta_gen_cond
        return new_mean

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
