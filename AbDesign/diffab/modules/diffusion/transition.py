import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffab.utils.misc import hotspot_distance_fn, pair2edge,batchfy,clash_loss
from diffab.modules.common.layers import clampped_one_hot
from diffab.modules.common.so3 import ApproxAngularDistribution, random_normal_so3, so3vec_to_rotation, rotation_to_so3vec


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)
 
        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)
        # calculate X0 = sqrt_recip_alphas_cumprod * Xt - sqrt_recipm1_alphas_cumprod * noise
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alpha_bars))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alpha_bars - 1))
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        self.posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        
    def to(self,device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
                
        
        
class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
         
    def add_noise(self, p_0, mask_generate,  t):
        """
        Args:
            p_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        nb_non_batch_dims = len(p_0.shape) - 1
        c0 = torch.sqrt(alpha_bar).view(-1, *(nb_non_batch_dims*(1,)))
        c1 = torch.sqrt(1 - alpha_bar).view(-1, *(nb_non_batch_dims*(1,)))
        prior_p = torch.randn_like(p_0)
           
        p_noisy = c0*p_0 + c1*prior_p
        p_noisy = torch.where(mask_generate[..., None].expand_as(p_0), p_noisy, p_0)

        return p_noisy, prior_p
    
    def pred_start_from_noise(self, p_t, eps_p, mask_generate, t):
        nb_non_batch_dims = len(p_t.shape) - 1
        sqrt_recip_alphas_cumprod = self.var_sched.sqrt_recip_alphas_cumprod[t].view(-1, *(nb_non_batch_dims*(1,)))
        sqrt_recipm1_alphas_cumprod = self.var_sched.sqrt_recipm1_alphas_cumprod[t].view(-1, *(nb_non_batch_dims*(1,)))
        
        p_0 = sqrt_recip_alphas_cumprod * p_t - sqrt_recipm1_alphas_cumprod * eps_p

        p_0 = torch.where(mask_generate[..., None].expand_as(p_t), p_0, p_t)
        return p_0
    
    def pred_noise_from_start(self, p_t, p_0, mask_generate, t):
        nb_non_batch_dims = len(p_t.shape) - 1
        sqrt_recip_alphas_cumprod = self.var_sched.sqrt_recip_alphas_cumprod[t].view(-1, *(nb_non_batch_dims*(1,)))
        sqrt_recipm1_alphas_cumprod = self.var_sched.sqrt_recipm1_alphas_cumprod[t].view(-1, *(nb_non_batch_dims*(1,)))
        eps_p = (sqrt_recip_alphas_cumprod * p_t - p_0) / sqrt_recipm1_alphas_cumprod

        eps_p = torch.where(mask_generate[..., None].expand_as(p_t), eps_p, p_t)
        return eps_p
    def denoise_from_p0(self, p_t, p_0, mask_generate, t,guidance_kwargs=None):
        self.var_sched.to(p_t.device)
        nb_non_batch_dims = len(p_t.shape) - 1
        posterior_mean = (
            self.var_sched.posterior_mean_coef1[t].view(-1, *(nb_non_batch_dims*(1,))) * p_0 +
            self.var_sched.posterior_mean_coef2[t].view(-1, *(nb_non_batch_dims*(1,))) * p_t
        )
        posterior_variance = self.var_sched.posterior_variance[t].view(-1, *(nb_non_batch_dims*(1,)))
        posterior_log_variance_clipped = self.var_sched.posterior_log_variance_clipped[t].view(-1, *(nb_non_batch_dims*(1,)))
        
        # with torch.enable_grad():
        #     batched_mean = batchfy(posterior_mean,guidance_kwargs['lengths']).detach().requires_grad_(True)
        #     hotspot_mask = guidance_kwargs['to_hotspot_dist'] != 0
        #     hotspot_idx = (torch.where(hotspot_mask)[1])
        #     cdr_idx = (torch.where(guidance_kwargs['generate_flag'])[1])
        #     cdr2hotspot_dist = hotspot_distance_fn(batched_mean, hotspot_idx, cdr_idx)
            
        #     grad = torch.autograd.grad(cdr2hotspot_dist.sum(), batched_mean)[0] * 1
        #     grad = grad[guidance_kwargs['mask']]
        #     posterior_mean = posterior_mean -  grad #* posterior_variance 
        #     print(f"\n dist: {10*cdr2hotspot_dist.mean():.3f},\
        #          native dist: {guidance_kwargs['to_hotspot_dist'].max():.3f}\
        #         grad: {grad[mask_generate].mean()},  posterior_mean {posterior_mean[mask_generate].mean()}")  
        
        z = torch.where(
            (t > 1)[...,None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )
        p_next = posterior_mean + (0.5 * posterior_log_variance_clipped).exp() * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        
        return p_next
    
    
    def denoise(self, p_t, eps_p, mask_generate, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        nb_non_batch_dims = len(p_t.shape) - 1
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, *(nb_non_batch_dims*(1,)))

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(-1, *(nb_non_batch_dims*(1,)))
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(-1, *(nb_non_batch_dims*(1,)))
        
        if len(mask_generate.shape) == 2:
            t = t[:, None].expand(*mask_generate.shape)
        elif len(mask_generate.shape) == 1:
            assert len(t.shape) == 1
        else:
            raise ValueError('mask_generate must be 1D or 2D')
        
        z = torch.where(
            (t > 1)[...,None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )
        mean = c0 * (p_t - c1 * eps_p)
        p_next = mean + sigma * z
        p_next = torch.where(mask_generate[..., None].expand_as(p_t), p_next, p_t)
        return p_next


class RotationTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}, angular_distrib_fwd_opt={}, angular_distrib_inv_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        # Forward (perturb)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars) # (T,).
        self.angular_distrib_fwd = ApproxAngularDistribution(c1.tolist(), **angular_distrib_fwd_opt)

        # Inverse (generate)
        sigma = self.var_sched.sigmas
        self.angular_distrib_inv = ApproxAngularDistribution(sigma.tolist(), **angular_distrib_inv_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))
    
    def add_noise(self, v_0, mask_generate, t, ):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        """
        nb_non_batch_dims = len(v_0.shape) - 1
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, *(nb_non_batch_dims*(1,)))
        c1 = torch.sqrt(1 - alpha_bar).view(-1, *(nb_non_batch_dims*(1,)))

        # Noise rotation
        if len(mask_generate.shape) == 2:
            t = t[:, None].expand(*mask_generate.shape)
        elif len(mask_generate.shape) == 1:
            assert len(t.shape) == 1
        else:
            raise ValueError('mask_generate must be 1D or 2D')
        e_scaled = random_normal_so3(t, self.angular_distrib_fwd, device=self._dummy.device)    # (N, L, 3)
        e_normal = e_scaled / (c1 + 1e-8)
        E_scaled = so3vec_to_rotation(e_scaled)   # (N, L, 3, 3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (N, L, 3, 3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        if len(mask_generate.shape) == 2:
            # [N, L]
            t = t[:, None].expand(*mask_generate.shape)
        elif len(mask_generate.shape) == 1:
            # [nb_all_res,]
            assert len(t.shape) == 1
        else:
            raise ValueError('mask_generate must be 1D or 2D')
        
        e = random_normal_so3(t, self.angular_distrib_inv, device=self._dummy.device) # (N, L, 3)
        e = torch.where(
            (t > 1)[...,None].expand_as(e),
            e, 
            torch.zeros_like(e) # Simply denoise and don't add noise at the last step
        )
        E = so3vec_to_rotation(e)

        R_next = E @ so3vec_to_rotation(v_next)
        v_next = rotation_to_so3vec(R_next)
        v_next = torch.where(mask_generate[..., None].expand_as(v_next), v_next, v_t)

        return v_next


class AminoacidCategoricalTransition(nn.Module):
    
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L).
        """
        N, L, K = c.size()
        c = c.view(N*L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N, L)
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            c_t:    Probability, (N, L, K).
            x_t:    Sample, LongTensor, (N, L).
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K).
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)
        c_noisy = (alpha_bar*c_0) + ( (1-alpha_bar)/K )
        c_t = torch.where(mask_generate[..., None].expand(N,L,K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        """
        Args:
            x_t:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            x_0:    Category LongTensor (N, L) or Probability FloatTensor (N, L, K).
            t:  (N,).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 3:
            c_t = x_t   # When x_t is probability distribution.
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float() # (N, L, K)

        if x_0.dim() == 3:
            c_0 = x_0   # When x_0 is probability distribution.
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, L, K)

        alpha = self.var_sched.alpha_bars[t][:, None, None]     # (N, 1, 1)
        alpha_bar = self.var_sched.alpha_bars[t][:, None, None] # (N, 1, 1)

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)   # (N, L, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta

    def denoise(self, x_t, c_0_pred, mask_generate, t):
        """
        Args:
            x_t:        (N, L).
            c_0_pred:   Normalized probability predicted by networks, (N, L, K).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (N, L, K)
        post = self.posterior(c_t, c_0_pred, t=t)   # (N, L, K)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next
