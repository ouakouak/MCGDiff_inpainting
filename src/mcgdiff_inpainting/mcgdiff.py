"""MCGdiff noiseless inpainting using Chopin's particles library."""

from __future__ import annotations

import numpy as np
import particles


class MCGdiff(particles.FeynmanKac):
    """MCGdiff for noiseless image inpainting.

    Expected diffusion interface:
        - diffusion.alphas_cumprod:
            Array of cumulative alphas, with alpha[0] = 1.

        - diffusion.transition(x, t, t_prev):
            Returns (mean, std) for the backward transition
            from diffusion level t to t_prev.

            x:    shape (N, d)
            mean: shape (N, d)
            std:  positive scalar or broadcastable array

    Arguments:
        - diffusion:
            Pretrained diffusion model adapter.

        - observation:
            Observed pixel values, shape (d_obs,).

        - mask:
            Boolean array of shape (d,).
            True = observed pixel.
            False = missing pixel.

        - timesteps:
            Strictly decreasing diffusion levels, ending at 0.
            Level 0 represents the clean image.
    """


    def __init__(self, diffusion, observation, mask, timesteps):

        self.diffusion = diffusion

        # Observation mask
        self.mask = np.asarray(mask, dtype=bool).reshape(-1)

        self.obs_idx = np.flatnonzero(self.mask)
        self.free_idx = np.flatnonzero(~self.mask)

        self.d = self.mask.size

        # Observed pixel values
        self.y = np.asarray(
            observation, dtype=float
        ).reshape(-1)

        # Diffusion levels: n -> ... -> 0
        self.timesteps = tuple(int(t) for t in timesteps)

        if len(self.timesteps) < 2:
            raise ValueError("At least two timesteps are required.")

        if self.timesteps[-1] != 0:
            raise ValueError("The final timestep must be 0.")

        if any(
            a <= b
            for a, b in zip(
                self.timesteps,
                self.timesteps[1:]
            )
        ):
            raise ValueError(
                "Timesteps must be strictly decreasing."
            )

        if len(self.y) != len(self.obs_idx):
            raise ValueError(
                "Observation size does not match the mask."
            )

        # Diffusion noise schedule
        self.alpha = np.asarray(
            diffusion.alphas_cumprod,
            dtype=float
        )

        if not np.isclose(self.alpha[0], 1.0):
            raise ValueError(
                "alpha[0] must equal 1 (clean image level)."
            )

        for t in self.timesteps[:-1]:
            if not 0 < self.alpha[t] < 1:
                raise ValueError(
                    "Noisy levels must satisfy 0 < alpha[t] < 1."
                )

        super().__init__(T=len(self.timesteps))


    @staticmethod
    def _log_normal_diag(x, mean, variance):
        """Row-wise log density of a diagonal Gaussian."""

        variance = np.asarray(variance, dtype=float)

        if np.any(variance <= 0):
            raise ValueError(
                "Gaussian variance must be strictly positive."
            )

        return -0.5 * np.sum(
            (x - mean) ** 2 / variance
            + np.log(2 * np.pi * variance),
            axis=-1
        )


    def M0(self, N):
        """Sample N times from initial distribution M_0 of the FK model
        
        Arguments;
            - N: number of particles to sample
            
        Return:
            N particles from M_0 distribution
        """
        
        t = self.timesteps[0]
        alpha = self.alpha[t]

        # Initial gaussian prior of diffusion ~ N(0, Id)
        x = np.random.standard_normal((N,self.d)) # Matrix of size (N, d) from N(0, 1)

        # Filled observed pixels with diffusion correction
        mean = np.sqrt(alpha) * self.y / (2 - alpha)
        std = np.sqrt((1 - alpha) / (2 - alpha))

        x[:, self.obs_idx] = (
            mean
            + std
            * np.random.standard_normal(
                (N, len(self.obs_idx))
            )
        )

        return x

    # ---------------------------------------------------------
    # Diffusion transition parameters
    # ---------------------------------------------------------

    def _step_info(self, k, xp):
        """Get backward Gaussian parameters from diffusion."""

        parent_t = self.timesteps[k - 1]
        child_t = self.timesteps[k]

        mean, std = self.diffusion.transition(
            xp,
            parent_t,
            child_t
        )

        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)

        # if mean.shape != xp.shape:
        #     raise ValueError(
        #         "Transition mean must have shape (N, d)."
        #     )

        # if np.any(std <= 0):
        #     raise ValueError(
        #         "Transition std must be positive."
        #     )

        return parent_t, child_t, mean, std


    # ---------------------------------------------------------
    # Guiding potential for guided diffusion
    # ---------------------------------------------------------

    def _log_g(self, x, t):
        """Log guiding potential at diffusion level t."""

        a = self.alpha[t]

        return self._log_normal_diag(
            x[:, self.obs_idx],
            np.sqrt(a) * self.y,
            1 - a
        )


    # ---------------------------------------------------------
    # Auxiliary weights
    # ---------------------------------------------------------

    def _log_omega(self, k, xp):
        """Compute log weight """

        parent_t, child_t, mean, std = (
            self._step_info(k, xp)
        )

        a = self.alpha[child_t]

        # Normalization constant of guided transition
        #
        # C = integral p(child | parent) g_child(child) dchild
        #
        # At child_t = 0, g_0 is an exact observation
        # constraint (Dirac), handled by the same formula.

        mean_obs = mean[:, self.obs_idx]

        var_transition = np.broadcast_to(
            std**2,
            mean.shape
        )[:, self.obs_idx]

        log_C = self._log_normal_diag(
            np.broadcast_to(
                np.sqrt(a) * self.y,
                mean_obs.shape
            ),
            mean_obs,
            var_transition + (1 - a)
        )

        log_g_parent = self._log_g(
            xp,
            parent_t
        )

        return log_C - log_g_parent

    
    # ---------------------------------------------------------
    # Auxiliary ancestor selection
    # ---------------------------------------------------------

    def logeta(self, k, x):
        """Look-ahead weights used by Chopin's Auxiliary PF."""

        return self._log_omega(k + 1, x)


    # ---------------------------------------------------------
    # Feynman-Kac incremental potential
    # ---------------------------------------------------------

    def logG(self, k, xp, x):
        """Incremental FK weights, accounting for Auxiliary PF correction."""

        if k == 0:
            return np.zeros(x.shape[0])

        return self._log_omega(k, xp)
    

    def M(self, k, xp):
        """Sample particles from the guided backward diffusion transition.

        Arguments:
            k: current SMC iteration
            xp: selected ancestors, shape (N, d)

        Returns:
            New particles, shape (N, d)
        """

        # 1. Get backward diffusion parameters
        _, child_t, mean, std = self._step_info(k, xp)

        # 2. Sample from the ordinary backward transition
        x = np.random.normal(loc=mean, scale=std)

        # 3. Final step: observed pixels are fixed exactly to y
        if child_t == 0:
            x[:, self.obs_idx] = self.y
            return x

        # 4. Gaussian guidance on observed coordinates
        alpha = self.alpha[child_t]

        var_p = np.broadcast_to(std**2, mean.shape)[:, self.obs_idx]
        var_g = 1 - alpha

        # Product of the two Gaussian distributions
        guided_var = (var_p * var_g) / (var_p + var_g)

        guided_mean = (
            var_g * mean[:, self.obs_idx]
            + var_p * np.sqrt(alpha) * self.y
        ) / (var_p + var_g)

        # 5. Replace observed coordinates with guided samples
        x[:, self.obs_idx] = np.random.normal(
            loc=guided_mean,
            scale=np.sqrt(guided_var)
        )

        return x