# ------------------------------------------------------------------------------
# Online Complexity Label Generator
# Replaces ICNet teacher model with annotation-derived task-relevant
# complexity pseudo labels.
#
# C = alpha * C_boundary + beta * C_entropy + gamma * C_rare
#
# Performance-optimized: zero one_hot at native res, zero bincount (GPU sync),
# fused interpolate calls, reuse valid_mask.
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math


class OnlineComplexityLabel:
    """
    Generate pixel-wise complexity labels from segmentation ground truth.

    Three complementary dimensions:
    1. Boundary band: 4-neighbor direct comparison (O(HW), no one_hot)
    2. Local entropy:  downsampled histogram via scatter_add (fully GPU-native)
    3. Rare class:     EMA via scatter_add, index lookup + single downsample

    All ops under torch.no_grad(). Zero CPU-GPU sync points.
    """

    def __init__(self, num_classes, window_size=15, boundary_width=3,
                 alpha=1.0, beta=1.0, gamma=0.5, ignore_label=255,
                 freq_ema_momentum=0.99, entropy_downsample=4):

        self.num_classes = num_classes
        self.window_size = window_size
        self.boundary_width = boundary_width
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.freq_ema_momentum = freq_ema_momentum
        self.entropy_downsample = entropy_downsample

        self.max_entropy = math.log(max(num_classes, 2))

        # Lazily created buffers (device-dependent)
        self._freq_ema = None
        self._ds_labels_buf = None  # reused (B, H//D, W//D) tensor

    def _ensure_buffers(self, device):
        if self._freq_ema is None:
            self._freq_ema = torch.ones(self.num_classes, device=device) / self.num_classes

    # ------------------------------------------------------------------
    # Dimension 1: Boundary Band — 4-neighbor comparison, NO one_hot
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_boundary_complexity(self, labels, valid_mask):
        padded = F.pad(labels.float(), (1, 1, 1, 1), mode='constant', value=-1)

        diff = torch.zeros_like(labels, dtype=torch.float32)
        diff += (labels != padded[:, 2:, 1:-1]).float()   # up
        diff += (labels != padded[:, :-2, 1:-1]).float()  # down
        diff += (labels != padded[:, 1:-1, 2:]).float()   # left
        diff += (labels != padded[:, 1:-1, :-2]).float()  # right

        boundary_map = diff.clamp(0, 1).unsqueeze(1)  # (B, 1, H, W)

        if self.boundary_width > 0:
            ks = 2 * self.boundary_width + 1
            C_b = F.avg_pool2d(boundary_map, kernel_size=ks, stride=1, padding=ks // 2)
        else:
            C_b = boundary_map

        return C_b.clamp(0, 1)

    # ------------------------------------------------------------------
    # Dimension 2: Local Entropy — scatter_add histograms, NO one_hot alloc
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_entropy_complexity(self, labels, valid_mask):
        B = labels.shape[0]
        H, W = int(labels.shape[-2]), int(labels.shape[-1])
        D = self.entropy_downsample
        H_ds, W_ds = H // D, W // D

        # Downsample labels (nearest, lossless for class index)
        labels_ds = F.interpolate(
            labels.float().unsqueeze(1), size=(H_ds, W_ds), mode='nearest'
        ).squeeze(1).long()

        valid_ds = F.interpolate(
            valid_mask.float().unsqueeze(1), size=(H_ds, W_ds), mode='nearest'
        ).squeeze(1) > 0.5

        # --- Sliding-window histogram via avg_pool2d of scatter ---
        # Build one_hot via scatter at DOWNSAMPLED resolution (small: 6 x 19 x 256 x 256)
        labels_clamped = labels_ds.clamp(0, self.num_classes - 1)
        one_hot = torch.zeros(B, self.num_classes, H_ds, W_ds, device=labels.device)
        one_hot.scatter_(1, labels_clamped.unsqueeze(1), 1.0)
        one_hot.mul_(valid_ds.float().unsqueeze(1))

        ws = max(3, self.window_size // D)
        p = F.avg_pool2d(one_hot, kernel_size=ws, stride=1, padding=ws // 2)

        eps = 1e-8
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)

        C_e_ds = (entropy / self.max_entropy).clamp(0, 1)

        C_e = F.interpolate(C_e_ds, size=(H, W), mode='bilinear', align_corners=False)
        return C_e

    # ------------------------------------------------------------------
    # Dimension 3: Rare Class — scatter_add freq (NO bincount), fused downsample
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_rare_class_complexity(self, labels, valid_mask):
        B = labels.shape[0]
        H, W = int(labels.shape[-2]), int(labels.shape[-1])
        D = self.entropy_downsample

        # --- GPU-native class frequency via scatter_add (no CPU sync) ---
        labels_clamped = labels.clamp(0, self.num_classes - 1)
        flat_labels = labels_clamped[valid_mask].long()
        batch_counts = torch.zeros(self.num_classes, device=labels.device).scatter_add_(
            0, flat_labels, torch.ones_like(flat_labels, dtype=torch.float32)
        )
        batch_total = batch_counts.sum()
        if batch_total > 0:
            batch_freq = batch_counts / batch_total
            self._freq_ema = (self.freq_ema_momentum * self._freq_ema +
                              (1 - self.freq_ema_momentum) * batch_freq)

        # Per-class weight
        freq_max = self._freq_ema.max()
        w_c = (1.0 - self._freq_ema / freq_max) if freq_max > 0 else torch.zeros(self.num_classes, device=labels.device)

        # Pixel-level + downsample + maxpool + upsample — fused into one pass
        C_r_per_pixel = w_c[labels_clamped] * valid_mask.float()
        C_r_ds = F.interpolate(
            C_r_per_pixel.unsqueeze(1), size=(H // D, W // D), mode='nearest'
        )
        ws = max(3, self.window_size // D)
        C_r_ds = F.max_pool2d(C_r_ds, kernel_size=ws, stride=1, padding=ws // 2)
        C_r = F.interpolate(C_r_ds, size=(H, W), mode='bilinear', align_corners=False)

        return C_r

    # ------------------------------------------------------------------
    # Main — unified, single-pass valid_mask
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, labels):
        device = labels.device
        self._ensure_buffers(device)

        valid_mask = (labels != self.ignore_label)

        C_b = self.compute_boundary_complexity(labels, valid_mask)
        C_e = self.compute_entropy_complexity(labels, valid_mask)
        C_r = self.compute_rare_class_complexity(labels, valid_mask)

        C = (self.alpha * C_b + self.beta * C_e + self.gamma * C_r)
        C = C / (self.alpha + self.beta + self.gamma + 1e-8)
        C = C * valid_mask.float().unsqueeze(1)

        return C
