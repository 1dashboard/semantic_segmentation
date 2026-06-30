# ------------------------------------------------------------------------------
# Online Complexity Label Generator
# Replaces ICNet teacher model with annotation-derived task-relevant
# complexity pseudo labels.
#
# C = alpha * C_boundary + beta * C_entropy + gamma * C_rare
#
# Performance: all operations avoid full one_hot at native resolution.
# Boundary uses direct neighbor comparison (O(HW)).
# Entropy uses downsampled one_hot (O(HWC / 16)).
# Rare class uses direct index lookup (O(HW)).
# ------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math


class OnlineComplexityLabel:
    """
    Generate pixel-wise complexity labels from segmentation ground truth.

    Three complementary dimensions:
    1. Boundary band: pixels near class boundaries are harder
    2. Local entropy:  pixels in multi-class windows are harder (Shannon entropy)
    3. Rare class:     pixels of infrequent classes are harder (EMA frequency tracking)

    All computation happens under torch.no_grad(). This module has no trainable
    parameters and only serves as a label generator for the IC loss.
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

        # EMA-tracked global class frequencies (updated per batch)
        self.register_freq_buffer = False

        # Normalization constant for entropy
        self.max_entropy = math.log(max(num_classes, 2))

    def _ensure_freq_buffer(self, device):
        """Lazily create the EMA frequency buffer on the correct device."""
        if not self.register_freq_buffer:
            self.freq_ema = torch.ones(self.num_classes, device=device) / self.num_classes
            self.register_freq_buffer = True

    # ------------------------------------------------------------------
    # Dimension 1: Boundary Band Complexity
    # ------------------------------------------------------------------
    def compute_boundary_complexity(self, labels, valid_mask):
        """
        Efficient boundary detection via direct neighbor comparison.
        A pixel is on a boundary if any 4-neighbor has a different class label.

        Complexity: O(B * H * W) with small constant. No one-hot needed.

        labels: (B, H, W)
        valid_mask: (B, H, W)

        Returns: C_b (B, 1, H, W) in [0, 1]
        """
        # Pad with a value that's not any valid class (use -1)
        padded = F.pad(labels.float(), (1, 1, 1, 1), mode='constant', value=-1)

        # Compare with 4 neighbors: any different → boundary
        diff_up    = (labels != padded[:, 2:, 1:-1]).float()
        diff_down  = (labels != padded[:, :-2, 1:-1]).float()
        diff_left  = (labels != padded[:, 1:-1, 2:]).float()
        diff_right = (labels != padded[:, 1:-1, :-2]).float()

        boundary_map = (diff_up + diff_down + diff_left + diff_right).clamp(0, 1)

        # Soft boundary band via average pooling (approximates Gaussian blur)
        if self.boundary_width > 0:
            ks = 2 * self.boundary_width + 1
            C_b = F.avg_pool2d(
                boundary_map.unsqueeze(1), kernel_size=ks, stride=1, padding=ks // 2
            )
        else:
            C_b = boundary_map.unsqueeze(1)

        return C_b.clamp(0, 1)

    # ------------------------------------------------------------------
    # Dimension 2: Local Category Entropy
    # ------------------------------------------------------------------
    def compute_entropy_complexity(self, labels, valid_mask):
        """
        Shannon entropy via downsampled one_hot + avg_pool for efficiency.

        Downsampling by factor D reduces one_hot memory from O(BCHW) to
        O(BCHW/D^2), roughly 16x savings at D=4.

        labels: (B, H, W)
        valid_mask: (B, H, W)

        Returns: C_e (B, 1, H, W) in [0, 1]
        """
        B = labels.shape[0]
        H, W = labels.shape[-2], labels.shape[-1]
        # Ensure plain python ints for interpolate
        H, W = int(H), int(W)
        D = self.entropy_downsample
        H_ds, W_ds = H // D, W // D
        labels_ds = F.interpolate(
            labels.float().unsqueeze(1), size=(H_ds, W_ds),
            mode='nearest'
        ).squeeze(1).long()

        # Valid mask at downsampled resolution
        valid_ds = F.interpolate(
            valid_mask.float().unsqueeze(1), size=(H_ds, W_ds),
            mode='nearest'
        ).squeeze(1) > 0.5

        # --- One-hot at downsampled resolution ---
        labels_clamped = labels_ds.clamp(0, self.num_classes - 1)
        one_hot = torch.zeros(B, self.num_classes, H_ds, W_ds,
                              device=labels.device)
        one_hot.scatter_(1, labels_clamped.unsqueeze(1), 1.0)
        one_hot = one_hot * valid_ds.float().unsqueeze(1)

        # --- Sliding-window class proportions (window scaled to downsample) ---
        ws = max(3, self.window_size // D)  # scale window to downsampled grid
        p = F.avg_pool2d(one_hot, kernel_size=ws, stride=1, padding=ws // 2)

        # --- Per-pixel entropy ---
        eps = 1e-8
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)  # (B, 1, H_ds, W_ds)

        C_e_ds = (entropy / self.max_entropy).clamp(0, 1)

        # --- Upsample back to original resolution ---
        C_e = F.interpolate(C_e_ds, size=(H, W), mode='bilinear',
                            align_corners=False)

        return C_e

    # ------------------------------------------------------------------
    # Dimension 3: Rare Class Weight
    # ------------------------------------------------------------------
    def compute_rare_class_complexity(self, labels, valid_mask):
        """
        Pixels of rare classes are harder. Weights from EMA class frequency.
        Rare classes radiate difficulty into their local neighborhood.

        Complexity: O(B * H * W) via direct index lookup.

        labels: (B, H, W)
        valid_mask: (B, H, W)

        Returns: C_r (B, 1, H, W) in [0, 1]
        """
        B = labels.shape[0]
        H, W = int(labels.shape[-2]), int(labels.shape[-1])
        device = labels.device

        # --- Update EMA class frequency ---
        valid_labels = labels[valid_mask].clamp(0, self.num_classes - 1)
        batch_counts = torch.bincount(valid_labels, minlength=self.num_classes).float()
        batch_total = batch_counts.sum()
        if batch_total > 0:
            batch_freq = batch_counts / batch_total
            self.freq_ema = (self.freq_ema_momentum * self.freq_ema +
                             (1 - self.freq_ema_momentum) * batch_freq)

        # --- Per-class rare weight ---
        freq_max = self.freq_ema.max()
        if freq_max > 0:
            w_c = (1.0 - self.freq_ema / freq_max)  # (C,) in [0, 1)
        else:
            w_c = torch.zeros(self.num_classes, device=device)

        # --- Pixel-level assignment via index lookup ---
        labels_clamped = labels.clamp(0, self.num_classes - 1)
        C_r_per_pixel = w_c[labels_clamped]  # (B, H, W)
        C_r_per_pixel = C_r_per_pixel * valid_mask.float()

        # --- Local max-pool radiation (downsampled for efficiency) ---
        D = self.entropy_downsample
        H_ds, W_ds = H // D, W // D
        C_r_ds = F.interpolate(
            C_r_per_pixel.unsqueeze(1), size=(H_ds, W_ds), mode='nearest'
        )
        ws = max(3, self.window_size // D)
        C_r_ds = F.max_pool2d(C_r_ds, kernel_size=ws, stride=1, padding=ws // 2)
        C_r = F.interpolate(C_r_ds, size=(H, W), mode='bilinear', align_corners=False)

        return C_r

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, labels):
        """
        Generate online complexity pseudo label from segmentation annotations.

        Args:
            labels: (B, H, W) tensor, values in [0, num_classes-1],
                    with ignore_label=255

        Returns:
            C: (B, 1, H, W) tensor, values in [0, 1]
        """
        device = labels.device
        self._ensure_freq_buffer(device)

        valid_mask = (labels != self.ignore_label)

        C_b = self.compute_boundary_complexity(labels, valid_mask)
        C_e = self.compute_entropy_complexity(labels, valid_mask)
        C_r = self.compute_rare_class_complexity(labels, valid_mask)

        # Weighted fusion
        C = (self.alpha * C_b + self.beta * C_e + self.gamma * C_r)
        C = C / (self.alpha + self.beta + self.gamma + 1e-8)

        # Mask ignore regions
        C = C * valid_mask.float().unsqueeze(1)

        return C
