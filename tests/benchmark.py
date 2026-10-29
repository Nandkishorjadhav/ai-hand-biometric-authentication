import sys
import os

# Project root path setup
PROJECT_ROOT = 'd:/Hand Biometri Authentication/hand-biometric-auth'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for automated/CLI execution
import matplotlib.pyplot as plt

from core.normalizer import LandmarkNormalizer
from core.matcher import BiometricMatcher
from core.database import Database


class SyntheticHandGenerator:
    """
    Generates realistic 3D hand landmark topologies and composite 71-D biometric vectors.
    Simulates morphological biometric variation across users and intra-session capture noise.
    """

    # Canonical 21 MediaPipe hand landmarks (wrist at base, open palm facing camera)
    CANONICAL_HAND = np.array([
        [0.500, 0.820,  0.000],  # 0: Wrist
        [0.440, 0.740, -0.010],  # 1: Thumb CMC
        [0.390, 0.660, -0.018],  # 2: Thumb MCP
        [0.345, 0.580, -0.022],  # 3: Thumb IP
        [0.300, 0.510, -0.025],  # 4: Thumb Tip
        [0.430, 0.540, -0.008],  # 5: Index MCP
        [0.410, 0.430, -0.012],  # 6: Index PIP
        [0.398, 0.350, -0.015],  # 7: Index DIP
        [0.388, 0.270, -0.018],  # 8: Index Tip
        [0.500, 0.520,  0.000],  # 9: Middle MCP
        [0.500, 0.390, -0.005],  # 10: Middle PIP
        [0.500, 0.300, -0.010],  # 11: Middle DIP
        [0.500, 0.200, -0.012],  # 12: Middle Tip
        [0.570, 0.540, -0.008],  # 13: Ring MCP
        [0.585, 0.430, -0.012],  # 14: Ring PIP
        [0.595, 0.350, -0.015],  # 15: Ring DIP
        [0.605, 0.270, -0.018],  # 16: Ring Tip
        [0.635, 0.580, -0.012],  # 17: Pinky MCP
        [0.660, 0.490, -0.016],  # 18: Pinky PIP
        [0.675, 0.420, -0.018],  # 19: Pinky DIP
        [0.690, 0.350, -0.020],  # 20: Pinky Tip
    ], dtype=np.float64)

    FINGER_JOINTS = {
        "thumb": (1, [2, 3, 4]),
        "index": (5, [6, 7, 8]),
        "middle": (9, [10, 11, 12]),
        "ring": (13, [14, 15, 16]),
        "pinky": (17, [18, 19, 20]),
    }

    @classmethod
    def create_user_profile(cls, user_id: int, rng: np.random.Generator) -> dict:
        """
        Creates a distinctive hand morphological profile for a user.
        Defines finger length proportions, finger splay angles, and palm scale.
        """
        return {
            "user_id": user_id,
            "name": f"User_{user_id:02d}",
            "palm_scale": float(rng.uniform(0.90, 1.12)),
            "finger_lengths": {
                "thumb": float(rng.uniform(0.85, 1.15)),
                "index": float(rng.uniform(0.88, 1.12)),
                "middle": float(rng.uniform(0.90, 1.10)),
                "ring": float(rng.uniform(0.88, 1.12)),
                "pinky": float(rng.uniform(0.85, 1.15)),
            },
            "finger_spreads": {
                "thumb": float(rng.uniform(-0.06, 0.06)),
                "index": float(rng.uniform(-0.05, 0.05)),
                "middle": float(rng.uniform(-0.02, 0.02)),
                "ring": float(rng.uniform(-0.05, 0.05)),
                "pinky": float(rng.uniform(-0.06, 0.06)),
            },
            "depth_factor": float(rng.uniform(0.85, 1.15)),
        }

    @classmethod
    def generate_landmarks(cls, profile: dict, rng: np.random.Generator, add_noise: bool = True) -> list:
        """
        Generates 21 3D landmarks for a user profile, incorporating anatomical morphology,
        natural micro-tremor noise, and small random camera presentation variations (rotation/scale/translation).
        """
        pts = cls.CANONICAL_HAND.copy()
        wrist = pts[0].copy()

        # Step 1: Apply user palm scale relative to wrist
        for i in range(1, 21):
            pts[i] = wrist + (pts[i] - wrist) * profile["palm_scale"]

        # Step 2: Apply user-specific finger length scaling and spread angles
        for finger, (mcp_idx, joint_indices) in cls.FINGER_JOINTS.items():
            mcp = pts[mcp_idx].copy()
            length_factor = profile["finger_lengths"][finger]
            spread_angle = profile["finger_spreads"][finger]

            cos_s = math.cos(spread_angle)
            sin_s = math.sin(spread_angle)
            rot_2d = np.array([[cos_s, -sin_s], [sin_s, cos_s]])

            for j_idx in joint_indices:
                diff = pts[j_idx] - mcp
                # Scale finger bone length
                diff[:2] = diff[:2] * length_factor
                # Apply finger splay/spread rotation
                diff[:2] = np.dot(rot_2d, diff[:2])
                pts[j_idx] = mcp + diff

        # Step 3: Apply user depth factor
        pts[:, 2] = pts[:, 2] * profile["depth_factor"]

        # Step 4: Add sensor noise and natural hand tremor (micro-tremor: sigma ~ 0.004)
        if add_noise:
            jitter = rng.normal(0.0, 0.004, size=pts.shape)
            pts += jitter

            # Random camera presentation variation: small translation, tilt, and scale
            t_x, t_y = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
            tilt_angle = rng.uniform(-math.radians(8), math.radians(8))
            cos_t, sin_t = math.cos(tilt_angle), math.sin(tilt_angle)
            rot_tilt = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            scale_var = rng.uniform(0.97, 1.03)

            pts[:, :2] = np.dot(pts[:, :2] - wrist[:2], rot_tilt.T) * scale_var + wrist[:2]
            pts[:, 0] += t_x
            pts[:, 1] += t_y

        # Convert to list of tuples for LandmarkNormalizer
        return [(float(p[0]), float(p[1]), float(p[2])) for p in pts]

    @classmethod
    def generate_biometric_vector(cls, profile: dict, rng: np.random.Generator) -> np.ndarray:
        """Generates a complete 71-D normalized invariant biometric feature vector."""
        lms = cls.generate_landmarks(profile, rng, add_noise=True)
        return LandmarkNormalizer.get_composite_biometric_vector(lms)


class BiometricBenchmark:
    """
    Executes a biometric verification benchmark:
    - N=10 synthetic users enrolled with 10 samples each
    - M=50 authentication attempts per user (25 genuine + 25 impostor = 50 attempts/user)
    - Computes FAR, FRR, EER, AUC, Decidability Index (d')
    - Generates ROC curve and Confusion Matrix plots
    """

    def __init__(self, num_users: int = 10, enrollment_samples: int = 10, attempts_per_user: int = 50, seed: int = 42):
        self.num_users = num_users
        self.enrollment_samples = enrollment_samples
        self.attempts_per_user = attempts_per_user
        self.genuine_attempts_per_user = attempts_per_user // 2
        self.impostor_attempts_per_user = attempts_per_user - self.genuine_attempts_per_user
        self.rng = np.random.default_rng(seed)

        self.user_profiles = []
        self.enrolled_templates = {}  # user_id -> master_vector
        self.adaptive_thresholds = {}  # user_id -> threshold

        self.genuine_scores = []
        self.impostor_scores = []
        self.inter_user_similarity_matrix = np.zeros((num_users, num_users))
        self.confusion_matrix = {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        self.results = {}

    def setup_and_enroll(self):
        """Creates user profiles and enrolls master templates with adaptive thresholds."""
        self.user_profiles = [
            SyntheticHandGenerator.create_user_profile(i + 1, self.rng)
            for i in range(self.num_users)
        ]

        for profile in self.user_profiles:
            uid = profile["user_id"]
            samples = [
                SyntheticHandGenerator.generate_biometric_vector(profile, self.rng)
                for _ in range(self.enrollment_samples)
            ]
            master_vec = BiometricMatcher.create_master_template(samples)
            ad_thresh = BiometricMatcher.compute_adaptive_threshold(samples)
            self.enrolled_templates[uid] = master_vec
            self.adaptive_thresholds[uid] = ad_thresh

    def run_trials(self):
        """Runs genuine and impostor authentication attempts across all enrolled users."""
        self.genuine_scores = []
        self.impostor_scores = []
        default_thresh = BiometricMatcher.DEFAULT_THRESHOLD  # 0.90

        # Run Genuine Trials (Query user matches own enrolled template)
        for profile in self.user_profiles:
            uid = profile["user_id"]
            master_vec = self.enrolled_templates[uid]

            for _ in range(self.genuine_attempts_per_user):
                query_vec = SyntheticHandGenerator.generate_biometric_vector(profile, self.rng)
                sim = BiometricMatcher.compute_cosine_similarity(query_vec, master_vec)
                self.genuine_scores.append(sim)

                if sim >= default_thresh:
                    self.confusion_matrix["TP"] += 1
                else:
                    self.confusion_matrix["FN"] += 1

        # Run Impostor Trials (Unenrolled/other users attempt authentication as user)
        for i, profile in enumerate(self.user_profiles):
            target_uid = profile["user_id"]
            target_master = self.enrolled_templates[target_uid]

            # Choose impostor profiles different from target user
            other_profiles = [p for p in self.user_profiles if p["user_id"] != target_uid]

            for k in range(self.impostor_attempts_per_user):
                impostor_profile = other_profiles[k % len(other_profiles)]
                impostor_query = SyntheticHandGenerator.generate_biometric_vector(impostor_profile, self.rng)
                sim = BiometricMatcher.compute_cosine_similarity(impostor_query, target_master)
                self.impostor_scores.append(sim)

                if sim >= default_thresh:
                    self.confusion_matrix["FP"] += 1
                else:
                    self.confusion_matrix["TN"] += 1

        # Compute 10x10 Inter-user Master Template Cosine Similarity Matrix
        for i in range(self.num_users):
            vec_i = self.enrolled_templates[self.user_profiles[i]["user_id"]]
            for j in range(self.num_users):
                vec_j = self.enrolled_templates[self.user_profiles[j]["user_id"]]
                self.inter_user_similarity_matrix[i, j] = BiometricMatcher.compute_cosine_similarity(vec_i, vec_j)

    def compute_metrics(self) -> dict:
        """
        Computes biometric operating characteristics:
        FAR, FRR, EER, AUC, Decidability Index d', and Zero-FAR / Zero-FRR points.
        """
        gen = np.array(self.genuine_scores, dtype=np.float64)
        imp = np.array(self.impostor_scores, dtype=np.float64)

        num_gen = len(gen)
        num_imp = len(imp)

        # Threshold sweep across [0.0, 1.0]
        thresholds = np.linspace(0.0, 1.0, 2001)
        far_list = []
        frr_list = []
        tar_list = []

        for th in thresholds:
            # FAR: Impostors accepted (score >= th)
            far = float(np.sum(imp >= th)) / num_imp
            # FRR: Genuines rejected (score < th)
            frr = float(np.sum(gen < th)) / num_gen
            far_list.append(far)
            frr_list.append(frr)
            tar_list.append(1.0 - frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)
        tar_arr = np.array(tar_list)

        # Find Equal Error Rate (EER) where FAR ~= FRR
        diff = np.abs(far_arr - frr_arr)
        eer_idx = int(np.argmin(diff))
        eer_threshold = float(thresholds[eer_idx])
        eer_val = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)

        # Area Under ROC Curve (AUC) using trapezoidal rule
        # Sort by FAR for clean integration
        sort_order = np.argsort(far_arr)
        auc = float(np.abs(np.trapz(tar_arr[sort_order], far_arr[sort_order])))

        # Default operating threshold (0.90) metrics
        def_th = 0.90
        def_far = float(np.sum(imp >= def_th)) / num_imp
        def_frr = float(np.sum(gen < def_th)) / num_gen
        def_tar = 1.0 - def_frr
        def_acc = float(np.sum(gen >= def_th) + np.sum(imp < def_th)) / (num_gen + num_imp)

        # Decidability Index d' (Daugman metric)
        mu_gen, sigma_gen = float(np.mean(gen)), float(np.std(gen))
        mu_imp, sigma_imp = float(np.mean(imp)), float(np.std(imp))
        pooled_variance = 0.5 * (sigma_gen ** 2 + sigma_imp ** 2)
        d_prime = (abs(mu_gen - mu_imp) / math.sqrt(pooled_variance)) if pooled_variance > 1e-9 else 0.0

        # Zero-FAR (smallest threshold where FAR = 0)
        zero_far_indices = np.where(far_arr == 0.0)[0]
        zero_far_th = float(thresholds[zero_far_indices[0]]) if len(zero_far_indices) > 0 else 1.0
        zero_far_frr = float(frr_arr[zero_far_indices[0]]) if len(zero_far_indices) > 0 else 1.0

        # Zero-FRR (largest threshold where FRR = 0)
        zero_frr_indices = np.where(frr_arr == 0.0)[0]
        zero_frr_th = float(thresholds[zero_frr_indices[-1]]) if len(zero_frr_indices) > 0 else 0.0
        zero_frr_far = float(far_arr[zero_frr_indices[-1]]) if len(zero_frr_indices) > 0 else 1.0

        self.results = {
            "num_users": self.num_users,
            "enrollment_samples": self.enrollment_samples,
            "total_genuine_trials": num_gen,
            "total_impostor_trials": num_imp,
            "mu_gen": mu_gen,
            "sigma_gen": sigma_gen,
            "median_gen": float(np.median(gen)),
            "min_gen": float(np.min(gen)),
            "max_gen": float(np.max(gen)),
            "mu_imp": mu_imp,
            "sigma_imp": sigma_imp,
            "median_imp": float(np.median(imp)),
            "min_imp": float(np.min(imp)),
            "max_imp": float(np.max(imp)),
            "d_prime": d_prime,
            "eer": eer_val,
            "eer_threshold": eer_threshold,
            "auc": auc,
            "default_threshold": def_th,
            "default_far": def_far,
            "default_frr": def_frr,
            "default_tar": def_tar,
            "default_accuracy": def_acc,
            "zero_far_threshold": zero_far_th,
            "zero_far_frr": zero_far_frr,
            "zero_frr_threshold": zero_frr_th,
            "zero_frr_far": zero_frr_far,
            "thresholds": thresholds,
            "far_arr": far_arr,
            "frr_arr": frr_arr,
            "tar_arr": tar_arr,
        }
        return self.results

    def plot_roc_curve(self, save_path: str):
        """Generates publication-quality ROC curve and Error-vs-Threshold plots."""
        res = self.results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Hand Biometric Verification Performance: ROC & DET Curves", fontsize=15, fontweight='bold', y=0.98)

        # --- Subplot 1: ROC Curve (FAR vs TAR) ---
        ax1.plot(res["far_arr"] * 100, res["tar_arr"] * 100, color="#1e40af", lw=2.5, label=f"Hand Biometrics (AUC = {res['auc']:.4f})")
        ax1.plot([0, 100], [0, 100], color="#94a3b8", linestyle="--", lw=1.2, label="Random Chance (AUC = 0.50)")

        # Operating points
        eer_far = res["eer"] * 100
        eer_tar = (1.0 - res["eer"]) * 100
        ax1.scatter([eer_far], [eer_tar], color="#16a34a", s=90, zorder=5, label=f"EER Point ({res['eer']*100:.2f}%)")

        def_far_pct = res["default_far"] * 100
        def_tar_pct = res["default_tar"] * 100
        ax1.scatter([def_far_pct], [def_tar_pct], color="#dc2626", marker="*", s=160, zorder=6, label=f"System Default th=0.90 (FAR={def_far_pct:.2f}%, TAR={def_tar_pct:.2f}%)")

        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight='bold', pad=10)
        ax1.set_xlabel("False Acceptance Rate (FAR) [%]", fontsize=11, fontweight='semibold')
        ax1.set_ylabel("True Acceptance Rate (TAR) [%]", fontsize=11, fontweight='semibold')
        ax1.set_xlim([-1, 30])
        ax1.set_ylim([70, 102])
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="lower right", frameon=True, fontsize=9)

        # --- Subplot 2: FAR & FRR vs Cosine Similarity Threshold ---
        th_pct = res["thresholds"]
        ax2.plot(th_pct, res["far_arr"] * 100, color="#dc2626", lw=2.2, label="FAR (False Accept Rate)")
        ax2.plot(th_pct, res["frr_arr"] * 100, color="#2563eb", lw=2.2, label="FRR (False Reject Rate)")

        # Mark EER crossover
        ax2.axvline(x=res["eer_threshold"], color="#16a34a", linestyle="--", lw=1.5, alpha=0.9, label=f"EER Threshold ({res['eer_threshold']:.3f})")
        ax2.axvline(x=res["default_threshold"], color="#f59e0b", linestyle=":", lw=1.8, label=f"System Threshold (0.90)")

        ax2.scatter([res["eer_threshold"]], [res["eer"] * 100], color="#16a34a", s=80, zorder=5)
        ax2.annotate(
            f"EER = {res['eer']*100:.2f}%\n@ th={res['eer_threshold']:.3f}",
            xy=(res["eer_threshold"], res["eer"] * 100),
            xytext=(res["eer_threshold"] - 0.08, res["eer"] * 100 + 12),
            arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
            fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc="#dcfce7", ec="#16a34a", lw=1.2)
        )

        ax2.set_title("Error Rates vs Decision Threshold", fontsize=13, fontweight='bold', pad=10)
        ax2.set_xlabel("Cosine Similarity Threshold", fontsize=11, fontweight='semibold')
        ax2.set_ylabel("Error Rate [%]", fontsize=11, fontweight='semibold')
        ax2.set_xlim([0.70, 1.00])
        ax2.set_ylim([-1, 50])
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper center", frameon=True, fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def plot_confusion_matrix(self, save_path: str):
        """Generates verification confusion matrix and inter-user similarity heatmaps."""
        cm = self.confusion_matrix
        tp, fn = cm["TP"], cm["FN"]
        fp, tn = cm["FP"], cm["TN"]
        total_gen = tp + fn
        total_imp = fp + tn

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Biometric Decision Confusion Matrix & User Separation Matrix", fontsize=15, fontweight='bold', y=0.98)

        # --- Subplot 1: 2x2 Binary Verification Confusion Matrix ---
        mat = np.array([[tp, fn], [fp, tn]])
        percentages = np.array([
            [tp / total_gen * 100 if total_gen else 0, fn / total_gen * 100 if total_gen else 0],
            [fp / total_imp * 100 if total_imp else 0, tn / total_imp * 100 if total_imp else 0]
        ])

        im1 = ax1.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(["Predicted\nACCESS GRANTED", "Predicted\nACCESS DENIED"], fontsize=10, fontweight='semibold')
        ax1.set_yticklabels(["Actual\nGENUINE", "Actual\nIMPOSTOR"], fontsize=10, fontweight='semibold')

        labels = [
            [f"True Accept (TP)\n{tp} / {total_gen}\n({percentages[0, 0]:.1f}%)",
             f"False Reject (FN)\n{fn} / {total_gen}\n({percentages[0, 1]:.1f}%)"],
            [f"False Accept (FP)\n{fp} / {total_imp}\n({percentages[1, 0]:.1f}%)",
             f"True Reject (TN)\n{tn} / {total_imp}\n({percentages[1, 1]:.1f}%)"]
        ]

        for i in range(2):
            for j in range(2):
                color = "white" if percentages[i, j] > 50 else "black"
                ax1.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=10.5, fontweight='bold')

        ax1.set_title(f"Authentication Decision Matrix (th = {self.results['default_threshold']:.2f})", fontsize=12, fontweight='bold', pad=10)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Rate (%)")

        # --- Subplot 2: 10x10 Inter-User Template Similarity Heatmap ---
        sim_mat = self.inter_user_similarity_matrix
        im2 = ax2.imshow(sim_mat, cmap="viridis", vmin=0.70, vmax=1.0)
        user_labels = [f"U{i+1}" for i in range(self.num_users)]
        ax2.set_xticks(range(self.num_users))
        ax2.set_yticks(range(self.num_users))
        ax2.set_xticklabels(user_labels, fontsize=9)
        ax2.set_yticklabels(user_labels, fontsize=9)

        # Overlay text on diagonal
        for i in range(self.num_users):
            ax2.text(i, i, f"{sim_mat[i, i]:.2f}", ha="center", va="center", color="black", fontsize=7.5, fontweight='bold')

        ax2.set_title("Master Template Cross-Similarity (10 Users)", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Enrolled User", fontsize=10, fontweight='semibold')
        ax2.set_ylabel("Enrolled User", fontsize=10, fontweight='semibold')
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Cosine Similarity")

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def generate_report(self, roc_path: str, cm_path: str) -> str:
        """Constructs an academic benchmark text report."""
        r = self.results
        report = f"""
================================================================================
          BIOMETRIC SYSTEM BENCHMARK REPORT: CONTACTLESS HAND BIOMETRICS
================================================================================
Enrolled Users:                     {r['num_users']}
Enrollment Samples Per User:        {r['enrollment_samples']} (Total templates: {r['num_users'] * r['enrollment_samples']})
Authentication Attempts Per User:   {self.attempts_per_user} ({self.genuine_attempts_per_user} Genuine + {self.impostor_attempts_per_user} Impostor)
Total Evaluation Trials:            {r['total_genuine_trials'] + r['total_impostor_trials']} ({r['total_genuine_trials']} Genuine + {r['total_impostor_trials']} Impostor)
Feature Vector Dimension:           71-D (63 Invariant Landmark Coords + 8 Morphological Ratios)

--------------------------------------------------------------------------------
SIMILARITY SCORE DISTRIBUTIONS (COSINE SIMILARITY)
--------------------------------------------------------------------------------
Metric                  Genuine Attempts        Impostor Attempts
--------------------------------------------------------------------------------
Mean Similarity:        {r['mu_gen']:.4f}                  {r['mu_imp']:.4f}
Std Deviation:          {r['sigma_gen']:.4f}                  {r['sigma_imp']:.4f}
Median Similarity:      {r['median_gen']:.4f}                  {r['median_imp']:.4f}
Min Similarity:         {r['min_gen']:.4f}                  {r['min_imp']:.4f}
Max Similarity:         {r['max_gen']:.4f}                  {r['max_imp']:.4f}
Decidability Index (d'): {r['d_prime']:.4f} (Separability: High statistical distinction)

--------------------------------------------------------------------------------
BIOMETRIC ACCURACY METRICS & OPERATING CHARACTERISTICS
--------------------------------------------------------------------------------
Equal Error Rate (EER):                 {r['eer'] * 100:.2f}%  at Threshold = {r['eer_threshold']:.4f}
Area Under ROC Curve (AUC):             {r['auc']:.4f}

Performance at System Default Threshold ({r['default_threshold']:.4f}):
  - False Acceptance Rate (FAR):        {r['default_far'] * 100:.2f}%  ({self.confusion_matrix['FP']} / {r['total_impostor_trials']} trials)
  - False Rejection Rate (FRR):         {r['default_frr'] * 100:.2f}%  ({self.confusion_matrix['FN']} / {r['total_genuine_trials']} trials)
  - True Acceptance Rate (TAR):         {r['default_tar'] * 100:.2f}%
  - Overall Classification Accuracy:    {r['default_accuracy'] * 100:.2f}%

Zero-FAR Operating Point (Maximum Security):
  - Threshold for FAR = 0.00%:          {r['zero_far_threshold']:.4f}
  - Corresponding FRR:                  {r['zero_far_frr'] * 100:.2f}%

Zero-FRR Operating Point (Maximum Convenience):
  - Threshold for FRR = 0.00%:          {r['zero_frr_threshold']:.4f}
  - Corresponding FAR:                  {r['zero_frr_far'] * 100:.2f}%

--------------------------------------------------------------------------------
GENERATED PLOT ARTIFACTS
--------------------------------------------------------------------------------
[✓] ROC Curve Plot:            {roc_path}
[✓] Confusion Matrix Plot:     {cm_path}
================================================================================
"""
        return report


class TestBiometricBenchmark(unittest.TestCase):
    """Unittest test case ensuring benchmark integrity and reproducibility."""

    @classmethod
    def setUpClass(cls):
        cls.tests_dir = os.path.dirname(os.path.abspath(__file__))
        cls.roc_path = os.path.join(cls.tests_dir, "roc_curve.png")
        cls.cm_path = os.path.join(cls.tests_dir, "confusion_matrix.png")

        cls.benchmark = BiometricBenchmark(num_users=10, enrollment_samples=10, attempts_per_user=50, seed=42)
        cls.benchmark.setup_and_enroll()
        cls.benchmark.run_trials()
        cls.metrics = cls.benchmark.compute_metrics()
        cls.benchmark.plot_roc_curve(cls.roc_path)
        cls.benchmark.plot_confusion_matrix(cls.cm_path)

    def test_01_synthetic_vector_dimensions_and_normalization(self):
        """Validates that synthetic 71-D biometric vectors are unit L2-normalized."""
        profile = SyntheticHandGenerator.create_user_profile(1, np.random.default_rng(100))
        vec = SyntheticHandGenerator.generate_biometric_vector(profile, np.random.default_rng(101))
        self.assertEqual(len(vec), 71)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=4)
        self.assertFalse(np.isnan(vec).any())
        self.assertFalse(np.isinf(vec).any())

    def test_02_biometric_separability(self):
        """Ensures genuine similarity is significantly higher than impostor similarity."""
        m = self.metrics
        self.assertGreater(m["mu_gen"], m["mu_imp"])
        self.assertGreater(m["mu_gen"], 0.92, "Mean genuine similarity should exceed 0.92")
        self.assertLess(m["mu_imp"], 0.88, "Mean impostor similarity should remain below 0.88")
        self.assertGreater(m["d_prime"], 2.0, "Decidability index d' must show robust distribution separation")

    def test_03_eer_and_auc_bounds(self):
        """Validates that EER is below 5% and AUC is above 0.98."""
        m = self.metrics
        self.assertLessEqual(m["eer"], 0.05, f"EER {m['eer']*100:.2f}% exceeds acceptable 5% bound")
        self.assertGreaterEqual(m["auc"], 0.98, f"AUC {m['auc']:.4f} is below acceptable 0.98 bound")

    def test_04_default_operating_accuracy(self):
        """Verifies accuracy at standard 0.90 threshold."""
        m = self.metrics
        self.assertGreaterEqual(m["default_accuracy"], 0.95, "Classification accuracy at 0.90 must be >= 95%")
        self.assertLessEqual(m["default_far"], 0.05, "FAR at 0.90 must be <= 5%")
        self.assertLessEqual(m["default_frr"], 0.05, "FRR at 0.90 must be <= 5%")

    def test_05_plot_files_created(self):
        """Verifies that ROC and confusion matrix plot image files exist and are not empty."""
        self.assertTrue(os.path.exists(self.roc_path), "ROC curve file was not created")
        self.assertGreater(os.path.getsize(self.roc_path), 10000, "ROC curve file is unusually small")

        self.assertTrue(os.path.exists(self.cm_path), "Confusion matrix file was not created")
        self.assertGreater(os.path.getsize(self.cm_path), 10000, "Confusion matrix file is unusually small")


def run_standalone_benchmark():
    """Executes the benchmark and prints the report when called directly."""
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    roc_path = os.path.join(tests_dir, "roc_curve.png")
    cm_path = os.path.join(tests_dir, "confusion_matrix.png")

    print("\n[BENCHMARK] Initializing Biometric Verification Evaluation Suite...")
    bm = BiometricBenchmark(num_users=10, enrollment_samples=10, attempts_per_user=50, seed=42)

    print("[BENCHMARK] Enrolling 10 synthetic users (10 samples each)...")
    bm.setup_and_enroll()

    print("[BENCHMARK] Executing 500 authentication trials (250 genuine + 250 impostor)...")
    bm.run_trials()

    print("[BENCHMARK] Computing FAR, FRR, EER, and statistical metrics...")
    bm.compute_metrics()

    print(f"[BENCHMARK] Rendering ROC curve to: {roc_path}")
    bm.plot_roc_curve(roc_path)

    print(f"[BENCHMARK] Rendering Confusion Matrix to: {cm_path}")
    bm.plot_confusion_matrix(cm_path)

    report = bm.generate_report(roc_path, cm_path)
    print(report)


if __name__ == "__main__":
    run_standalone_benchmark()
    unittest.main()
