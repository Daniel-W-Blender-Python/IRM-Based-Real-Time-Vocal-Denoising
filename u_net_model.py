# u_net_model.py — IRM denoiser architecture, loss function, and training wrapper

import tensorflow as tf
from tensorflow.keras import layers, models

from config import N_FREQ_BINS, CONTEXT_FRAMES, BAND_SIZE, NUM_BANDS, PADDED_FREQ


# ====================================================================
# Architecture
# ====================================================================
def build_irm_model(
    n_freq_bins=N_FREQ_BINS,
    context_frames=CONTEXT_FRAMES,
    band_size=BAND_SIZE,
):
    """
    Band-grouped Dense + frequency-attention U-Net style model.
    ~350k parameters; designed for low cpu-consumption
    """
    num_bands   = (n_freq_bins + band_size - 1) // band_size
    padded_freq = num_bands * band_size
    freq_pad    = padded_freq - n_freq_bins

    inp = layers.Input(shape=(n_freq_bins, context_frames, 2), name="input")

    if freq_pad > 0:
        x = layers.ZeroPadding2D(((0, freq_pad), (0, 0)), name="freq_pad")(inp)
    else:
        x = inp

    x = layers.Reshape((padded_freq, context_frames * 2), name="flatten_tc")(x)

    # Encoder 1 — 48 channels
    e1   = layers.Dense(48, activation="relu", kernel_initializer="he_normal", name="e1")(x)
    e1   = layers.BatchNormalization(momentum=0.9, name="e1_bn")(e1)
    e1   = layers.Dropout(0.1, name="e1_drop")(e1)
    e1_b = layers.Reshape((num_bands, band_size * 48), name="e1_to_bands")(e1)
    e1_b = layers.Dense(band_size * 48, activation="relu", kernel_initializer="he_normal", name="e1_band_mix")(e1_b)
    e1_b = layers.Reshape((padded_freq, 48), name="e1_from_bands")(e1_b)
    e1   = layers.Add(name="e1_res")([e1, e1_b])

    # Encoder 2 — 64 channels
    e2   = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="e2")(e1)
    e2   = layers.BatchNormalization(momentum=0.9, name="e2_bn")(e2)
    e2   = layers.Dropout(0.1, name="e2_drop")(e2)
    e2_b = layers.Reshape((num_bands, band_size * 64), name="e2_to_bands")(e2)
    e2_b = layers.Dense(band_size * 64, activation="relu", kernel_initializer="he_normal", name="e2_band_mix")(e2_b)
    e2_b = layers.Reshape((padded_freq, 64), name="e2_from_bands")(e2_b)
    e2   = layers.Add(name="e2_res")([e2, e2_b])

    # Frequency attention bottleneck
    freq_attn = layers.GlobalAveragePooling1D(keepdims=True, name="freq_gap")(e2)
    freq_attn = layers.Dense(16, activation="relu",    name="attn_down")(freq_attn)
    freq_attn = layers.Dense(64, activation="sigmoid", name="attn_up")(freq_attn)
    b = layers.Multiply(name="attn_gate")([e2, freq_attn])
    b = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="bottleneck")(b)
    b = layers.BatchNormalization(momentum=0.9, name="bottleneck_bn")(b)

    # Decoder
    d1   = layers.Concatenate(name="dec_cat")([b, e1])
    d1   = layers.Dense(48, activation="relu", kernel_initializer="he_normal", name="dec1")(d1)
    d1   = layers.BatchNormalization(momentum=0.9, name="dec1_bn")(d1)
    d1_b = layers.Reshape((num_bands, band_size * 48), name="dec1_to_bands")(d1)
    d1_b = layers.Dense(64, activation="relu", kernel_initializer="he_normal", name="dec1_band_mix")(d1_b)
    d1_b = layers.BatchNormalization(momentum=0.9, name="dec1_band_bn")(d1_b)

    # IRM output — sigmoid per freq bin
    irm_b = layers.Dense(band_size, activation="sigmoid",
                          kernel_initializer="glorot_uniform", name="irm_bands")(d1_b)
    irm   = layers.Reshape((padded_freq,), name="irm_flat")(irm_b)
    if freq_pad > 0:
        irm = layers.Lambda(lambda t: t[:, :n_freq_bins], name="freq_crop")(irm)

    return models.Model(inp, irm, name="irm_denoiser")


# ====================================================================
# Loss
# ====================================================================
def irm_total_loss(irm_true, irm_pred, ync, ycc):
    eps      = 1e-8
    irm_pred = tf.cast(irm_pred, tf.float32)
    irm_true = tf.cast(irm_true, tf.float32)

    mask_mse  = tf.reduce_mean(tf.square(irm_true - irm_pred))

    ync_c     = tf.cast(ync, tf.complex64)
    ycc_c     = tf.cast(ycc, tf.complex64)
    enh_mag   = tf.abs(ync_c) * irm_pred
    clean_mag = tf.abs(ycc_c)

    log_loss = tf.reduce_mean(
        tf.abs(tf.math.log(enh_mag + eps) - tf.math.log(clean_mag + eps))
    )
    mag_l1   = tf.reduce_mean(tf.abs(enh_mag - clean_mag))

    return mask_mse + 0.3 * log_loss + 0.2 * mag_l1


# ====================================================================
# Training wrapper
# ====================================================================
class IRMModel(models.Model):
    def _forward(self, x, tgts, training):
        irm_true, ync, ycc = tgts
        irm_pred = self(x, training=training)
        loss     = irm_total_loss(irm_true, irm_pred, ync, ycc)
        return loss, tf.reduce_mean(irm_pred), tf.reduce_mean(irm_true)

    def train_step(self, data):
        x, tgts = data
        with tf.GradientTape() as tape:
            loss, mm, im = self._forward(x, tgts, True)
        gs, gn = tf.clip_by_global_norm(
            tape.gradient(loss, self.trainable_variables), 0.5
        )
        self.optimizer.apply_gradients(zip(gs, self.trainable_variables))
        return {"loss": loss, "mask_mean": mm, "ideal_mean": im, "grad_norm": gn}

    def test_step(self, data):
        x, tgts = data
        loss, mm, im = self._forward(x, tgts, False)
        return {"loss": loss, "mask_mean": mm, "ideal_mean": im}
