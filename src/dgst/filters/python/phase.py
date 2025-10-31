import numpy as np
import math

def phase_congruency(image: np.ndarray,
                     nscale: int = 4,
                     norient: int = 6,
                     min_wavelength: float = 3.0,
                     mult: float = 2.1,
                     sigma_onf: float = 0.55,
                     k: float = 2.0,
                     cut_off: float = 0.5,
                     g: float = 10.0,
                     eps: float = 1e-8) -> np.ndarray:
    """Compute a phase congruency map for a grayscale image.

    This is an approximate implementation inspired by the Kovesi phase
    congruency algorithm. It builds log-Gabor band-pass filters across
    multiple scales and orientations in the frequency domain, computes
    quadrature responses via inverse FFT, forms local energy, and
    normalizes to produce a phase congruency map in range [0, 1].

    Args:
        image: 2D uint8 or float image.
        nscale: Number of scales.
        norient: Number of orientations.
        min_wavelength: Wavelength of smallest scale filter.
        mult: Scaling factor between successive filters.
        sigma_onf: Ratio of the bandwidth of the log-Gabor filter.
        k, cut_off, g: Parameters for noise compensation / weighting (see Kovesi).
        eps: Small constant to avoid division by zero.

    Returns:
        phase_cong: float32 array same shape as input with values ~[0,1].
    """
    if image.ndim != 2:
        raise ValueError("phase_congruency expects a 2D grayscale image")

    # Convert to float
    img = image.astype(np.float32)
    rows, cols = img.shape

    # Frequency grids
    y = np.arange(-rows//2, rows - rows//2)
    x = np.arange(-cols//2, cols - cols//2)
    xv, yv = np.meshgrid(x, y)
    # normalized frequency radius
    radius = np.sqrt((xv.astype(np.float32) / cols) ** 2 + (yv.astype(np.float32) / rows) ** 2)
    # avoid log(0)
    radius[rows//2, cols//2] = 1.0
    theta = np.arctan2(yv, xv)

    # prepare FFT of image
    imagefft = np.fft.fftshift(np.fft.fft2(img))

    pc_sum = np.zeros_like(img, dtype=np.float32)
    amplitude_sum = np.zeros_like(img, dtype=np.float32)

    for o in range(norient):
        angl = o * math.pi / norient
        # angular spread
        ds = np.sin(theta) * math.cos(angl) - np.cos(theta) * math.sin(angl)
        dtheta = np.abs(np.arctan2(ds, np.cos(theta) * math.cos(angl) + np.sin(theta) * math.sin(angl)))
        # spread function (raised cosine-like)
        angl_spread = np.exp((-dtheta ** 2) / (2 * ( (math.pi / norient * 1.2) ** 2)))

        # For this orientation, accumulate even and odd responses across scales
        sum_even = np.zeros_like(img, dtype=np.float32)
        sum_odd = np.zeros_like(img, dtype=np.float32)
        an = np.zeros_like(img, dtype=np.float32)  # amplitude sum per orientation

        wavelength = min_wavelength
        for s in range(nscale):
            fo = 1.0 / wavelength
            log_rad = np.log(radius / fo)
            radial = np.exp(-(log_rad ** 2) / (2 * (np.log(sigma_onf) ** 2)))
            radial[rows//2, cols//2] = 0.0

            filter_fft = radial * angl_spread

            # convolution in frequency domain
            conv = np.fft.ifft2(np.fft.ifftshift(imagefft * filter_fft))

            # even and odd responses (real and imag parts respectively)
            even = np.real(conv)
            odd = np.imag(conv)

            # accumulate
            sum_even += even
            sum_odd += odd

            # amplitude at this scale
            an += np.sqrt(even * even + odd * odd)

            wavelength *= mult

        # local energy for this orientation: energy of the sum of even/odd
        energy_orient = np.sqrt(sum_even * sum_even + sum_odd * sum_odd)

        # accumulate across orientations
        amplitude_sum += an
        pc_sum += energy_orient

    # noise compensation / normalization
    # simple normalization by amplitude sum
    denom = amplitude_sum + eps
    phase_cong = pc_sum / denom

    # clamp to [0,1]
    phase_cong = np.clip(phase_cong, 0.0, 1.0).astype(np.float32)

    return phase_cong
