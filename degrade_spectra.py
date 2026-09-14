#!/usr/bin/env python
"""Degrade a synthetic spectral cube to Sunrise/SUSI resolution.

Spatially convolves each wavelength slice with the Sunrise Airy-disk PSF,
then spectrally convolves with a Gaussian, and writes the result to FITS.
Input/output FITS: HDU0 = cube (NX, NZ, NL), HDU1 = wavelength axis (nm).
"""
import argparse
import numpy as np
import astropy.io.fits as fits
import lightweaver as lw
from astropy.convolution import AiryDisk2DKernel, convolve
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


def degrade(spectrum, ll_vac, aperture=0.7, pix_km=22.0, kernel_size=33,
            spec_fwhm=0.004):
    """Return (degraded cube, air wavelength axis)."""
    ll_air = lw.vac_to_air(ll_vac)

    # Airy PSF: first-zero radius (km) -> pixels, at the bluest wavelength.
    diff_limit = 1.22 * ll_vac[0] * 1e-9 / aperture * 206265.0 * 725.0  # km
    psf = AiryDisk2DKernel(diff_limit / pix_km,
                           x_size=kernel_size, y_size=kernel_size)

    out = np.zeros_like(spectrum)
    for i in tqdm(range(spectrum.shape[2]), desc='spatial PSF'):
        out[:, :, i] = convolve(spectrum[:, :, i], psf,
                                boundary='extend', normalize_kernel=True)

    # Spectral Gaussian (FWHM in nm -> sigma in pixels).
    sigma_pix = (spec_fwhm / (2 * np.sqrt(2 * np.log(2)))) / (ll_air[1] - ll_air[0])
    for j in tqdm(range(out.shape[0]), desc='spectral'):
        out[j] = gaussian_filter1d(out[j], sigma=sigma_pix, axis=-1)

    return out, ll_air


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', help='synthetic cube FITS (HDU0 cube, HDU1 lambda_vac)')
    p.add_argument('output', help='output FITS')
    p.add_argument('--aperture', type=float, default=0.7, help='telescope aperture [m]')
    p.add_argument('--pix-km', type=float, default=22.0, help='spatial pixel scale [km]')
    p.add_argument('--kernel-size', type=int, default=33, help='PSF kernel size [pix]')
    p.add_argument('--spec-fwhm', type=float, default=0.004, help='spectral FWHM [nm]')
    args = p.parse_args()

    with fits.open(args.input) as h:
        spectrum, ll_vac = h[0].data, h[1].data

    out, ll_air = degrade(spectrum, ll_vac, args.aperture, args.pix_km,
                          args.kernel_size, args.spec_fwhm)

    fits.HDUList([fits.PrimaryHDU(out),
                  fits.ImageHDU(ll_air)]).writeto(args.output, overwrite=True)
    print(f'wrote {args.output}  {out.shape}')


if __name__ == '__main__':
    main()
