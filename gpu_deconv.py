#!/usr/bin/env python

# Richardson-Lucy deconvolution code
# James Manton, 2022
# jmanton@mrc-lmb.cam.ac.uk

import numpy as np
import cupy as cp
import timeit
import tifffile
import argparse

def main():
	# Get input arguments
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input', type = str, required = True)
	parser.add_argument('--psf', type = str, required = True)
	parser.add_argument('--output', type = str, required = True)
	parser.add_argument('--num_iters', type = int, default = 10)
	parser.add_argument('--reblurred', type = str, required = False)
	parser.add_argument('--process_psf', type = int, default = 1)
	args = parser.parse_args()

	# Load data
	image = tifffile.imread(args.input)

	# Load and pad PSF if necessary
	psf_temp = tifffile.imread(args.psf)

	if (args.process_psf):
		print("Processing PSF...")
		# Take upper left 16x16 pixels to estimate noise level and create appropriate fake noise
		noisy_region = psf_temp[0:16, 0:16, 0:16]
		psf = np.random.normal(np.mean(noisy_region), np.std(noisy_region), image.shape)
	else:
		psf = np.zeros(image.shape)

	psf[:psf_temp.shape[0], :psf_temp.shape[1], :psf_temp.shape[2]] = psf_temp
	for axis, axis_size in enumerate(psf_temp.shape):
		psf = np.roll(psf, -int(axis_size / 2), axis=axis)

	if (args.process_psf):	
		psf = psf - np.mean(noisy_region)
		psf[psf < 0] = 0

	# Load data and PSF onto GPU
	image = cp.array(image, dtype=cp.float32)
	psf = cp.array(psf, dtype=cp.float32)

	# Calculate OTF and transpose
	otf = cp.fft.rfftn(psf)
	psfT = cp.flip(psf, (0, 1, 2))
	otfT = cp.fft.rfftn(psfT)

	# Log which files we're working with and the number of iterations
	print('Input file: %s' % args.input)
	print('Input shape: %s' % (image.shape, ))
	print('PSF file: %s' % args.psf)
	print('PSF shape: %s' % (psf_temp.shape, ))
	print('Output file: %s' % args.output)
	print('Number of iterations: %d' % args.num_iters)
	print('PSF processing: %s' % args.process_psf)
	print('')

	# Get dimensions of data
	num_z = image.shape[0]
	num_x = image.shape[1]
	num_y = image.shape[2]

	# Calculate Richardson-Lucy iterations
	HTones = fftconv(cp.ones_like(image), otfT)
	recon = cp.ones((num_z, num_x, num_y))

	for iter in range(args.num_iters):
		start_time = timeit.default_timer()
		Hu = fftconv(recon, otf)
		ratio = image / (Hu + 1E-12)
		HTratio = fftconv(ratio, otfT)
		recon = recon * HTratio / HTones
		calc_time = timeit.default_timer() - start_time
		print("Iteration %d completed in %f s." % (iter + 1, calc_time))
	
	# Reblur, collect from GPU and save if argument given
	if args.reblurred is not None:
		reblurred = fftconv(recon, otf)
		reblurred = reblurred.get()
		tifffile.imwrite(args.reblurred, reblurred, bigtiff=True)

	# Collect reconstruction from GPU and save
	recon = recon.get()
	tifffile.imwrite(args.output, recon, bigtiff=True)


def fftconv(x, H):
	return cp.fft.irfftn(cp.fft.rfftn(x) * H, x.shape)

if __name__ == '__main__':
	main()
