#!/usr/bin/env python2

# Futures
from __future__ import print_function, division

# Standard
import os
import multiprocessing

# Non-standard
import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_DIR = "timelapse"
OUTPUT_DIR = "output"


def list_images(input_dir):
	""" Produces a list of all the filenames in the input directory. """
	# List the file names
	filenames = sorted(os.listdir(input_dir))

	# Filter out our instructional file
	# (putting it there was a great idea)
	filenames = [f for f in filenames if "DROP_INPUT" not in f]

	# And return them
	return filenames


def load_images(filenames):
	"""
	Loads images from the folder into an array.
	"""
	# An empty initial list for images
	images = []

	# Load an image per filename
	for filename in filenames:
		path = os.path.join(INPUT_DIR, filename)
		images.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

	# And we're done!
	return images


def reduce_images(images):
	"""
	Reduce all images, to speed up processing.
	"""
	# Again, an empty list
	reduced_images = []

	# Reduce each individual image
	for image in images:
		fraction = 0.2
		new_size = (int(image.shape[1] * fraction), int(image.shape[0] * fraction))
		small_img = cv2.resize(image, new_size)
		reduced_images.append(small_img)

	# And we're done!
	return reduced_images


def calculate_luminance(image):
	"""
	Calculates the luminance or brightness or whatever of a single OpenCV image.

	https://stackoverflow.com/questions/6442118/python-measuring-pixel-brightness
	"""
	# Get image dimensions
	h = image.shape[0]
	w = image.shape[1]

	# Calculate for each pixel
	brightness = []
	for y in range(0, h, int(h/50)):
		for x in range(0, w, int(w/50)):
			r,g,b = image[y, x]
			#brightness.append(0.299*r + 0.587*g + 0.114*b)
			brightness.append(0.333*r + 0.333*g + 0.333*b)

	# And return an average
	return np.mean(brightness)


def calculate_luminances(images):
	"""
	Calculates the sequence of luminances in a list of opencv images.
	"""
	# An empty list for luminances
	luminances = []

	# Calculate for each image
	p = multiprocessing.Pool(multiprocessing.cpu_count())
	luminances = p.map(calculate_luminance, images)

	#print(p.map(f, [1, 2, 3]))
	#for image in images:
	#	luminances.append(calculate_luminance(image))

	# And we're done!
	return luminances


def worker_func(path):
	"""
	Worker function for calculate_luminances_files()
	"""
	image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
	return calculate_luminance(image)


def calculate_luminances_files(directory, filenames):
	"""
	Calculates the luminance of each image using a pool of processes.
	"""
	# Create a list of all the paths
	paths = []
	for filename in filenames:
		path = os.path.join(directory, filename)
		paths.append(path)
		
	# Start a pool with a worker_func for each path
	p = multiprocessing.Pool(multiprocessing.cpu_count())
	luminances = p.map(worker_func, paths)

	# And we're done!
	return luminances


def export_images(images, out_dir):
	"""
	Export a set of images to an output directory.
	"""
	for i, image in enumerate(images):
		path = os.path.join(out_dir, str(i)+".jpg")
		cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def set_luminance(image, init_luminance, final_luminance):
	"""
	Tries to approximate the luminance of a certain image to a given value.
	"""
	# Copy the image to a new array, so we don't screw it up
	new_image = np.copy(image)

	# Get image dimensions
	h = new_image.shape[0]
	w = new_image.shape[1]

	# Calculate the luminance difference
	lum_diff = (final_luminance - init_luminance)

	# Adjust gamma
	new_image = adjust_gamma(image, 1 + (4*lum_diff)/255)

	# Re-calculate luminance
	new_luminance = calculate_luminance(new_image)

	# Calculate the luminance difference
	lum_diff = (final_luminance - new_luminance)

	# And give it a second pass for good measure
	new_image = adjust_gamma(new_image, 1 + (4*lum_diff)/255)

	# Re-calculate luminance
	new_luminance = calculate_luminance(new_image)

	# Calculate the luminance difference
	lum_diff = (final_luminance - new_luminance)

	# And a third because why not, we have the CPU to spare
	new_image = adjust_gamma(new_image, 1 + (4*lum_diff)/255)

	# And we're done!
	return new_image


def adjust_gamma(image, gamma=1.2):
	# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def equalize_luminances(images, luminances):
	"""
	Given a luminance, tries to equalize all images to it.
	"""
	# Calculate the luminance we'll be setting the images to
	avg_luminance = np.mean(luminances)

	# Set luminances individually
	equalized_images = []
	for i, image in enumerate(images):
		equalized_images.append(set_luminance(image, luminances[i], avg_luminance))

	# And we're done!
	return equalized_images


def curve_luminances(images, init_luminances, target_luminances):
	"""
	Given a luminance, tries to make luminance follow the curve.
	"""
	# Set luminances individually
	equalized_images = []
	for i, image in enumerate(images):
		equalized_images.append(set_luminance(image, init_luminances[i], target_luminances[i]))

	# And we're done!
	return equalized_images


def worker_func_luminance(args):
	# Unpack arguments
	in_path = args[0]
	init_luminance = args[1]
	target_luminance = args[2]
	out_path = args[3]

	# Do the roar
	image = cv2.cvtColor(cv2.imread(in_path), cv2.COLOR_BGR2RGB)
	equalized_image = set_luminance(image, init_luminance, target_luminance)
	cv2.imwrite(out_path, cv2.cvtColor(equalized_image, cv2.COLOR_RGB2BGR))


def curve_luminances_files(filenames, init_luminances, target_luminances):
	"""
	Given a luminance, tries to make luminance follow the curve.
	"""
	# Generate the vector of arguments we'll need
	args = []
	equalized_image_filenames = []
	for i, filename in enumerate(filenames):
		# Generate the input and output paths, and the initial and target luminances
		in_path = os.path.join(INPUT_DIR, filename)
		out_filename = str(i)+".jpg"
		out_path = os.path.join(OUTPUT_DIR, out_filename)
		args.append([in_path, init_luminances[i], target_luminances[i], out_path])
		equalized_image_filenames.append(out_filename)

	# Start a pool with a worker_func for each path
	p = multiprocessing.Pool(multiprocessing.cpu_count())
	p.map(worker_func_luminance, args)

	# And we're done!
	return equalized_image_filenames


def plot_luminance_curves(curves, filename, labels=None):
	"""
	Plots a luminance curve.
	"""
	plt.figure()
	for i, curve in enumerate(curves):
		if labels==None:
			plt.plot(curve)
		else:
			plt.plot(curve, label=labels[i])
	plt.legend()
	plt.savefig(filename)
	plt.clf()


def fit_luminance_curve(luminances):
	"""
	Interpolates and smoothes a new luminance curve.
	"""
	poly = np.polyfit(range(len(luminances)), luminances, 5)
	return np.polyval(poly, range(len(luminances)))


def calculate_error(luminances, ref_curve):
	"""
	Calculates the average, std and total error.
	"""
	# Calculate the error
	error = np.abs(np.array(luminances) - np.array(ref_curve))
	# And return it
	return np.sum(error), np.mean(error), np.std(error)


def deflicker_in_memory():
	print("Listing filenames")
	filenames = list_images(INPUT_DIR)

	print("Loading images")
	images = load_images(filenames)
	print("Loaded", len(images), "images.")

	print("Calculating luminances")
	luminances = calculate_luminances(images)
	print("Initial luminances:")
	print("Mean:", np.mean(luminances), "std:", np.std(luminances))

	print("Fitting luminance curve")
	fitted_curve = fit_luminance_curve(luminances)

	calculate_error(luminances, fitted_curve)

	print("Equalizing luminances to", np.mean(luminances))
	equalized_images = curve_luminances(images, luminances, fitted_curve)

	print("Calculating luminances")
	new_luminances = calculate_luminances(reduce_images(equalized_images))
	print("New luminances:")
	print("Mean:", np.mean(new_luminances), "std:", np.std(new_luminances))

	calculate_error(new_luminances, fitted_curve)
	
	plot_luminance_curves([luminances, fitted_curve, new_luminances], "curves.pdf", ["original", "fitted", "result"])


def deflicker_with_files(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
	"""
	Instead of loading everything into memory and blowing up, this function
	uses file-based methods. Thanks to caching and whatnot, the IO latency is
	not enough to bottleneck the process, at least for now.
	"""
	print("Listing filenames")
	original_filenames = list_images(input_dir)

	print("Calculating luminances")
	luminances = calculate_luminances_files(input_dir, original_filenames)
	print("Initial luminances:")
	print("Mean:", np.mean(luminances), "std:", np.std(luminances))

	# TODO: Filter outliers

	print("Fitting luminance curve")
	fitted_curve = fit_luminance_curve(luminances)

	print("Calculating error")
	err_sum, err_mean, err_std = calculate_error(luminances, fitted_curve)
	print("Total error:", err_sum, "avg", err_mean, "std", err_std)

	print("Curving luminances")
	equalized_image_filenames = curve_luminances_files(original_filenames, luminances, fitted_curve)

	print("Calculating luminances")
	new_luminances = calculate_luminances_files(output_dir, equalized_image_filenames)
	print("New luminances:")
	print("Mean:", np.mean(new_luminances), "std:", np.std(new_luminances))

	print("Calculating error")
	err_sum, err_mean, err_std = calculate_error(new_luminances, fitted_curve)
	print("Total error:", err_sum, "avg", err_mean, "std", err_std)
	
	print("Plotting curves")
	plot_luminance_curves([luminances, fitted_curve, new_luminances], "curves.pdf", ["original", "fitted", "result"])


if __name__ == "__main__":
	deflicker_with_files()	

