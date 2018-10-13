# Welcome!

I built this script as a small Saturday project after taking a timelapse of about 500 images and realizing that, because it involved direct light, the sunrise, etc..., there was a lot of flickering, i.e. sudden changes in exposition caused by the camera's metering deciding different things for different images.

This script implements a very simple procedure:

1. It estimates the brightness of the sequence of pictures you drop in the `timelapse`  folder, and fits a polynomial function to it to smooth it out;
2. It changes the gamma of each picture to better approximate the polynomial, ensuring smooth transitions in brightness between pictures;
3. Dumps the resulting images into the `output` folder, ordered and ready to be used by Adobe Premiere or whatever you use to collate your pictures into timelapses.

It was tested on a set of ~520 20 megapixel images, and managed to chew through them in about 3 minutes on my i7-6700HQ, on a virtual machine limited to 4 cores. It's fast enough for my purposes so, for now, I won't focus a lot on optimization, although I'm pretty sure most of what I'm doing (looking at you, IO) could probably be greatly optimized. Feel free to open issues or PRs if you have any great ideas.