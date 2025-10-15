# Function requirement

I want you first help me write a plan for coding. I want to write a simple package (in @spxquery) that can satisify following functions

## Common workflow

- Users pass a location of a source (ra = xx, dec = xx) and a location to save the download images (optional, if not
  specify, download the files at the location where runs the script)
- For the fully automatic mode:
  - The package query to the database and find out how many observations are available in SPHEREx
  - Print out a basic archive search information:
    - How many observations been found in total and in each band (`results["energy_bandpassname"]=='SPHEREx-D[1-6]'`)
    - The time span of the observations (in days). Use the observation time of the latest one minus the earlest one.
      (The observation time can be calculated as: `(results["t_max"]+results["t_min"])/2`)
    - The estimated total volume having all files downloaded (unless user specify the band). One image is about 70Mb.
    - Save the query infomation out.
  - Start the file downloading in parallel. If possible, show a progress bar for user to know the expected time
    cosumption.
  - After all files are downloaded (be aware there could be some failed), print out an overview info of downloading task.
  - I'm interested in the time-domain astronomy, so you will then generate a rough data series for this source. For each
    of the image, you should implement following procedures to extract necessary informations:
    - Subtract the zodical light background.
    - Apply a force photometry to the source (by default 3 pixels diameter aperture but make it optional to users),
      extract the MJD, flux, error, wavelength, bandwidth and flag (OR for flag in each pixel). You can document the flag in
      binary format.
  - Finally, you provide the light curve of this source in a csv file. You should also attach necessary informations
    e.g., pix_x and pix_y of this position, the obs_id for this observation. Save the file along side the image folder.
  - Generate a plot to show the information of this source. It should contain two subplots:
    - The upper plot shows the spectral of this target. The x-axis is wavelength, y-axis is the flux. You can use
      x-errorbar to indicate the bandwidth, and y-errorbar to indicate the flux error. In case that error is larger than
      expected flux, use an upper limit marker.
    - The lower plot shows the light curve. The x-axis is the mjd and y-axis is the flux. You should color-code the
      wavelength of each observation. The expected wavelength range is 0.75-5.1 microns.
- This code should also be able to run step-by-step. So I require you pack the code properly so that the package can
  start from:
  - After query and before the file download (or any time if the download is breaked somehow)
  - After file fully downloaded and before processing
  - After the csv of lightcurve is generated and before plotting the QA image.