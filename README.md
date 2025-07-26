# TNG50 Dataset Visualization GUI

**Description:** Reading from the TNG50 dataset to extract Mily Way-like galaxies in hdf5 format. This repository contains code to read this data and display relevant 3D models with a PyQt5 GUI. For each galaxy, you can interatively view the galaxy from multiple angles, choose to view its stellar origin, velocity, or metallicity mappings, or filter stellar particles in the model depending on where they originated from (see images below)

**Motivation:** Our end goal is to use a CNN to detect streams, which are evidence left by galaxies when they merge together. These streams give us more information about the galaxy merger, and about the merged galaxies themselves. Since we're not sure how streams would appear in our data files, we would have to go through each galaxy extensively to look for streams and label the data. Reading the file and downloading graphs of the galaxy is time consuming, and non interactive. This GUI is designed to help save time in the data preparation and data labeling stage of this research. 

**Technologies:** TensorFlow, NumPy, PyQt5, PyVista, Matplotlib, Scipy, Mounted Volume Drives

**Visualization Examples:**
<img width="1193" height="794" alt="image" src="https://github.com/user-attachments/assets/d630180c-c5bc-47cf-b028-821c932eea0f" />
(Viewing velocity map of the galaxy)

<img width="1187" height="793" alt="image" src="https://github.com/user-attachments/assets/969b822f-a422-454b-b305-5850d89b1218" />
(Viewing velocity map of the galaxy with only the stars accreted from other galaxies)

<img width="1192" height="790" alt="image" src="https://github.com/user-attachments/assets/90e0571f-aed5-4751-85b4-46ad2acbf240" />
(Distribution of stellar particles born from only the main galaxy)
