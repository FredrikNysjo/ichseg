"""
.. module:: image_manager
   :platform: Linux, Windows
   :synopsis: Manager for handling image data (volumes and masks)

.. moduleauthor:: Fredrik Nysjo
"""

import image_utils
import image_nifti
import image_dicom

import numpy as np


class ImageManager:
    def __init__(self):
        self.volume = None
        self.header = None
        self.mask = None
        self.measured_volume_ml = 0.0
        self.labels = []
        self.active_label = 0

    def load_volume_fromfile(self, filename):
        """Load a grayscale volume from file"""
        self.current, self.label = 0, 0
        if ".vtk" in filename:
            self.volume, self.header = image_utils.load_vtk(filename)
        elif ".nii" in filename:
            self.volume, self.header = image_nifti.load_nii(filename)
        elif filename:  # Assume format is DICOM
            self.volume, self.header = image_dicom.load_dicom(filename)
        else:
            self.volume, self.header = image_utils.create_dummy_volume()
        self.mask = np.zeros(self.volume.shape, dtype=np.uint8)

    def load_mask_fromfile(self, filename):
        """Load a segmentation mask from file

        If the grayscale volume we are segmenting does not match the size
        of the loaded mask, a new empty volume will be created

        Currently, only 8-bit masks in VTK or NIfTI format are supported
        """
        if ".vtk" in filename:
            mask, mask_header = image_utils.load_vtk(filename)
        elif ".nii" in filename:
            mask, mask_header = image_nifti.load_nii(filename)
        else:
            return  # TODO Should we create an empty mask here instead?
        if mask.dtype == np.uint8:
            if mask.shape != self.volume.shape:
                self.header = mask_header
                self.volume = np.zeros(mask.shape, dtype=np.uint8)
            self.mask = mask

    def resample_volume(self):
        """Resample volume with orientation data into non-oriented version with
        twice the axial resolution

        This method will also upsample non-oriented volumes
        """
        self.volume, self.header = image_dicom.resample_volume(self.volume, self.header)
        self.mask = np.zeros(self.volume.shape, dtype=np.uint8)

    def update_measured_volume(self):
        """Update the measured volume (calculated in millilitres) of the
        segmentation in the mask
        """
        voxel_volume_ml = np.prod(self.header["spacing"]) * 1e-3
        self.measured_volume_ml = np.sum(self.mask > 127) * voxel_volume_ml
