"""
.. module:: model_manager
   :platform: Linux, Windows
   :synopsis: Manager for handling calling of DL segmentation models

.. moduleauthor:: Fredrik Nysjo
"""

import json
import sys
import os
import subprocess


class ModelManager:
    def __init__(self):
        self.modelinfo = []
        self.result_filename = ""

    def load_model_info_from_json(self, filename):
        """Load info stored in JSON file about available DL models"""
        root = {}
        with open(filename, "r") as stream:
            root = json.load(stream)
        self.modelinfo = root["models"] if "models" in root else []

    def launch_model(self, model, input_filename):
        """Run DL segmentation model inside a separate process"""
        assert type(model) == dict
        assert "conda_env" in model
        assert "working_dir" in model
        assert "cmd" in model
        assert "arguments" in model
        assert "output_filename" in model

        self.result_filename = ""
        if ".nii" not in input_filename:
            print("Only NIfTI files supported as model input at the moment")
            return
        if not os.path.isabs(input_filename):
            print("Input filename must be an absolute path")
            return

        conda_env = model["conda_env"]
        working_dir = model["working_dir"]
        cmd = model["cmd"]
        arguments = model["arguments"]
        output_filename = model["output_filename"]
        if output_filename == "":
            print("Output filename in models.json cannot be empty")
            return

        input_basename = os.path.basename(input_filename).split(".")[0]
        input_dir = os.path.dirname(os.path.normpath(input_filename))
        input_basedir = os.path.split(input_dir)[1]

        arguments = arguments.replace("{working_dir}", working_dir)
        arguments = arguments.replace("{input_filename}", input_filename)
        arguments = arguments.replace("{input_dir}", input_dir)
        arguments = arguments.replace("{input_basedir}", input_basedir)
        output_filename = output_filename.replace("{working_dir}", working_dir)
        output_filename = output_filename.replace("{input_basename}", input_basename)
        output_filename = output_filename.replace("{input_dir}", input_dir)
        output_filename = output_filename.replace("{input_basedir}", input_basedir)

        if sys.platform.startswith("linux"):
            args = 'bash -i -c "'
            args += "conda activate " + conda_env + " ; "
            args += cmd + " " + arguments + " ; "
            args += '"'
        else:
            print("Models are not supported yet on this platform")
            return

        print("Launching segmentation model...")
        status = subprocess.run(args, cwd=working_dir, shell=True)
        print(status)
        self.result_filename = output_filename
        print("Done.")
