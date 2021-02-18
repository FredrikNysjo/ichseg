import os

for input_filename in os.listdir("."):
    if not input_filename.count(".nii"):
        continue
    prefix = input_filename.split(".")[0]
    output_filename = prefix + ".vtk"
    cmd = "c3d %s -type short %s" % (input_filename, output_filename)
    print(cmd)
    os.system(cmd)
