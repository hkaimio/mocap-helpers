# OpenSim to BVH Exporter

This script exports OpenSim models and animations to BVH (BioVision Hierarchy)
format, making them compatible with animation software like Blender, iClone, and
other motion capture tools.

Main use case for me for this script is usign OpenSim  for complex inverse
kinematics use cases (like fitting full body armature to motion capture marker
traejctories, which is more thatn Blender IK can handle) My typical workflow is

- Scale Blender model armature to match tracked person's size
- Export Bledner armature as OpenSim skeleton (I have another script for that
  which I'll try to share in near future)
- Run IK in OpenSim
- Convert the OpenSim `.mot` file (that contains IK results) and skeleton
  definition `.osim`file to BVH using this script
- Import the BVH file back to Blender or other animation tool for retargeting (I
  use also iClone which is a commercial charqacter animation tool)

But the script should work also with other OpenSim skeletons; support for the
more complex joint types has not been tested.

## Environment Setup

OpenSim Python packages are not available for all Python versions. It's
recommended to use Miniconda to create a compatible Python environment.

### Setting Up Miniconda Environment

Follow the official OpenSim Python setup instructions:

**👉 [OpenSim Scripting in Python Guide](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python)**

The guide covers:
- Installing Miniconda
- Creating a Python environment with OpenSim
- Installing the OpenSim Python API

### Additional Requirements

Once you have OpenSim installed, install the following dependencies:

```bash
# Activate your OpenSim conda environment
conda activate opensim-scripting  # or your environment name

# Install NumPy & bvhsdk(if not already installed)
conda install numpy bvhsdk
```

## Usage

### Export Rest Pose Only

To export just the model structure without animation:

```bash
python opensim_to_bvh.py --model model.osim -o output.bvh
```

### Export Animation

To export a model with motion data:

```bash
python opensim_to_bvh.py --model model.osim --motion motion.mot -o output.bvh
```

### Advanced Options

```bash
python opensim_to_bvh.py --model model.osim --motion motion.mot -o output.bvh \
    --start-frame 0 \
    --end-frame 100 \
    --framerate 30.0 \
    --root-body pelvis
```

#### Command Line Arguments

- `--model` (required): Path to OpenSim model file (.osim)
- `-o, --output` (required): Output BVH file path
- `--motion`: Motion file (.mot, .sto) - if omitted, exports rest pose only
- `--framerate`: Output framerate in fps (default: 30.0)
- `--start-frame`: Starting frame number (default: 0)
- `--end-frame`: Ending frame number (default: all frames)
- `--root-body`: OpenSim root body name (default: pelvis)

## Output Compatibility

The exported BVH files have been tested and confirmed to work with:

- **Blender** - Import via File → Import → Motion Capture (.bvh)
- **iClone** - Import as external motion data. For proper import, you need to create characterization for the skeleton using iClone/cCharacter Creator tools.
- Other standard animation tools that support BVH format

The script uses standard BVH format conventions, ensuring broad compatibility across motion capture and animation software.

## Technical Details

- Converts OpenSim body hierarchy to BVH joint hierarchy
- Exports joint rotations and translations
- Uses OpenSim body names as BVH joint names for clarity
- Converts units from meters (OpenSim) to centimeters (BVH standard)

## Troubleshooting

### ImportError: OpenSim Python API not available

Make sure you've properly installed OpenSim in your Python environment. Follow the [OpenSim Python setup guide](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python).

### File not found errors

Ensure that:
- Your model file (.osim) path is correct
- If using motion data, the motion file (.mot, .sto) exists
- You have read permissions for the input files
- The output directory is writable

### Root body not found

If you get an error about the root body not being found, list the available bodies in your model and specify the correct root body using the `--root-body` argument.
