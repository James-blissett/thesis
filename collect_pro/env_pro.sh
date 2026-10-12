# Source after env.sh: points `libero` at the LIBERO-PRO clone without touching the
# existing install or ~/.libero/config.yaml.
export PYTHONPATH=/ephemeral/code/LIBERO-PRO${PYTHONPATH:+:$PYTHONPATH}
export LIBERO_CONFIG_PATH=/data/tmp/libero_pro_config
export MUJOCO_GL=egl
