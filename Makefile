AF_PATH=/home/abhijit/jacket/arrayfire
BIN := gpu_lbm cpu_lbm af_lbm
include $(AF_PATH)/examples/common.mk
CFLAGS += -arch=sm_21 --ptxas-options=-v
LDFLAGS += -lafGFX
