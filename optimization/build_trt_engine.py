import tensorrt as trt
import os
import sys
from calibrate_int8 import INT8EntropyCalibrator, get_calibration_files

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def build_engine(onnx_file_path, engine_file_path, mode='fp16', calibration_data_dir=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, 3, 640, 640), (8, 3, 640, 640), (16, 3, 640, 640))
    config.add_optimization_profile(profile)

    if mode == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    
    elif mode == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibration_images = get_calibration_files(calibration_data_dir, limit=300)
            calibrator = INT8EntropyCalibrator(
                calibration_images, 
                cache_file='models/calibration.cache',
                batch_size=8
            )
            config.int8_calibrator = calibrator

    serialized_engine = builder.build_serialized_network(network, config)
    
    os.makedirs(os.path.dirname(engine_file_path), exist_ok=True)
    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)


if __name__ == "__main__":
    onnx_path = "../models/best.onnx"
    data_path = "./Hard-Hat-Universe-17/valid/images"
    
    build_engine(onnx_path, "../models/yolov8l_fp16.engine", mode='fp16')
    build_engine(onnx_path, "../models/yolov8l_int8.engine", mode='int8', calibration_data_dir=data_path)