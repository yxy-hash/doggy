#include "inference/include/inference_node.h"

#include <stdio.h>
#include <string.h>
#include <sys/time.h>

#include <rclcpp/rclcpp.hpp>

#include "rknn_api.h"

rclcpp::Node::SharedPtr InferenceNode::param_node = nullptr;

InferenceNode::InferenceNode(const std::string& name_,
                             const std::string& namespace_)
    : rclcpp::Node(name_, namespace_,
                   rclcpp::NodeOptions().use_global_arguments(false)) {
    depth_image_subscription = this->create_subscription<dog::msg::Depthimage>(
        "depth_image", 1,
        std::bind(&InferenceNode::depth_image_callback, this,
                  std::placeholders::_1));
    if (!param_node) {
        param_node = std::make_shared<rclcpp::Node>("infer", "dog");
        return;
    }
};

void InferenceNode::infer_rknn_callback(
    const std::shared_ptr<dog::srv::InferRKNN_Request> request,
    std::shared_ptr<dog::srv::InferRKNN_Response> response) {

    infer_service = this->create_service<dog::srv::InferRKNN>(
        "infer_rknn", std::bind(&InferenceNode::infer_rknn_callback, this,
                                std::placeholders::_1, std::placeholders::_2));

    rknn_context context;

    std::string model_path_str;
    this->get_parameter("model_path", model_path_str);
    if (model_path_str.empty()) {
        RCLCPP_ERROR(this->get_logger(), "model_path is empty!");
        return;
    }
    const char* model_path = model_path_str.c_str();  // 确保生命周期足够
    int model_len = 0;
    unsigned char* model = load_model(model_path, &model_len);
    int ret = rknn_init(&context, model, 0, 0, NULL);
    if (ret != RKNN_SUCC) {
        RCLCPP_ERROR(this->get_logger(), "RKNN_init Fail, error code: %d", ret);
        return;
    }

    // Get sdk and driver version
    rknn_sdk_version sdk_ver;
    ret =
        rknn_query(context, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(context);
        return;
    }
    printf("rknn_api/rknnrt version: %s, driver version: %s\n",
           sdk_ver.api_version, sdk_ver.drv_version);

    // Get Model Input Output Info
    rknn_input_output_num io_num;
    ret = rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(context);
        return;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
           io_num.n_output);

    // Print input and output tensor
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error! ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&input_attrs[i]);
    }
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                         sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    float* input_data_0 =
        (float*)malloc(input_attrs[0].n_elems * sizeof(float));
    ;
    float* input_data_1 =
        (float*)malloc(input_attrs[1].n_elems * sizeof(float));
    ;
    rknn_tensor_mem* input_mem[2];
    input_mem[0] = rknn_create_mem(context, input_attrs[0].size_with_stride);
    input_mem[1] = rknn_create_mem(context, input_attrs[1].size_with_stride);
    memcpy(input_mem[0]->virt_addr, input_data_0,
           input_attrs[0].n_elems * sizeof(float));
    memcpy(input_mem[1]->virt_addr, input_data_1,
           input_attrs[1].n_elems * sizeof(float));
    rknn_tensor_mem* input_mems[2] = {input_mem[0], input_mem[1]};

    float* output_data =
        (float*)malloc(output_attrs[0].n_elems * sizeof(float));
    rknn_tensor_mem* output_mem;
    output_mem = rknn_create_mem(context, output_attrs[0].size_with_stride);
    memcpy(output_mem->virt_addr, output_data,
           output_attrs[0].n_elems * sizeof(float));

    ret = rknn_set_io_mem(context, input_mems[0], &input_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem input[0] error %d\n", ret);
        return;
    }
    ret = rknn_set_io_mem(context, input_mems[1], &input_attrs[1]);
    if (ret < 0) {
        printf("rknn_set_io_mem input[1] error %d\n", ret);
        return;
    }
    ret = rknn_set_io_mem(context, output_mem, &output_attrs[0]);
    if (ret < 0) {
        printf("rknn_set_io_mem input error %d\n", ret);
        return;
    }

    // rknn run
    printf("Begin rknn inference ...\n");
    int64_t start_us = getCurrentTimeUs();
    ret = rknn_run(context, NULL);
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    if (ret < 0) {
        printf("rknn run error %d\n", ret);
        return;
    }
    printf("Elapse Time = %.2fms, FPS = %.2f\n", elapse_us / 1000.f,
           1000.f * 1000.f / elapse_us);

    // Destroy rknn memory
    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        rknn_destroy_mem(context, input_mems[i]);
    }
    rknn_destroy_mem(context, output_mem);

    if (input_data_0 != nullptr && input_data_1 != nullptr) {
        free(input_data_0);
        free(input_data_1);
    }

    if (model != nullptr) {
        free(model);
    }
    // destroy
    rknn_destroy(context);
}

int main(int argv, char** argc) {
    rclcpp::init(argv, argc);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto inference_node = std::make_shared<InferenceNode>();
    executor.add_node(inference_node);
    executor.spin();

    rclcpp::shutdown();
}