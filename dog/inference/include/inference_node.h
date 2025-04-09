#pragma once
#include <sys/time.h>

#include <memory>
#include <rclcpp/rclcpp.hpp>

class InferenceNode : public rclcpp::Node {
   public:
    InferenceNode(const std::string& name_, const std::string& namespace_);
    // callback接收信息
    void depth_image_callback(const dog::msg::Depth_image&);
    rclcpp::Subscription<dog::msg::Depth_image>::SharedPtr
        depth_image_subscription;

    static rclcpp::Node::SharedPtr param_node;

   private:
    void infer_rknn_callback(
        const std::shared_ptr<dog::srv::InferRKNN_Request> request,
        std::shared_ptr<dog::srv::InferRKNN_Response> response);

    static void dump_tensor_attr(rknn_tensor_attr* attr) {
        printf(
            "  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], "
            "n_elems=%d, "
            "size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1],
            attr->dims[2], attr->dims[3], attr->n_elems, attr->size,
            get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
    }

    static unsigned char* load_model(const char* filename, int* model_size) {
        FILE* fp = fopen(filename, "rb");
        if (fp == nullptr) {
            printf("fopen %s fail!\n", filename);
            return NULL;
        }
        fseek(fp, 0, SEEK_END);
        int model_len = ftell(fp);
        unsigned char* model = (unsigned char*)malloc(model_len);
        fseek(fp, 0, SEEK_SET);
        if (model_len != fread(model, 1, model_len, fp)) {
            printf("fread %s fail!\n", filename);
            free(model);
            return NULL;
        }
        *model_size = model_len;
        if (fp) {
            fclose(fp);
        }
        return model;
    }

    static inline int64_t getCurrentTimeUs() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec * 1000000 + tv.tv_usec;
    }

    rclcpp::Service<dog::srv::InferRKNN>::Sharedptr infer_rknn_service;
};