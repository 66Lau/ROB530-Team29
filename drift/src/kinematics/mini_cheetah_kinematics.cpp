#include "drift/kinematics/mini_cheetah_kinematics.h"

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <iostream>

namespace measurement::kinematics {

  #define NLEG 4    // num of legs
  #define NAPL 3    // dofs of legs
  
  MiniCheetahKinematics::MiniCheetahKinematics() : LeggedKinematicsMeasurement() {
    positions_.setConstant(3, NLEG, 0);
    jacobians_.setConstant(3, NLEG * NAPL, 0);
    contacts_.setConstant(NLEG, 1, 0);
    encoders_.setConstant(NLEG * NAPL, 1, 0);
  
    urdf_path_ = "/home/lau/hw/rob530/final/drift/src/kinematics/robots/go1.urdf";
    foot_name_fr_ = "FR_foot";
    foot_name_fl_ = "FL_foot";
    foot_name_hr_ = "RR_foot";
    foot_name_hl_ = "RL_foot";
  
    initializePinocchio();
  }
  
  MiniCheetahKinematics::MiniCheetahKinematics(
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& encoders,
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& d_encoders,
      const Eigen::Matrix<bool, Eigen::Dynamic, 1>& contacts)
      : LeggedKinematicsMeasurement(encoders, d_encoders, contacts) {
    positions_.setConstant(3, NLEG, 0);
    jacobians_.setConstant(3, NLEG * NAPL, 0);
  
    urdf_path_ = "/home/lau/hw/rob530/final/drift/src/kinematics/robots/go1.urdf";
    foot_name_fr_ = "FR_foot";
    foot_name_fl_ = "FL_foot";
    foot_name_hr_ = "RR_foot";
    foot_name_hl_ = "RL_foot";
  
    initializePinocchio();
  }
  
  void MiniCheetahKinematics::initializePinocchio() {
    try {
      pinocchio::urdf::buildModel(urdf_path_, model_);
      data_ = pinocchio::Data(model_);
    } catch (std::exception& e) {
      std::cerr << "Error loading URDF model: " << e.what() << std::endl;
      throw e;
    }
  
    std::string body_frame_name = "trunk";
    body_frame_id_ = model_.getFrameId(body_frame_name);
    if (body_frame_id_ == -1) {
      std::cerr << "Error: Body frame '" << body_frame_name << "' not found in the model." << std::endl;
      throw std::runtime_error("Body frame not found");
    }
  
    // get feet frame ID
    frame_id_fr_ = model_.getFrameId(foot_name_fr_);
    if (frame_id_fr_ == -1) {
      std::cerr << "Error: Frame '" << foot_name_fr_ << "' not found in the model." << std::endl;
      throw std::runtime_error("Front right foot frame not found");
    }
  
    frame_id_fl_ = model_.getFrameId(foot_name_fl_);
    if (frame_id_fl_ == -1) {
      std::cerr << "Error: Frame '" << foot_name_fl_ << "' not found in the model." << std::endl;
      throw std::runtime_error("Front left foot frame not found");
    }
  
    frame_id_hr_ = model_.getFrameId(foot_name_hr_);
    if (frame_id_hr_ == -1) {
      std::cerr << "Error: Frame '" << foot_name_hr_ << "' not found in the model." << std::endl;
      throw std::runtime_error("Hind right foot frame not found");
    }
  
    frame_id_hl_ = model_.getFrameId(foot_name_hl_);
    if (frame_id_hl_ == -1) {
      std::cerr << "Error: Frame '" << foot_name_hl_ << "' not found in the model." << std::endl;
      throw std::runtime_error("Hind left foot frame not found");
    }
    //     std::cout << "Joint names and their indices:" << std::endl;
    // for (pinocchio::JointIndex i = 0; i < model_.njoints; ++i) {
    //   std::cout << "Index " << i << ": " << model_.names[i] << std::endl;
    // }
  }
  
  // using Pinocchio compute kinematics，and update foot pos and jocobian
  void MiniCheetahKinematics::ComputeKinematics() {
    Eigen::Matrix<double, 12, 1> ordered_encoders;
    ordered_encoders.segment<3>(0) = encoders_.segment<3>(3);  // FL -> index 0~2
    ordered_encoders.segment<3>(3) = encoders_.segment<3>(0);  // FR -> index 3~5
    ordered_encoders.segment<3>(6) = encoders_.segment<3>(9);  // RL -> index 6~8
    ordered_encoders.segment<3>(9) = encoders_.segment<3>(6);  // RR -> index 9~11

    Eigen::Matrix<double, 12, 1> ordered_dencoders;
    ordered_dencoders.segment<3>(0) = d_encoders_.segment<3>(3);  // FL -> index 0~2
    ordered_dencoders.segment<3>(3) = d_encoders_.segment<3>(0);  // FR -> index 3~5
    ordered_dencoders.segment<3>(6) = d_encoders_.segment<3>(9);  // RL -> index 6~8
    ordered_dencoders.segment<3>(9) = d_encoders_.segment<3>(6);  // RR -> index 9~11


    pinocchio::forwardKinematics(model_, data_, ordered_encoders);
    pinocchio::updateFramePlacements(model_, data_);
  
    Eigen::Isometry3d X_base = Eigen::Isometry3d::Identity();
    X_base.linear() = data_.oMf[body_frame_id_].rotation();
    X_base.translation() = data_.oMf[body_frame_id_].translation();
  
    // compute the foot pos relative to base
    {
      Eigen::Isometry3d X_fr = Eigen::Isometry3d::Identity();
      X_fr.linear() = data_.oMf[frame_id_fr_].rotation();
      X_fr.translation() = data_.oMf[frame_id_fr_].translation();
      Eigen::Vector3d p_fr = (X_base.inverse() * X_fr).translation();
      positions_.col(FR) = p_fr;
    }
    {
      Eigen::Isometry3d X_fl = Eigen::Isometry3d::Identity();
      X_fl.linear() = data_.oMf[frame_id_fl_].rotation();
      X_fl.translation() = data_.oMf[frame_id_fl_].translation();
      Eigen::Vector3d p_fl = (X_base.inverse() * X_fl).translation();
      positions_.col(FL) = p_fl;
    }
    {
      Eigen::Isometry3d X_hr = Eigen::Isometry3d::Identity();
      X_hr.linear() = data_.oMf[frame_id_hr_].rotation();
      X_hr.translation() = data_.oMf[frame_id_hr_].translation();
      Eigen::Vector3d p_hr = (X_base.inverse() * X_hr).translation();
      positions_.col(HR) = p_hr;
    }
    {
      Eigen::Isometry3d X_hl = Eigen::Isometry3d::Identity();
      X_hl.linear() = data_.oMf[frame_id_hl_].rotation();
      X_hl.translation() = data_.oMf[frame_id_hl_].translation();
      Eigen::Vector3d p_hl = (X_base.inverse() * X_hl).translation();
      positions_.col(HL) = p_hl;
    }

    // std::cout << " " << std::endl;
    // std::cout << "Foot positions: " << std::endl;
    // std::cout << "FR: " << positions_.col(FR) << std::endl;
    // std::cout << "FL: " << positions_.col(FL) << std::endl;
    // std::cout << "HR: " << positions_.col(HR) << std::endl;
    // std::cout << "HL: " << positions_.col(HL) << std::endl;
    // std::cout << "Contact FR: " << get_contact(FR) << std::endl;
    // std::cout << "Contact FL: " << get_contact(FL) << std::endl;
    // std::cout << "Contact HR: " << get_contact(HR) << std::endl;
    // std::cout << "Contact HL: " << get_contact(HL) << std::endl;
  
    // compute feet jocobian
    Eigen::Matrix<double, 6, Eigen::Dynamic> J_fr_full, J_fl_full, J_hr_full, J_hl_full;
    J_fr_full.resize(6, ordered_encoders.size());
    J_fl_full.resize(6, ordered_encoders.size());
    J_hr_full.resize(6, ordered_encoders.size());
    J_hl_full.resize(6, ordered_encoders.size());
    J_fr_full.setZero();
    J_fl_full.setZero();
    J_hr_full.setZero();
    J_hl_full.setZero();


  
    pinocchio::computeFrameJacobian(model_, data_, ordered_encoders, frame_id_fr_, pinocchio::WORLD, J_fr_full);
    pinocchio::computeFrameJacobian(model_, data_, ordered_encoders, frame_id_fl_, pinocchio::WORLD, J_fl_full);
    pinocchio::computeFrameJacobian(model_, data_, ordered_encoders, frame_id_hr_, pinocchio::WORLD, J_hr_full);
    pinocchio::computeFrameJacobian(model_, data_, ordered_encoders, frame_id_hl_, pinocchio::WORLD, J_hl_full);
  
    // Extract the translation part (first 3 rows) and transform it to the body coordinate system
    Eigen::Matrix<double, 3, Eigen::Dynamic> J_fr = X_base.linear().transpose() * J_fr_full.block(0, 0, 3, ordered_encoders.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> J_fl = X_base.linear().transpose() * J_fl_full.block(0, 0, 3, ordered_encoders.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> J_hr = X_base.linear().transpose() * J_hr_full.block(0, 0, 3, ordered_encoders.size());
    Eigen::Matrix<double, 3, Eigen::Dynamic> J_hl = X_base.linear().transpose() * J_hl_full.block(0, 0, 3, ordered_encoders.size());
  
    // Store the Jacobian matrix into a separate area for each leg (NAPL column per leg)
    jacobians_.block(0, FR * NAPL, 3, NAPL) = J_fr.block(0, FR * NAPL, 3, NAPL);
    jacobians_.block(0, FL * NAPL, 3, NAPL) = J_fl.block(0, FL * NAPL, 3, NAPL);
    jacobians_.block(0, HR * NAPL, 3, NAPL) = J_hr.block(0, HR * NAPL, 3, NAPL);
    jacobians_.block(0, HL * NAPL, 3, NAPL) = J_hl.block(0, HL * NAPL, 3, NAPL);
  }
  
  // Calculate initial velocity
  const Eigen::Vector3d MiniCheetahKinematics::get_init_velocity(
      const Eigen::Vector3d& w) {
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  
    // if some legs in contact, calculate the velocity based on the leg
    if (this->get_contact(FR) == 1) {
      Eigen::Vector3d p_fr = positions_.col(FR);
      Eigen::Matrix<double, 3, Eigen::Dynamic> J_fr = jacobians_.block(0, FR * NAPL, 3, NAPL);
      velocity = -J_fr * d_encoders_ - lie_group::skew(w) * p_fr;
    } else if (this->get_contact(FL) == 1) {
      Eigen::Vector3d p_fl = positions_.col(FL);
      Eigen::Matrix<double, 3, Eigen::Dynamic> J_fl = jacobians_.block(0, FL * NAPL, 3, NAPL);
      velocity = -J_fl * d_encoders_ - lie_group::skew(w) * p_fl;
    } else if (this->get_contact(HR) == 1) {
      Eigen::Vector3d p_hr = positions_.col(HR);
      Eigen::Matrix<double, 3, Eigen::Dynamic> J_hr = jacobians_.block(0, HR * NAPL, 3, NAPL);
      velocity = -J_hr * d_encoders_ - lie_group::skew(w) * p_hr;
    } else if (this->get_contact(HL) == 1) {
      Eigen::Vector3d p_hl = positions_.col(HL);
      Eigen::Matrix<double, 3, Eigen::Dynamic> J_hl = jacobians_.block(0, HL * NAPL, 3, NAPL);
      velocity = -J_hl * d_encoders_ - lie_group::skew(w) * p_hl;
    }
    // std::cout << "Velocity: " << velocity << std::endl;
    return velocity;
    
  }
  
  int MiniCheetahKinematics::get_num_legs() {
    return NLEG;
  }
  
  }