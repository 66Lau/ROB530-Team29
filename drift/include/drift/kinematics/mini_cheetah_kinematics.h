#ifndef KINEMATICS_MC_KIN_H
#define KINEMATICS_MC_KIN_H

#include <pinocchio/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <Eigen/Dense>
#include <string>

#include "drift/math/lie_group.h"
#include "drift/measurement/legged_kinematics.h"

namespace mini_cheetah_kinematics {
enum Leg { FR, FL, HR, HL };
}

using namespace mini_cheetah_kinematics;
using namespace math;

namespace measurement::kinematics {

/**
 * @class MiniCheetahKinematics
 * @brief Mini Cheetah kinematics solver
 *
 * Using Pinocchio compute kinematics and Jacobian
 * p_Body_to_FrontRightFoot、Jp_Body_to_HindLeftFoot。
 */
class MiniCheetahKinematics : public LeggedKinematicsMeasurement {
 public:
  /**
   * @brief 
   */
  MiniCheetahKinematics();

  /**
   * @brief
   * @param[in] encoders joint angle（12×1）
   * @param[in] d_encoders joint vel（12×1）
   * @param[in] contacts contact（4×1 bool）
   */
  MiniCheetahKinematics(
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& encoders,
      const Eigen::Matrix<double, Eigen::Dynamic, 1>& d_encoders,
      const Eigen::Matrix<bool, Eigen::Dynamic, 1>& contacts);

  /**
   * @brief compute jacobian
   */
  void ComputeKinematics() override;

  /**
   * @brief 
   * @return
   */
  int get_num_legs() override;

  /**
   * @brief 
   * @param[in] w 
   * @return 
   */
  const Eigen::Vector3d get_init_velocity(const Eigen::Vector3d& w) override;

 private:
  pinocchio::Model model_;
  pinocchio::Data data_;

  int body_frame_id_;

  int frame_id_fr_;
  int frame_id_fl_;
  int frame_id_hr_;
  int frame_id_hl_;

  std::string urdf_path_;
  std::string foot_name_fr_;
  std::string foot_name_fl_;
  std::string foot_name_hr_;
  std::string foot_name_hl_;

  /**
   * @brief 
   */
  void initializePinocchio();
};

} // namespace measurement::kinematics

#endif // KINEMATICS_MC_KIN_H
