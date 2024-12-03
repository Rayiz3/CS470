"""바퀴가 달린 드론을 위한 PID 컨트롤러"""

import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel

class WheelDSLPIDControl(DSLPIDControl):
    """바퀴 달린 Crazyflie를 위한 PID 컨트롤 클래스."""
    
    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """초기화 메서드."""
        super().__init__(drone_model=drone_model, g=g)
        
        # 바퀴 제어를 위한 PID 게인
        self.P_COEFF_WHEEL = 50
        self.I_COEFF_WHEEL = 0.1
        self.D_COEFF_WHEEL = 10
        
        # x축 PID 오차값 저장
        self.last_x_error = 0
        self.integral_x_error = 0
        
        # 바닥 감지를 위한 임계값 추가
        self.GROUND_THRESHOLD = 0.1  # 바닥으로부터의 높이 임계값 (미터)
        self.Z_MOVEMENT_THRESHOLD = 0.05  # z축 이동 감지 임계값
        
    def computeControl(self,
                      control_timestep,
                      cur_pos,
                      cur_quat,
                      cur_vel,
                      cur_ang_vel,
                      target_pos,
                      target_rpy=np.zeros(3),
                      target_vel=np.zeros(3),
                      target_rpy_rates=np.zeros(3)
                      ):
        """제어 입력 계산."""
        # 현재 상태 분석
        is_near_ground = cur_pos[2] < self.GROUND_THRESHOLD
        z_movement_required = abs(target_pos[2] - cur_pos[2]) > self.Z_MOVEMENT_THRESHOLD

        # 바퀴 제어 계산 (x축 이동)
        wheel_velocities = np.zeros(4)
        if is_near_ground and not z_movement_required:
            x_error = target_pos[0] - cur_pos[0]
            self.integral_x_error += x_error * control_timestep
            derivative_x_error = (x_error - self.last_x_error) / control_timestep
            self.last_x_error = x_error
            
            base_velocity = self.P_COEFF_WHEEL * x_error + \
                            self.I_COEFF_WHEEL * self.integral_x_error + \
                            self.D_COEFF_WHEEL * derivative_x_error
            
            wheel_velocities = np.array([base_velocity] * 4)
        
        # 프로펠러 제어 계산
        prop_rpms = np.zeros(4)
        if not is_near_ground or z_movement_required:
            prop_rpms, _, _ = super().computeControl(
                control_timestep=control_timestep,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_pos=target_pos,
                target_rpy=target_rpy,
                target_vel=target_vel,
                target_rpy_rates=target_rpy_rates
            )
        prop_rpms = prop_rpms
        
        return wheel_velocities, prop_rpms

    def computeFromStates(self,
                         control_timestep,
                         state,
                         target_pos,
                         target_rpy=np.zeros(3),
                         target_vel=np.zeros(3),
                         target_rpy_rates=np.zeros(3)
                         ):
        """상태 벡터로부터 제어 입력 계산."""
        # 상태 벡터 분해
        pos = state[0:3]
        quat = np.array([state[6], state[3], state[4], state[5]])
        vel = state[10:13]
        ang_vel = state[13:16]
        
        # computeControl 호출
        wheel_velocities, prop_rpms = self.computeControl(
            control_timestep=control_timestep,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=target_pos,
            target_rpy=target_rpy,
            target_vel=target_vel,
            target_rpy_rates=target_rpy_rates
        )
        
        return wheel_velocities, prop_rpms