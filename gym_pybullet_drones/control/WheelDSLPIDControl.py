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
        self.P_COEFF_WHEEL = 0.5
        self.I_COEFF_WHEEL = 0.001
        self.D_COEFF_WHEEL = 0.1
        
        # x축 PID 오차값 저장
        self.last_x_error = 0
        self.integral_x_error = 0
        
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
        # 바퀴 제어 계산 (x축 이동)
        x_error = target_pos[0] - cur_pos[0]
        self.integral_x_error += x_error * control_timestep
        derivative_x_error = (x_error - self.last_x_error) / control_timestep
        self.last_x_error = x_error
        
        base_velocity = self.P_COEFF_WHEEL * x_error + \
                        self.I_COEFF_WHEEL * self.integral_x_error + \
                        self.D_COEFF_WHEEL * derivative_x_error
        
        wheel_velocities = np.array([
            base_velocity,    # 전좌
            base_velocity,   # 전우
            base_velocity,    # 후좌
            base_velocity    # 후우
        ])
        
        # 프로펠러 제어 계산 (높이 제어)
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
        
        return wheel_velocities, prop_rpms 