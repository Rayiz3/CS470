"""프로펠러와 바퀴를 동시에 제어하는 Aviary"""

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class WheelPropAviary(CtrlAviary):
    """프로펠러와 바퀴를 동시에 제어하는 Aviary 클래스."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int=240,
                 ctrl_freq: int=30,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 output_folder="results"
                 ):
        """초기화 메서드."""
        super().__init__(
            drone_model=drone_model,
            num_drones=num_drones,
            neighbourhood_radius=neighbourhood_radius,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pyb_freq,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obstacles=obstacles,
            user_debug_gui=user_debug_gui,
            output_folder=output_folder
        )
        
        # 바퀴 관련 설정
        self.WHEEL_FORCE = 1.0
        self.wheel_joints = []
        
        # joint 이름으로 ID 찾기
        wheel_names = ["wheel_front_left_joint", "wheel_front_right_joint", 
                      "wheel_back_left_joint", "wheel_back_right_joint"]
        
        for i in range(p.getNumJoints(self.DRONE_IDS[0])):
            joint_info = p.getJointInfo(self.DRONE_IDS[0], i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name in wheel_names:
                self.wheel_joints.append(i)

    def step(self, action):
        """시뮬레이션 스텝 실행.
        
        Parameters
        ----------
        action : dict
            prop_action: (NUM_DRONES, 4)-shaped array of propeller RPMs
            wheel_action: (NUM_DRONES, 4)-shaped array of wheel velocities
        """
        # 프로펠러 제어 적용
        prop_action = action.get('prop_action', np.zeros((self.NUM_DRONES, 4)))
        
        # 바퀴 제어 적용
        if 'wheel_action' in action:
            for i in range(self.NUM_DRONES):
                for j, wheel_id in enumerate(self.wheel_joints):
                    p.setJointMotorControl2(
                        self.DRONE_IDS[i],
                        wheel_id,
                        p.VELOCITY_CONTROL,
                        targetVelocity=action['wheel_action'][i, j],
                    )
        
        # 기존 step 실행 (프로펠러 제어)
        obs, reward, done, truncated, info = super().step(prop_action)
        
        return obs, reward, done, truncated, info 