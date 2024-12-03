"""Script demonstrating ground movement with wheeled drones.

Example
-------
In a terminal, run as:

    $ python ground_movement.py

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.WheelDSLPIDControl import WheelDSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
import pybullet as p

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 10
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONE, 
        gui=DEFAULT_GUI, 
        record_video=DEFAULT_RECORD_VIDEO,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        plot=True,
        colab=DEFAULT_COLAB
    ):
    #### 시뮬레이션 초기화 #############################
    INIT_XYZS = np.array([[0.0, 0, 0.02]])  # 지면에서 2cm 위
    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     physics=Physics.PYB,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=False
                     )

    #### 바퀴 모터 설정 ###########################
    WHEEL_VELOCITY = 30.0
    WHEEL_FORCE = 1.0
    
    # joint 정보 가져오기
    num_joints = p.getNumJoints(env.DRONE_IDS[0])
    wheel_joints = []
    
    # joint 이름으로 ID 찾기
    wheel_names = ["wheel_front_left_joint", "wheel_front_right_joint", 
                  "wheel_back_left_joint", "wheel_back_right_joint"]
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(env.DRONE_IDS[0], i)
        joint_name = joint_info[1].decode('utf-8')
        if joint_name in wheel_names:
            wheel_joints.append(i)
    
    # 바퀴 모터 설정
    for wheel_id in wheel_joints:
        p.setJointMotorControl2(env.DRONE_IDS[0], 
                              wheel_id,
                              p.VELOCITY_CONTROL,
                              targetVelocity=WHEEL_VELOCITY,
                              force=WHEEL_FORCE)

    #### 경로 초기화 ###########################
    PERIOD = 5
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [10*i/NUM_WP, 0]  # x축으로 선형 이동
    wp_counters = 0

    #### 로거 초기화 #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                   num_drones=1,
                   duration_sec=duration_sec,
                   output_folder=output_folder,
                   colab=colab
                   )

    #### 컨트롤러 초기화 ############################
    ctrl = WheelDSLPIDControl(drone_model=drone)
    
    #### 초기 상태 가져오기 ########################
    obs, info = env.reset()
    
    #### 시뮬레이션 실행 ####################################
    prop_action = np.zeros((1,4))  # 프로펠러 RPM
    wheel_action = np.zeros((1,4))  # 바퀴 속도
    
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### 현재 상태 분해 #############
        state = obs[0]
        pos = state[0:3]
        print(pos)
        quat = np.array([state[6], state[3], state[4], state[5]])
        vel = state[10:13]
        ang_vel = state[13:16]
        
        #### 제어 계산 #############
        wheel_action[0, :], _ = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=np.hstack([TARGET_POS[wp_counters, :], INIT_XYZS[0, 2]])
        )
        
        # 바퀴 모터 제어 적용
        for j, wheel_id in enumerate(wheel_joints):
            p.setJointMotorControl2(
                env.DRONE_IDS[0],
                wheel_id,
                p.VELOCITY_CONTROL,
                targetVelocity=wheel_action[0, j],
                force=WHEEL_FORCE
            )

        #### 시뮬레이션 스텝 ###################################
        obs, reward, terminated, truncated, info = env.step(np.zeros((1, 4)))
        
        #### 다음 웨이포인트로 이동 #####################
        wp_counters = wp_counters + 1 if wp_counters < (NUM_WP-1) else 0

        #### 로깅 ####################################
        logger.log(drone=0,
                  timestamp=i/env.CTRL_FREQ,
                  state=obs[0],
                  control=np.hstack([TARGET_POS[wp_counters, :], INIT_XYZS[0, 2], np.zeros(9)])
                  )

        #### 렌더링 ##############################################
        env.render()

        #### 시뮬레이션 동기화 ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### 환경 종료 #################################
    env.close()

    #### 시뮬레이션 결과 저장 ###########################
    logger.save()
    logger.save_as_csv("ground_movement")

    #### 결과 플롯 ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### 인자 파싱 ##
    parser = argparse.ArgumentParser(description='Ground movement example script')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 10)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS)) 