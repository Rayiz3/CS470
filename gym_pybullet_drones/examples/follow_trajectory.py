"""Script demonstrating trajectory following with wheeled drones.

Example
-------
In a terminal, run as:

    $ python follow_trajectory.py

"""
import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.WheelPropAviary import WheelPropAviary
from gym_pybullet_drones.control.WheelDSLPIDControl import WheelDSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger

DEFAULT_DRONE = DroneModel('cf2x')
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
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
    INIT_XYZS = np.array([[0.0, 0, 0.05]])  # 지면에서 5cm 위
    env = WheelPropAviary(
        drone_model=drone,
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

    #### 궤적 생성 ###########################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    
    # 단계별 시간 설정
    GROUND_PHASE = 0.4  # 처음 40%는 지상 이동
    HOVER_PHASE = 0.1   # 10%는 제자리 호버링
    ASCEND_PHASE = 0.2  # 20%는 상승
    
    ground_idx = int(NUM_WP * GROUND_PHASE)
    hover_idx = int(NUM_WP * (GROUND_PHASE + HOVER_PHASE))
    ascend_idx = int(NUM_WP * (GROUND_PHASE + HOVER_PHASE + ASCEND_PHASE))
    
    # X 좌표: 지상 이동 구간에서만 이동
    ground_x = np.linspace(0, 5, ground_idx)  # 5m 전진
    TARGET_POS[:ground_idx, 0] = ground_x
    TARGET_POS[ground_idx:, 0] = ground_x[-1]  # 마지막 위치 유지
    
    # Y 좌표: 고정
    TARGET_POS[:, 1] = INIT_XYZS[0, 1]
    
    # Z 좌표: 단계별 설정
    TARGET_POS[:hover_idx, 2] = 0.05  # 지상 이동 및 호버링 (5cm 높이)
    
    # 상승 구간: 사인 곡선으로 부드럽게
    ascend_t = np.linspace(0, 1, ascend_idx - hover_idx)
    TARGET_POS[hover_idx:ascend_idx, 2] = 0.05 + 1.95 * (1 - np.cos(ascend_t * np.pi/2))
    
    # 공중 구간: 2m 높이 유지
    TARGET_POS[ascend_idx:, 2] = 2.0

    wp_counters = 0

    #### 로거 초기화 #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
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
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        #### 현재 상태 분해 #############
        state = obs[0]
        pos = state[0:3]
        quat = np.array([state[6], state[3], state[4], state[5]])
        vel = state[10:13]
        ang_vel = state[13:16]
        
        #### 제어 계산 #############
        wheel_velocities, prop_rpms = ctrl.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=ang_vel,
            target_pos=TARGET_POS[wp_counters, :]
        )

        wheel_action = np.zeros((1,4))
        prop_action = np.zeros((1,4))
        wheel_action[0, :] = wheel_velocities
        prop_action[0, :] = prop_rpms

        action = {
            'wheel_action': wheel_action,
            'prop_action': prop_action
        }
        print(action)
        
        #### 시뮬레이션 스텝 ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        #### 다음 웨이포인트로 이동 #####################
        wp_counters = wp_counters + 1 if wp_counters < (NUM_WP-1) else 0

        #### 로깅 ####################################
        logger.log(
            drone=0,
            timestamp=i/env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([TARGET_POS[wp_counters, :], np.zeros(9)])
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
    logger.save_as_csv("follow_trajectory")

    #### 결과 플롯 ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### 인자 파싱 ##
    parser = argparse.ArgumentParser(description='Trajectory following example script')
    parser.add_argument('--drone',              default=DEFAULT_DRONE,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 20)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS)) 