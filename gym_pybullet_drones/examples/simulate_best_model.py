from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import os
import numpy as np

# 상대 경로를 사용해 모델 파일 경로 설정
current_dir = os.path.dirname(__file__)  # 현재 파일의 디렉터리
model_path = os.path.join(current_dir, "../results/save-11.24.2024_23.25.56/best_model.zip")  # 상대 경로로 모델 파일 위치 지정

# 저장된 모델 불러오기
print("Loading the trained model...")
model = PPO.load(model_path)

# 테스트 환경 생성
print("Initializing the test environment...")
test_env = HoverAviary(gui=True,  # GUI 활성화
                       obs=ObservationType.KIN,  # 관찰 타입: Kinematic
                       act=ActionType.VEL)  # 액션 타입: Velocity Control

# 환경 초기화
obs, _ = test_env.reset(seed=42)

# 테스트 루프 시작
print("Starting the test loop...")
for i in range(500):  # 테스트할 스텝 수
    # 학습된 모델을 사용해 행동 예측
    action, _ = model.predict(obs, deterministic=True)  # 결정론적 정책
    obs, reward, terminated, truncated, info = test_env.step(action)  # 환경 업데이트
    test_env.render()  # PyBullet 환경 시각화

    # 에피소드 종료 조건 처리
    if terminated or truncated:
        print(f"Episode terminated or truncated at step {i}. Resetting the environment...")
        obs, _ = test_env.reset(seed=42)

# 환경 종료
test_env.close()
print("Test completed successfully!")