import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import time

# PyBullet 초기화
p.connect(p.GUI)  # PyBullet GUI를 활성화하여 시뮬레이션을 시각적으로 확인할 수 있도록 설정
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # PyBullet에서 기본 데이터 경로 설정

# 로봇 및 환경 설정
plane_id = p.loadURDF("plane.urdf")  # 평면(기본 지형)을 로드

urdf_path = os.path.join(os.path.dirname(__file__), "../assets/wall1.urdf")
p.loadURDF(urdf_path, useFixedBase=True)

class Node:
    def __init__(self, position, cost=0, parent=None):
        self.position = np.array(position)
        self.cost = cost  # 시작점부터 해당 노드까지의 비용
        self.parent = parent

def distance(a, b):
    return np.linalg.norm(a - b)  # 두 지점 a와 b 간의 유클리드 거리 계산

def is_collision(a, b):
    # 충돌 검사 (단순화된 예시)
    ray = p.rayTest(a, b)  # PyBullet의 rayTest를 사용하여 a에서 b로의 경로 상 충돌 여부 확인
    for hit in ray:
        if hit[0] != -1:  # 충돌된 객체가 있다면 True 반환
            return True
    return False  # 충돌이 없으면 False 반환
"""
def rrt_star(start, goal, max_iter=100000, step_size=0.1, neighbor_radius=0.1):
    tree = [Node(start, cost=0)]  # 시작 노드 초기화
    
    for i in range(max_iter):
        print(i)
        # 무작위 점 생성
        rand_point = np.array([random.uniform(0, 1), random.uniform(-1, 1), random.uniform(0, 1)])
        
        # 가장 가까운 노드 찾기
        nearest_node = min(tree, key=lambda node: distance(node.position, rand_point))
        direction = rand_point - nearest_node.position
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        direction = (direction / length) * step_size
        new_position = nearest_node.position + direction
        
        # 충돌 없는 경우 새 노드 추가
        if not is_collision(nearest_node.position, new_position):
            new_cost = nearest_node.cost + distance(nearest_node.position, new_position)
            new_node = Node(new_position, cost=new_cost, parent=nearest_node)
            tree.append(new_node)
            
            # Rewiring: 근처 노드들 탐색하여 비용 줄이기
            for other_node in tree:
                if distance(other_node.position, new_node.position) < neighbor_radius:
                    potential_cost = new_node.cost + distance(new_node.position, other_node.position)
                    if potential_cost < other_node.cost and not is_collision(new_node.position, other_node.position):
                        other_node.parent = new_node
                        other_node.cost = potential_cost
            
            # 목표에 도달한 경우 경로 반환
            if distance(new_node.position, goal) < step_size:
                path = [new_node.position]
                node = new_node
                while node.parent is not None:
                    node = node.parent
                    path.append(node.position)
                return path[::-1]
    return None
"""
def rrt(start, goal, max_iter=1000000, step_size=0.1):
    """
    RRT 알고리즘을 사용하여 시작점(start)에서 목표(goal)까지의 경로를 찾음.

    Parameters:
    - start: 시작 위치 (numpy array)
    - goal: 목표 위치 (numpy array)
    - max_iter: 최대 반복 횟수
    - step_size: 한 번의 이동 거리
    
    Returns:
    - path: 충돌 없는 경로 (목표에 도달 시), 도달 실패 시 None
    """
    tree = [Node(start)]  # 시작 노드를 트리에 추가
    for i in range(max_iter):
        print(i)
        # 무작위 점(rand_point)을 생성 (환경 범위 내에서 랜덤 샘플링)
        rand_point = np.array([random.uniform(0, 1), random.uniform(-1, 1), random.uniform(0,1)])
        
        # 트리 내에서 가장 가까운 노드를 찾음
        nearest_node = min(tree, key=lambda node: distance(node.position, rand_point))
        
        # 가까운 노드에서 무작위 점을 향하는 방향으로 한 단계 이동
        direction = rand_point - nearest_node.position
        length = np.linalg.norm(direction)  # 방향 벡터의 길이
        if length == 0:  # 무작위 점이 현재 노드와 동일한 경우 생략
            continue
        direction = (direction / length) * step_size  # 방향 벡터를 단위 벡터로 변환 후 step_size 크기로 확장
        new_position = nearest_node.position + direction  # 새 위치 계산

        # 새 위치로의 이동이 충돌하지 않는 경우
        if not is_collision(nearest_node.position, new_position):
            new_node = Node(new_position, 0, nearest_node)  # 새 노드를 트리에 추가
            tree.append(new_node)
            
            # 새 노드가 목표에 충분히 가까운 경우
            if distance(new_node.position, goal) < step_size:
                # 목표에 도달한 경로를 생성
                path = [new_node.position]
                node = new_node
                while node.parent is not None:  # 부모 노드를 따라가며 경로를 역으로 추적
                    node = node.parent
                    path.append(node.position)
                return path[::-1]  # 경로를 반대로 뒤집어 올바른 순서로 반환
    return None  # 목표에 도달하지 못한 경우

# 시작 및 목표 지점 설정
start = np.array([0, 0, 0])  # 로봇의 초기 위치
goal = np.array([1, 0, 0.25])  # 로봇이 도달해야 할 목표 위치

# RRT 알고리즘으로 경로 계산
path = rrt(start, goal)  # 시작점에서 목표점까지의 경로를 탐색

if path is not None:
    print("Generated Path:", path)
    print(f"Path Length: {len(path)}")
else:
    print("No path found.")

if path is not None:
    # 경로를 시각적으로 표시 (선으로 연결)
    for i in range(len(path) - 1):
        p.addUserDebugLine(path[i], path[i + 1], lineColorRGB=[0, 1, 1], lineWidth=2.0)
else:
    print("경로를 찾을 수 없습니다.")

time.sleep(100)

p.disconnect()  # PyBullet 시뮬레이션 종료