
# AI_Math

AI 학습의 수학적 원리를 NumPy와 Matplotlib으로
직접 구현하고 검증하는 프로젝트입니다.

## 개발 환경
- Python 3.9.6
- NumPy 2.0.2
- Matplotlib 3.9.4
- Seaborn 0.13.2

## 구현 현황

### 1. Linear Algebra
- Rotation, Scaling, Shearing 시각화
- 행렬식과 면적비 검증 (오차 1% 이내)
- Power Iteration 구현 (오차 5% 이내)
- SVD 이미지 압축 및 복원

### 2. Calculus
- 구현 예정

### 3. Backpropagation
- 구현 예정

### 4. Optimization

### 4. Optimization
- Vanilla Gradient Descent 구현
- Momentum(beta=0.9) 구현
- 학습률별 수렴 및 발산 실험
- 원형 함수 등고선 위 수렴 경로 비교
- 타원형 함수에서 Momentum 효과 검증

#### 주요 실험 결과
- Vanilla GD (lr=0.1, 100회): PASS
- lr=0.5: 원점 도달
- lr=1.0: 진동
- lr=1.2: 발산
- 타원형 함수 (lr=0.01, 100회):
  - Vanilla GD Loss: 0.439699
  - Momentum Loss: 0.002480
  - Momentum의 최종 손실 약 177배 감소

#### 출력 파일
- outputs/optimizer_comparison.png
- outputs/learning_rate_experiment.png
- outputs/elliptical_comparison.png

### 5. Probability & Statistics

### 5. Probability & Statistics

- 정규분포 N(0,1), N(2,0.5) PDF 시각화
- 베르누이 분포 B(0.3), B(0.7) PMF 시각화
- Softmax 함수 직접 구현
- Softmax 출력 확률 합계 검증 (오차 1e-6 이내)
- 정규분포 가정에서 MSE와 MLE의 관계 수식 유도
- 베르누이 분포에서 BCE와 MLE의 관계 수식 유도
- 카테고리 분포에서 Cross-Entropy와 MLE 관계 설명

#### 출력 파일
- outputs/normal_distributions.png
- outputs/bernoulli_distributions.png

#### Notebook
- notebooks/probability_loss.ipynb

## 실행 방법

가상환경 활성화:

    source .venv/bin/activate

선형대수 실험 실행:

    python src/linear_algebra.py

## 검증 결과

- Matrix Transformation: PASS
- Area Ratio Validation: PASS
- Power Iteration: PASS
- SVD Reconstruction: 실행 성공

## SVD 참고사항

64x64 이미지에서는 최대 rank가 64이므로,
k=100 요청 시 실제로는 k=64를 사용합니다.

## 출력 파일

- outputs/matrix_transformations.png
- outputs/svd_compression.png
