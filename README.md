
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
- 구현 예정

### 5. Probability & Statistics
- 구현 예정

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
