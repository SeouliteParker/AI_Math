
# AI_Math

## AI가 어떻게 학습하는지 수학으로 직접 풀어보기

AI 모델의 학습 원리를 선형대수, 미적분, 역전파,
최적화, 확률·통계의 관점에서 직접 구현하고 검증하는 프로젝트입니다.

NumPy로 수학적 계산을 구현하고,
Matplotlib으로 변환 과정과 학습 경로를 시각화했습니다.

PyTorch, TensorFlow, JAX 등의 자동 미분 라이브러리는 사용하지 않았습니다.

## 1. 프로젝트 목표

- 행렬 변환과 행렬식의 기하학적 의미 이해
- Power Iteration으로 최대 고유값과 고유벡터 계산
- SVD를 활용한 이미지 압축 및 복원
- 중심차분법과 Gradient 계산
- 연쇄 법칙을 이용한 신경망 역전파 수식 유도
- Gradient Descent와 Momentum의 수렴 특성 비교
- 확률분포와 MSE, Cross-Entropy, MLE의 관계 이해

## 2. 개발 환경

| 항목 | 사용 환경 |
|---|---|
| Python | 3.9.6 |
| NumPy | 2.0.2 |
| Matplotlib | 3.9.4 |
| Seaborn | 0.13.2 |
| Jupyter Notebook | 7.5.8 |

재현 가능한 실험을 위해 난수가 사용되는 주요 실험에서
`np.random.seed(42)`를 적용했습니다.

## 3. 프로젝트 구조

```text
AI_Math/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── linear_algebra.py
│   ├── calculus.py
│   └── optimizer.py
├── notebooks/
│   ├── backprop_derivation.ipynb
│   └── probability_loss.ipynb
└── outputs/
    ├── matrix_transformations.png
    ├── svd_compression.png
    ├── gradient_contour.png
    ├── optimizer_comparison.png
    ├── learning_rate_experiment.png
    ├── elliptical_comparison.png
    ├── divergence_paths.png
    ├── normal_distributions.png
    └── bernoulli_distributions.png
```

## 4. 구현 내용

### 4.1 Linear Algebra — 선형대수

파일: `src/linear_algebra.py`

#### 주요 구현

- 단위 원에 회전 행렬 적용
- 스케일링 S(2, 0.5) 적용
- 전단(Shearing) 행렬 적용
- 행렬식과 도형 면적비 비교
- Power Iteration 직접 구현
- `np.linalg.eig`와 고유값 비교
- SVD 기반 이미지 압축 및 복원

#### 핵심 수학 원리

행렬 변환에 따른 면적 변화율은 행렬식의 절댓값과 같습니다.

```text
변환 후 면적 / 변환 전 면적 = |det(A)|
```

Power Iteration은 행렬과 벡터의 곱셈을 반복하여
절댓값이 가장 큰 고유값과 대응 고유벡터를 추정합니다.

SVD는 이미지를 특이값 기준으로 분해하고,
상위 k개 성분을 이용하여 근사 복원합니다.

#### 검증 결과

| 검증 항목 | 결과 |
|---|---|
| 회전 행렬 면적비 | PASS |
| 스케일링 면적비 | PASS |
| 전단 행렬 면적비 | PASS |
| 행렬식 대비 면적비 오차 1% 이내 | PASS |
| Power Iteration 고유값 오차 5% 이내 | PASS |
| SVD 이미지 복원 | 실행 성공 |

Power Iteration 검증 행렬:

```text
A = [[4, 1],
     [1, 3]]
```

- 계산된 최대 고유값: 약 4.618034
- NumPy 고유값: 약 4.618034
- 고유값 오차: 약 0%

SVD 복원 실험:

| 요청 k | 실제 사용 k | MSE |
|---|---|---|
| 10 | 10 | 약 0.000508 |
| 50 | 50 | 약 0.000008 |
| 100 | 64 | 약 0 |

참고: 실험 이미지가 64×64이므로 실제 사용 가능한
특이값은 최대 64개입니다. 따라서 k=100 요청 시
실제 복원에는 k=64를 사용했습니다.

#### 출력 파일

- `outputs/matrix_transformations.png`
- `outputs/svd_compression.png`

### 4.2 Calculus — 미적분

파일: `src/calculus.py`

#### 주요 구현

- 중심차분법을 이용한 수치 미분
- 해석적 미분과의 오차 비교
- 2변수 함수의 Gradient 계산
- 등고선과 Gradient 벡터 시각화

#### 핵심 수학 원리

수치 미분은 다음 중심차분법으로 계산합니다.

```text
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```

검증 함수:

```text
f(x) = x²
f'(x) = 2x
```

x=3에서 해석적 미분값은 6입니다.

2변수 함수:

```text
f(x,y) = x² + y²
Gradient = (2x, 2y)
```

Gradient는 함수값이 가장 빠르게 증가하는 방향을
가리키며, 등고선과 수직입니다.

#### 검증 결과

| 검증 항목 | 결과 |
|---|---|
| 수치 미분값 | 약 6.00000000 |
| 해석적 미분값 | 6.00000000 |
| 절대오차 1e-4 이내 | PASS |
| Gradient 등고선 시각화 | 완료 |

#### 출력 파일

- `outputs/gradient_contour.png`

### 4.3 Backpropagation — 역전파

파일: `notebooks/backprop_derivation.ipynb`

#### 신경망 구조

```text
입력층(2) → 은닉층(2) → 출력층(1)
```

고정 예제:

```text
x  = [1, 0]

W1 = [[0.1, 0.2],
      [0.3, 0.4]]

b1 = [0, 0]

W2 = [0.5, 0.6]
b2 = 0

y_true = 1
```

은닉층과 출력층에 Sigmoid를 적용하고,
손실 함수로 Binary Cross-Entropy(BCE)를 사용합니다.

#### 순전파 결과

| 변수 | Shape | 계산 결과 |
|---|---|---|
| z1 | (2,) | [0.1, 0.3] |
| a1 | (2,) | [0.5250, 0.5744] |
| z2 | scalar | 약 0.6072 |
| y_pred | scalar | 약 0.6473 |
| BCE Loss | scalar | 약 0.434958 |

#### 역전파 유도

연쇄 법칙(Chain Rule)을 적용하여 출력층에서
입력층 방향으로 다음 6개 기울기를 계산했습니다.

1. dL/dy_pred
2. dL/dz2
3. dL/dW2
4. dL/da1
5. dL/dz1
6. dL/dW1

#### 검증 결과

- 순전파 중간값 계산 완료
- 6개 기울기 수식 유도 완료
- 각 기울기의 Shape 확인 완료
- 손계산 값과 NumPy 결과 비교: PASS
- 소수점 넷째 자리 검증: PASS

참고: 미션 예시에 제시된 일부 기울기 값은
주어진 가중치로 직접 계산한 결과와 차이가 있습니다.

특히 `dL/dz1`의 두 번째 값은 예시의 -0.0513이 아닌
약 -0.0517이며, Notebook에 계산 근거를 기록했습니다.

### 4.4 Optimization — 최적화

파일: `src/optimizer.py`

#### 주요 구현

- Vanilla Gradient Descent 클래스
- Momentum 클래스(beta=0.9)
- 경사하강법 수렴 검증
- 두 옵티마이저의 수렴 경로 비교
- Learning Rate별 수렴·진동·발산 실험
- 타원형 목적함수에서 Momentum 효과 검증
- 등고선 위 Learning Rate별 이동 경로 시각화

#### 실험 함수 1

```text
f(x,y) = x² + y²
Gradient = (2x, 2y)
```

초기점 (5,5), lr=0.1, 100회 반복 조건에서
Vanilla GD가 원점 반경 0.1 이내로 수렴하는지 검증했습니다.

#### 주요 실험 결과

| 항목 | 결과 |
|---|---|
| Vanilla GD, lr=0.1, 100회 | PASS |
| Vanilla GD 최종 원점 거리 | 약 1.44e-9 |
| Momentum, lr=0.1, 100회 원점 거리 | 약 0.02016 |
| 두 옵티마이저 수렴 경로 시각화 | 완료 |

#### Learning Rate별 수렴 및 발산

목적함수 `f(x,y)=x²+y²`에서 실험했습니다.

| Learning Rate | 관찰 결과 |
|---|---|
| 0.1 | 수렴 |
| 0.5 | 원점 도달 |
| 1.0 | 두 점 사이 진동 |
| 1.2 | 발산 |

미션의 "lr=0.5 이상에서 발산"이라는 표현은
이 목적함수에서는 수학적으로 성립하지 않습니다.

실제 Gradient Descent 갱신식은 다음과 같습니다.

```text
x_next = (1 - 2lr) * x
y_next = (1 - 2lr) * y
```

따라서 이 함수에서는 다음과 같이 구분됩니다.

- 0 < lr < 1: 수렴
- lr = 1: 진동
- lr > 1: 발산

추가로 초기점 (1,1), 8회 반복을 사용하여
lr=0.5, 1.0, 1.2의 이동 경로를 등고선 위에
직접 비교하는 그림을 생성했습니다.

#### 실험 함수 2

```text
f(x,y) = x² + 10y²
Gradient = (2x, 20y)
```

lr=0.01, 100회 반복 결과:

| 옵티마이저 | 최종 Loss |
|---|---:|
| Vanilla GD | 0.439699 |
| Momentum | 0.002480 |

이 조건에서 Momentum의 최종 손실이
Vanilla GD보다 약 177배 작았습니다.

다만 Momentum이 모든 학습률에서
항상 더 좋은 성능을 보이는 것은 아닙니다.

#### 출력 파일

- `outputs/optimizer_comparison.png`
- `outputs/learning_rate_experiment.png`
- `outputs/elliptical_comparison.png`
- `outputs/divergence_paths.png`

### 4.5 Probability & Statistics — 확률·통계

파일: `notebooks/probability_loss.ipynb`

#### 주요 구현

- 정규분포 PDF 시각화
- 베르누이 분포 PMF 시각화
- Softmax 함수 직접 구현
- Softmax 출력 확률 합계 검증
- MSE와 MLE의 관계 수식 유도
- Binary Cross-Entropy와 MLE의 관계 수식 유도
- Categorical Cross-Entropy와 MLE의 관계 설명

#### 확률분포 실험

정규분포:

- N(0,1)
- N(2,0.5)

여기서 두 번째 매개변수는 분산으로 사용했습니다.

베르누이 분포:

- B(0.3)
- B(0.7)

#### Softmax 검증

입력:

```text
[2.0, 1.0, 0.1]
```

출력:

```text
[0.65900114, 0.24243297, 0.09856589]
```

검증 결과:

```text
Probability sum: 1.0000000000
Error: 0.0000000000
Validation: PASS
```

#### 손실 함수와 최대우도추정(MLE)

정규분포를 따르는 독립 오차와 고정 분산을
가정하면 MSE를 최소화하는 것은
최대우도추정과 동일한 최적해를 구합니다.

이진 분류에서 Bernoulli 분포를 가정하면
Binary Cross-Entropy 최소화는
음의 로그우도 최소화와 연결됩니다.

다중 분류에서는 Categorical 분포와
Categorical Cross-Entropy가 같은 관계를 가집니다.

수식 유도 과정은 Notebook에 기록했습니다.

#### 출력 파일

- `outputs/normal_distributions.png`
- `outputs/bernoulli_distributions.png`

## 5. 실행 방법

명령어는 프로젝트 루트 디렉터리에서 실행합니다.

### 5.1 가상환경 활성화

```bash
source .venv/bin/activate
```

### 5.2 라이브러리 설치

새 환경에서 실행할 때:

```bash
python -m pip install -r requirements.txt
```

기존 가상환경에 필요한 라이브러리가 설치되어 있다면
다시 설치할 필요는 없습니다.

### 5.3 선형대수

```bash
python src/linear_algebra.py
```

### 5.4 미적분

```bash
python src/calculus.py
```

### 5.5 최적화

```bash
python src/optimizer.py
```

### 5.6 Jupyter Notebook

```bash
python -m notebook
```

브라우저에서 다음 파일을 실행합니다.

- `notebooks/backprop_derivation.ipynb`
- `notebooks/probability_loss.ipynb`

각 Notebook의 셀을 순서대로 실행하여
계산 결과와 수식을 확인할 수 있습니다.

## 6. 검증 요약

| 영역 | 핵심 검증 | 결과 |
|---|---|---|
| 선형대수 | 행렬식과 면적비 오차 1% 이내 | PASS |
| 선형대수 | Power Iteration 오차 5% 이내 | PASS |
| 선형대수 | SVD 이미지 복원 | 완료 |
| 미적분 | 수치 미분 절대오차 1e-4 이내 | PASS |
| 역전파 | 6개 기울기 소수점 4자리 비교 | PASS |
| 최적화 | GD 100회 후 반경 0.1 이내 | PASS |
| 최적화 | Learning Rate별 경로 비교 | 완료 |
| 최적화 | Momentum 비교 실험 | 완료 |
| 확률·통계 | Softmax 합계 오차 1e-6 이내 | PASS |
| 확률·통계 | 손실 함수와 MLE 관계 유도 | 완료 |

## 7. 구현 제약 준수

- NumPy로 수학적 계산 구현
- Matplotlib으로 실험 결과 시각화
- `np.linalg.eig`는 Power Iteration 검증에만 사용
- PyTorch, TensorFlow, JAX 미사용
- scikit-learn의 PCA 및 최적화 함수 미사용
- 함수 및 클래스에 Docstring 작성
- 수식은 코드 주석 또는 Notebook Markdown에 기록
- 의존성 버전은 `requirements.txt`에 명시
- 실험 시각화 결과는 `outputs/`에 PNG로 저장

## 8. 참고 사항

### SVD의 k=100 제한

실험 이미지가 64×64이므로 SVD 복원에서
실제 사용 가능한 최대 k는 64입니다.

요청값 k=100은 실제 k=64로 제한하며,
출력 그림에도 이를 표시합니다.

### Learning Rate와 발산 조건

`f(x,y)=x²+y²`에 Vanilla GD를 적용하면
lr=0.5에서는 원점에 도달하고
lr=1.0에서는 진동하며
lr>1.0에서 발산합니다.

이는 미션의 참고 설명보다
수학적으로 정확한 결과입니다.

### 역전파 계산값

미션 참고 예시의 -0.0513과 달리
주어진 가중치와 활성화 함수로 계산하면
해당 기울기는 약 -0.0517입니다.

실제 계산 결과와 검증 내용은
`backprop_derivation.ipynb`에 기록했습니다.

## 9. 추가 학습 과제

다음 항목은 보너스 과제이며 현재 구현 범위에서 제외했습니다.

- Adam Optimizer
- Newton Method
- Entropy, KL-Divergence, Cross-Entropy 직접 구현

## 10. 결론

이번 프로젝트에서는 AI 학습에 사용되는 핵심 수학을
직접 계산하고 시각화하여 검증했습니다.

행렬 변환, 미분, 연쇄 법칙, 경사하강법,
확률분포와 손실 함수의 관계를
실제 NumPy 코드와 수학적 수식으로 연결했습니다.

이를 통해 이후 AutoGrad 및 딥러닝 모델 구현에
필요한 수학적 기초를 학습했습니다.