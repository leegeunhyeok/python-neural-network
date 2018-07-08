# python-neural-network
- 파이썬으로 구현 된 신경망 기초 예제입니다.

# 의존 라이브러리 설치
본 예제는 `numpy`와 `scipy` 라이브러리를 사용합니다.

```bash
pip install numpy
pip install scipy
```

# 실행
- 실행 전 학습 데이터 및 테스트 데이터가 존재해야합니다.
- 아래 [MNIST 데이터](#mnist-데이터) 항목에서 확인 가능합니다.
```bash
python neural_network.py
```

# 결과
<img src="./result.png"><br>
위 결과는 60000개의 레코드 학습 및 10000개의 레코드를 테스트 한 결과입니다.
- 학습률(코드 내의 learning_rate)은 `0.1` 로 진행


# MNIST 데이터
자세한 내용은 [여기](https://github.com/leegeunhyeok/python-neural-network/blob/master/mnist_dataset/MNIST_DATA.md)에서 확인 가능합니다.

# 참고자료
도서: 신경망 첫걸음