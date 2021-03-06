# 학습, 테스트 MNIST 데이터
구현된 신경망은 CSV 파일을 통해 학습 및 테스트를 진행합니다. <br>
데이터 파일은 다운로드 후 현재 폴더에 저장하시면 됩니다. 

# 사용된 데이터의 레코드 형식
- CSV 데이터 각 행의 1번 째 `숫자(*)`는 해당 이미지 데이터의 레이블(정답)입니다.
- CSV 데이터 각 행의 2번 째부터 `784개의 문자(-)`는 레코드의 손글씨 이미지 데이터입니다.
- 이미지 데이터를 시각화해보면 `28x28 크기의 이미지(784픽셀)` 입니다.
- 각 픽셀 데이터는 `0 ~ 255` 의 범위를 가지고 있습니다.
```bash
*,-,-,-,...,-
*,-,-,-,...,-
*,-,-,-,...,-

# 각각 숫자 5와 0에 대한 예시 입니다.
5,0,0,0,230,255,0,...,0 
0,0,136,177,0,0,...,0
```

# MNIST 데이터 다운로드
아래 링크를 통해 다운로드 받아 사용하거나, 레코드 형식에 맞는 데이터셋을 사용하시면 됩니다.
### 학습 데이터 (Training data)
- [학습 데이터 (60000 레코드)](http://www.pjreddie.com/media/files/mnist_train.csv)
- [학습 데이터 (100 레코드)](https://git.io/vySZ1)

### 테스트 데이터 (Test data)
- [테스트 데이터 (10000 레코드)](http://www.pjreddie.com/media/files/mnist_test.csv)
- [테스트 데이터 (10 레코드)](https://git.io/vySZP)

# 참고자료
도서: 신경망 첫걸음