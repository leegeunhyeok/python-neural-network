import numpy
import scipy.special
import time

# 신경망 클래스
class neural_network:
    # 신경망 초기화
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        print("신경망 인스턴스 초기화\n입력 노드 수: {}\n은닉 노드 수: {}\n출력 노드 수: {}\n학습률: {}".format(input_nodes, hidden_nodes, output_nodes, learning_rate))
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        self.learning_rate = learning_rate
        
        # 가중치 행렬
        # 정규분포 중심은 0.0으로 설정
        # 행렬 크기: 은닉계층 노드 수 * 입력계층 노드 수
        self.w_in_hidden = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        # 행렬 크기: 출력계층 노드 수 * 은닉계층 노드 수
        self.w_hidden_out = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))
        
        # 활성화 함수는 시그모이드 함수 사용
        self.activation_function = lambda x: scipy.special.expit(x)
    
    # 신경망 학습 시키기
    def train(self, inputs_list, targets_list):
        # 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # 들어오는 신호 계산
        hidden_inputs = numpy.dot(self.w_in_hidden, inputs)
        # 은닉 계층에서 나가는 신호 연산
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어갈 신호 연산
        final_inputs = numpy.dot(self.w_hidden_out, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호 연산
        final_outputs = self.activation_function(final_inputs)
        
        # 오차(E) 계산
        output_errors = targets - final_outputs
        # 가중치에 의해 나뉜 출력 계층의 오차를 재조합하여 계산
        hidden_errors = numpy.dot(self.w_hidden_out.T, output_errors)
        
        # 가중치 업데이트
        self.w_hidden_out += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.w_in_hidden += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
    
    # 신경망에 질의하기
    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # 들어오는 신호 계산
        hidden_inputs = numpy.dot(self.w_in_hidden, inputs)
        # 은닉 계층에서 나가는 신호 연산
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 최종 출력 계층으로 들어갈 신호 연산
        final_inputs = numpy.dot(self.w_hidden_out, hidden_outputs)
        # 최종 출력 계층에서 나가는 신호 연산
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

# 입력 노드의 수(이미지 28*28 = 784 픽셀)
input_nodes = 784

# 은닉 노드의 수
hidden_nodes = 200

# 출력 노드의 수(결과는 0 ~ 9의 범위를 가짐)
output_nodes = 10

# 학습률
learning_rate = 0.1

# 신경망 인스턴스 생성
n = neural_network(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 학습 데이터 CSV 파일 불러오기
training_data_file = open("mnist_dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

print("총 {}개의 레이블 학습을 시작합니다.".format(len(training_data_list)))
# 학습 시작 시간
start_time = time.time()

for record in training_data_list:
    # 쉼표를 기준으로 분리
    all_values = record.split(",")
    
    # 입력값의 범위 및 값 조정 (0.01 ~ 1.00)
    # 색상 값은 0 ~ 255 이므로 (데이터/255 = 0 ~ 1)
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # 결과값 생성
    targets = numpy.zeros(output_nodes) + 0.01
    
    # 레코드에 대한 결과 값
    # all_values에 레코드의 레이블과 28*28 크기의 손글씨 이미지 데이터가 있음 (이미지 784 + 레이블 1 = 785)
    # 맨 앞 (인덱스 0) 데이터는 레이블(결과), 결과 데이터는 0.99로 설정
    # 아래 예는 해당 데이터의 결과가 3인 경우의 target 데이터 (목표)
    # [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    targets[int(all_values[0])] = 0.99
    
    # 신경망 학습
    n.train(inputs, targets)

print("학습이 완료되었습니다 (소요시간: {}초)".format(round(time.time() - start_time), 3))

# 테스트 데이터 CSV 파일 불러오기
test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# 신경망 성적 (정확도)
scorecard = []

print("총 {}개의 레이블에 대해 테스트를 시작합니다.".format(len(test_data_list)))
# 테스트 시작 시간
start_time = time.time()

for record in test_data_list:
    # 데이터를 쉼표 기준으로 분리
    all_values = record.split(",")
    
    # 레이블(정답)은 데이터의 첫 번째에 있음, 나머지는 이미지 데이터
    correct_label = int(all_values[0])
    
    # 입력 데이터 조절
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    # 신경망에 질의
    output = n.query(inputs)
    
    # 가장 큰 값의 인덱스
    label = numpy.argmax(output)
    
    # 테스트 데이터와 일치(정답)할 경우 성적표에 1추가
    if (label == correct_label):
        scorecard.append(1)
    else:
        # 틀렸을 경우 0 추가
        scorecard.append(0)

print("테스트가 완료되었습니다 (소요시간: {}초)".format(round(time.time() - start_time), 3))

scorecard_array = numpy.asarray(scorecard)
print("정확도: {}%".format(scorecard_array.sum() / scorecard_array.size * 100))