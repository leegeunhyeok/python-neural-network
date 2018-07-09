# 테스트 데이터 CSV 파일 불러오기
test_data_file = open("mnist_dataset/mnist_test_10.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

for test in test_data_list:
    c = 1
    test = test.split(",")
    for w in range(28):
        for h in range(28):
            if (int(test[c]) > 150):
                print("##", end="")
            else:
                print("  ", end="")
            c += 1
        print("")
    print(" =", test[0], "\n\n")