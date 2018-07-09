import sys
import os
import base64
import numpy
from io import BytesIO
from flask import Flask, request, render_template, json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import neural_network
n = None
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("./index.html")

@app.route("/query", methods=["POST"])
def query():
    if request.method == "POST":
        # Base64 해시 데이터를 디코딩하여 열기
        img = Image.open(BytesIO(base64.b64decode(request.json["data"])))

        # 이미지 크기 28 x 28 크기로 변환 + 흑백 변환
        img = img.resize((28, 28), Image.LANCZOS).convert(mode="L")

        # 이미지 픽셀
        img_pixel = img.load()

        # 픽셀 데이터
        pixel = []
        width, height = img.size
        for w in range(width):
            for h in range(height):
                # 흰 배경 영역은 0으로 설정
                pixel.append(255 - img_pixel[h, w])

        # 신경망 질의할 데이터로 변환
        inputs = (numpy.asfarray(pixel) / 255.0 * 0.99) + 0.01
        print(inputs)
        c = 0
        print("+", "--" * 28, end="+\n", sep="")
        for w in range(28):
            print("|", end="")
            for h in range(28):
                if (inputs[c] == 0.01):
                    print("  ", end="")
                else:
                    print("##", end="")
                c += 1
            print("|")
        print("+", "--" * 28, end="+\n", sep="")

        # 변환된 이미지 저장
        img.save('convert.png')

        # 이미지 닫기
        img.close()

        # 신경망에 이미지 질의
        result = numpy.argmax(n.query(inputs))
        print("결과:", result)

        return json.dumps({"result": int(result)})
    else:
        return json.dumps({"result": -1})

if __name__ == "__main__":
    print("신경망 학습이 끝난 후 서버가 실행 됩니다.")
    n = neural_network.start()
    app.run(debug=True)